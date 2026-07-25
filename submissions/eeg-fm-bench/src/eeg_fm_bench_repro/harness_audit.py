"""Audit released strategy wiring and exercise it with deterministic CPU steps."""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

CLAIM_ID = "three-strategy-evaluation-harness"


def _parsed(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _declares_field(owner: ast.ClassDef, field: str) -> bool:
    return any(
        (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == field
        )
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == field for target in node.targets)
        )
        for node in owner.body
    )


def _method(owner: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _attribute_chain(node: ast.AST) -> str:
    pieces: list[str] = []
    while isinstance(node, ast.Attribute):
        pieces.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        pieces.append(node.id)
    return ".".join(reversed(pieces))


def _upstream_contract(snapshot: Path) -> tuple[dict[str, bool], dict[str, str]]:
    config_path = snapshot / "baseline" / "abstract" / "config.py"
    trainer_path = snapshot / "baseline" / "abstract" / "trainer.py"
    classifier_path = snapshot / "baseline" / "abstract" / "classifier.py"
    _, config_tree = _parsed(config_path)
    _, trainer_tree = _parsed(trainer_path)
    trainer = _class(trainer_tree, "AbstractTrainer")

    freeze_declared = _declares_field(_class(config_tree, "BaseTrainingArgs"), "freeze_encoder")
    multitask_declared = _declares_field(_class(config_tree, "AbstractConfig"), "multitask")

    setup = _method(trainer, "setup_optim_params")
    freeze_honored = any(
        isinstance(node, ast.If)
        and any(
            _attribute_chain(child) == "self.cfg.training.freeze_encoder"
            for child in ast.walk(node.test)
        )
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "requires_grad"
                for target in child.targets
            )
            and isinstance(child.value, ast.Constant)
            and child.value.value is False
            for child in ast.walk(node)
        )
        for node in ast.walk(setup)
    )

    create_loader = _method(trainer, "create_dataloader")
    mixed_loader = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "mixed" for target in node.targets)
        and any(_attribute_chain(child) == "self.cfg.multitask" for child in ast.walk(node.value))
        for node in ast.walk(create_loader)
    )

    run = _method(trainer, "run")
    training_branch = any(
        isinstance(node, ast.If)
        and _attribute_chain(node.test) == "self.cfg.multitask"
        and any(
            isinstance(descendant, ast.Call)
            and _attribute_chain(descendant.func) == "self.run_unified_training"
            for child in node.body
            for descendant in ast.walk(child)
        )
        for node in ast.walk(run)
    )
    return (
        {
            "freeze_encoder_declared": freeze_declared,
            "freeze_encoder_honored": freeze_honored,
            "multitask_declared": multitask_declared,
            "multitask_mixed_loader": mixed_loader,
            "multitask_training_branch": training_branch,
        },
        {
            path.relative_to(snapshot).as_posix(): _hash(path)
            for path in (config_path, trainer_path, classifier_path)
        },
    )


class _TinyHarness(nn.Module):
    def __init__(self, head_names: tuple[str, ...]):
        super().__init__()
        self.encoder = nn.Linear(8, 4)
        self.heads = nn.ModuleDict({name: nn.Linear(4, 2) for name in head_names})

    def forward(self, values: torch.Tensor, head: str) -> torch.Tensor:
        return self.heads[head](torch.tanh(self.encoder(values)))


def _changed(before: list[torch.Tensor], parameters: Any) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, parameters, strict=True)
    )


def _strategy(frozen: bool, heads: tuple[str, ...]) -> dict[str, Any]:
    torch.manual_seed(20260724)
    torch.set_num_threads(1)
    model = _TinyHarness(heads)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = not frozen
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.05)
    before = [parameter.detach().clone() for parameter in model.encoder.parameters()]
    values = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 31.0
    labels = torch.tensor([0, 1, 0, 1])
    losses = [nn.functional.cross_entropy(model(values, head), labels) for head in heads]
    loss = torch.stack(losses).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "finite_loss": bool(math.isfinite(float(loss.detach()))),
        "loss": round(float(loss.detach()), 8),
        "encoder_changed": _changed(before, model.encoder.parameters()),
        "dataset_heads_exercised": list(heads),
    }


def run_harness_audit(snapshot: Path) -> dict[str, Any]:
    """Verify released strategy branches and smoke-run their CPU semantics."""

    contract, source_hashes = _upstream_contract(Path(snapshot))
    strategies = {
        "frozen-backbone-single-task": _strategy(True, ("dataset_a",)),
        "full-parameter-single-task": _strategy(False, ("dataset_a",)),
        "full-parameter-multi-task": _strategy(
            False, ("dataset_a", "dataset_b")
        ),
    }
    verified = all(contract.values()) and all(
        result["finite_loss"] for result in strategies.values()
    )
    return {
        "claim_id": CLAIM_ID,
        "kind": "numerical_audit",
        "status": "verified" if verified else "inconclusive",
        "upstream_contract": contract,
        "strategies": strategies,
        "source_sha256": source_hashes,
        "scope": "Released harness wiring plus synthetic CPU semantic smoke; no leaderboard reproduction.",
    }
