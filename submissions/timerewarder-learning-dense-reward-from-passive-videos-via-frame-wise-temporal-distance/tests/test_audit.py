import ast
import hashlib
import json
from pathlib import Path

import pytest

from timerewarder_repro.audit import audit_sources


REVISION = "f54234b67bd3f1fa190f62498d38513a2140f23f"
MODEL_REVISION = "23eded140eb8c8d9f194243a115d218b5072d800"
DATASET_REVISION = "b966abcebc110dd97dd96018e395180e069756c4"
PAPER_REVISION = "arxiv:2509.26627v3"
PINNED_SOURCE_ROOT = Path(__file__).parents[1] / "artifacts/source/TimeRewarder"
SOURCE_PATHS = (
    "LICENSE",
    "training/datasets/build.py",
    "models/clip_withhead.py",
    "models/discretesupport.py",
    "RL/agent/rl_agent.py",
    "RL/train.py",
    "RL/replay_buffer.py",
    "RL/utils.py",
)
SOURCES = {path: (PINNED_SOURCE_ROOT / path).read_bytes() for path in SOURCE_PATHS}

AUDITED_METHODS = (
    ("training/datasets/build.py", "BaseDataset", "_distance_probs"),
    ("training/datasets/build.py", "BaseDataset", "_progress_from_sample_ids"),
    ("training/datasets/build.py", "BaseDataset", "prepare_train_frames"),
    ("training/datasets/build.py", "BaseDataset", "prepare_test_frames"),
    ("training/datasets/build.py", "VideoDataset", "load_annotations"),
    ("models/clip_withhead.py", "CLIPwithHead", "_regression_logits"),
    ("models/clip_withhead.py", "CLIPwithHead", "predict_progress"),
    ("models/discretesupport.py", None, "transform_one"),
    ("models/discretesupport.py", "DiscreteSupport", "vector_to_scalar"),
    ("models/discretesupport.py", None, "slice_regression_logits"),
    ("RL/agent/rl_agent.py", "RLAgent", "update_critic"),
    ("RL/agent/rl_agent.py", "RLAgent", "update"),
    ("RL/agent/rl_agent.py", "RLAgent", "clip_rewarder"),
    ("RL/train.py", "WorkspaceIL", "_log_timerewarder_reward_curves"),
    ("RL/train.py", "WorkspaceIL", "setup"),
    ("RL/train.py", "WorkspaceIL", "replay_iter"),
    ("RL/train.py", "WorkspaceIL", "train_il"),
    ("RL/replay_buffer.py", None, "episode_len"),
    ("RL/replay_buffer.py", None, "save_episode"),
    ("RL/replay_buffer.py", None, "load_episode"),
    ("RL/replay_buffer.py", "ReplayBufferStorage", "__init__"),
    ("RL/replay_buffer.py", "ReplayBufferStorage", "add"),
    ("RL/replay_buffer.py", "ReplayBufferStorage", "_preload"),
    ("RL/replay_buffer.py", "ReplayBufferStorage", "_store_episode"),
    ("RL/replay_buffer.py", "ReplayBuffer", "__init__"),
    ("RL/replay_buffer.py", "ReplayBuffer", "_sample_episode"),
    ("RL/replay_buffer.py", "ReplayBuffer", "_store_episode"),
    ("RL/replay_buffer.py", "ReplayBuffer", "_try_fetch"),
    ("RL/replay_buffer.py", "ReplayBuffer", "_sample"),
    ("RL/replay_buffer.py", "ReplayBuffer", "__iter__"),
    ("RL/replay_buffer.py", None, "_worker_init_fn"),
    ("RL/replay_buffer.py", None, "make_replay_loader"),
    ("RL/utils.py", None, "to_torch"),
)


@pytest.fixture
def source_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    records = []
    for upstream_path, payload in SOURCES.items():
        path = f"TimeRewarder/{upstream_path}"
        destination = source_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        records.append(
            {
                "repository": "CowAndSheep/TimeRewarder",
                "url": "https://example.invalid/TimeRewarder.git",
                "revision": REVISION,
                "path": path,
                "upstream_path": upstream_path,
                "git_blob": hashlib.sha1(payload).hexdigest(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
                "license": "MIT",
            }
        )

    manifest = {
        "paper": {"revision": PAPER_REVISION},
        "model": {"revision": MODEL_REVISION},
        "dataset": {"revision": DATASET_REVISION},
        "sources": records,
    }
    receipt = {
        "commands": command_receipts(records),
        "sources": records,
    }
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "receipt.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return manifest_path, receipt_path, source_root


def test_audit_identifies_action_free_temporal_labels_and_inverse_sampling(
    source_fixture: tuple[Path, Path, Path],
) -> None:
    result = audit_sources(*source_fixture)
    assert result["annotation_fields"] == ["filename", "label"]
    assert result["action_sequence_consumed"] is False
    assert result["action_sequence_scope"] == "annotation and progress-label path"
    assert result["progress_expression"] == (
        "frame_inds[sample_id] / progress_max_frames"
    )
    assert result["distance_sampling"] == "1 / distance**power"


def test_audit_identifies_bidirectional_transition_reward(
    source_fixture: tuple[Path, Path, Path],
) -> None:
    result = audit_sources(*source_fixture)
    assert result["pairing"] == ["predecessor,current", "current,predecessor"]
    assert result["dense_reward"] == {
        "use_bin": "decode(forward) - decode(reverse)",
        "use_bin_false": "forward logit - reverse logit",
    }
    assert result["first_frame"] == "self-pair"
    assert result["cumulative_value_use"] == "visualization"
    assert result["success_term"] == {
        "when_enabled": "added to per-transition replay reward",
        "pre_success_output": "raw_reward",
    }
    assert result["replay_uses_per_transition_reward"] is True
    assert result["replay_update_path"] == (
        "stored transition reward sampled and batched into update_critic"
    )
    assert len(result["function_span_sha256"]) == len(AUDITED_METHODS)
    assert result["observation_source_refs"]["replay_uses_per_transition_reward"] == [
        "TimeRewarder/RL/agent/rl_agent.py",
        "TimeRewarder/RL/train.py",
        "TimeRewarder/RL/replay_buffer.py",
    ]
    assert result["observation_source_refs"]["replay_update_path"] == [
        "TimeRewarder/RL/train.py",
        "TimeRewarder/RL/replay_buffer.py",
        "TimeRewarder/RL/utils.py",
        "TimeRewarder/RL/agent/rl_agent.py",
    ]


def test_audit_rejects_changed_reward_expression(
    source_fixture: tuple[Path, Path, Path],
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    replace_verified_source(
        manifest_path,
        receipt_path,
        source_root,
        "models/clip_withhead.py",
        "reward = scores[:n] - scores[n:]",
        "reward = scores[:n] + scores[n:]",
    )

    with pytest.raises(ValueError, match="predict_progress"):
        audit_sources(manifest_path, receipt_path, source_root)


def test_audit_rejects_changed_annotation_values(
    source_fixture: tuple[Path, Path, Path],
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    replace_verified_source(
        manifest_path,
        receipt_path,
        source_root,
        "training/datasets/build.py",
        "video_infos.append(dict(filename=filename, label=label))",
        "video_infos.append(dict(filename=label, label=filename))",
    )

    with pytest.raises(ValueError, match="load_annotations"):
        audit_sources(manifest_path, receipt_path, source_root)


@pytest.mark.parametrize(
    ("upstream_path", "old", "new", "error"),
    [
        (
            "training/datasets/build.py",
            "distance_probs = 1.0 / (possible_distances ** self.weightedsample_distance_power)",
            "distance_probs = possible_distances\n        return distance_probs\n        distance_probs = 1.0 / (possible_distances ** self.weightedsample_distance_power)",
            "_distance_probs",
        ),
        (
            "training/datasets/build.py",
            "return frame_inds[sample_id].float() / self.progress_max_frames",
            "return action_sequence[sample_id]\n        return frame_inds[sample_id].float() / self.progress_max_frames",
            "_progress_from_sample_ids",
        ),
        (
            "models/clip_withhead.py",
            "reward = scores[:n] - scores[n:]",
            "reward = scores[:n] + scores[n:]\n        def decoy():\n            reward = scores[:n] - scores[n:]",
            "predict_progress",
        ),
        (
            "models/clip_withhead.py",
            "prev_features[0] = video_features[0]",
            "prev_features[0] = video_features[-1]\n        return video_features\n        prev_features[0] = video_features[0]",
            "predict_progress",
        ),
        (
            "models/clip_withhead.py",
            "scores = self.discrete_support.vector_to_scalar(logits) if self.use_bin else logits",
            "scores = self.discrete_support.vector_to_scalar(logits)\n        if False:\n            scores = self.discrete_support.vector_to_scalar(logits) if self.use_bin else logits",
            "predict_progress",
        ),
        (
            "models/clip_withhead.py",
            "reward = scores[:n] - scores[n:]",
            "reward = scores[:n] + scores[n:]\n        if 0:\n            reward = scores[:n] - scores[n:]",
            "predict_progress",
        ),
        (
            "models/discretesupport.py",
            "value = sign * abs_value",
            "value = abs_value\n        return value\n        value = sign * abs_value",
            "vector_to_scalar",
        ),
        (
            "RL/agent/rl_agent.py",
            "return reward, value, raw_reward, self.suc_signal_scale * self.suc_scale",
            "reward += value\n        return reward, value, raw_reward, self.suc_signal_scale * self.suc_scale",
            "clip_rewarder",
        ),
        (
            "RL/agent/rl_agent.py",
            "reward = self.cost_encoder.predict_progress(obs, self.text_feature)",
            "if 0:\n                reward = self.cost_encoder.predict_progress(obs, self.text_feature)",
            "clip_rewarder",
        ),
        (
            "RL/agent/rl_agent.py",
            "reward += goal_achieved * self.suc_scale * self.suc_signal_scale",
            "reward += goal_achieved * self.suc_scale * self.suc_signal_scale * 2\n            if False:\n                reward += goal_achieved * self.suc_scale * self.suc_signal_scale",
            "clip_rewarder",
        ),
        (
            "training/datasets/build.py",
            "filename = parts[0]",
            "filename = parts[0]\n                action_sequence = parts[2:]",
            "load_annotations",
        ),
        (
            "training/datasets/build.py",
            "filename = parts[0]",
            "filename = parts[0]\n                consume(action_sequence)",
            "load_annotations",
        ),
        (
            "models/clip_withhead.py",
            "scores = self.discrete_support.vector_to_scalar(logits) if self.use_bin else logits",
            "logits = logits * 2\n        scores = self.discrete_support.vector_to_scalar(logits) if self.use_bin else logits",
            "predict_progress",
        ),
        (
            "RL/train.py",
            "new_rewards = new_rewards_clip",
            "new_rewards = value_clip\n            if False:\n                new_rewards = new_rewards_clip",
            "train_il",
        ),
        (
            "RL/train.py",
            "new_rewards = new_rewards_clip",
            "new_rewards = new_rewards_clip\n            new_rewards = value_clip",
            "train_il",
        ),
        (
            "RL/train.py",
            "elt = elt._replace(reward=float(new_rewards[i - 1]))",
            "elt = elt._replace(reward=float(value_clip[i - 1]))\n                def decoy():\n                    elt = elt._replace(reward=float(new_rewards[i - 1]))",
            "train_il",
        ),
        (
            "RL/train.py",
            "self.replay_storage.add(elt)",
            "if maybe_store:\n                self.replay_storage.add(elt)",
            "train_il",
        ),
        (
            "RL/train.py",
            "self.replay_storage.add(elt)",
            "if True:\n                return\n            self.replay_storage.add(elt)",
            "train_il",
        ),
        (
            "RL/train.py",
            "plt.plot(value_clip)",
            "plt.plot(new_rewards_clip)\n        if False:\n            plt.plot(value_clip)",
            "_log_timerewarder_reward_curves",
        ),
        (
            "RL/train.py",
            "elt = elt._replace(reward=float(new_rewards[i - 1]))",
            "elt = elt._replace(reward=float(new_rewards[i - 1]))\n                elt = elt._replace(reward=float(value_clip[i - 1]))",
            "train_il",
        ),
    ],
)
def test_audit_rejects_inactive_decoys_and_unexpected_semantics(
    source_fixture: tuple[Path, Path, Path],
    upstream_path: str,
    old: str,
    new: str,
    error: str,
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    replace_verified_source(
        manifest_path, receipt_path, source_root, upstream_path, old, new
    )

    with pytest.raises(ValueError, match=error):
        audit_sources(manifest_path, receipt_path, source_root)


@pytest.mark.parametrize("missing", list(SOURCES))
def test_audit_rejects_each_missing_required_source(
    source_fixture: tuple[Path, Path, Path], missing: str
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    remove_verified_source(manifest_path, receipt_path, source_root, missing)

    with pytest.raises(ValueError, match="required audit sources"):
        audit_sources(manifest_path, receipt_path, source_root)


@pytest.mark.parametrize(
    ("upstream_path", "class_name", "method_name"), AUDITED_METHODS
)
def test_audit_rejects_one_byte_change_in_every_approved_method(
    source_fixture: tuple[Path, Path, Path],
    upstream_path: str,
    class_name: str | None,
    method_name: str,
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    add_trailing_space_to_method_definition(
        manifest_path,
        receipt_path,
        source_root,
        upstream_path,
        class_name,
        method_name,
    )

    with pytest.raises(ValueError, match="span hash"):
        audit_sources(manifest_path, receipt_path, source_root)


@pytest.mark.parametrize(
    ("upstream_path", "old", "new"),
    [
        (
            "RL/agent/rl_agent.py",
            "obs = obs.detach()\n        obs = obs[:, -3:]",
            "obs = obs[:, -3:]\n        obs = obs.detach()",
        ),
        (
            "training/datasets/build.py",
            "results = copy.deepcopy(self.video_infos[idx])",
            "return self.video_infos[idx]\n        results = copy.deepcopy(self.video_infos[idx])",
        ),
        (
            "training/datasets/build.py",
            "filename = parts[0]",
            "filename = parts[0]\n                consume(self.action_sequence)",
        ),
        (
            "models/discretesupport.py",
            "4 * epsilon * (torch.abs(value) + 1 + epsilon)",
            "5 * epsilon * (torch.abs(value) + 1 + epsilon)",
        ),
        (
            "RL/train.py",
            "self._save_train_episode_video(new_rewards, env_rewards)",
            "new_rewards = value_clip\n                    self._save_train_episode_video(new_rewards, env_rewards)",
        ),
        (
            "RL/agent/rl_agent.py",
            "return reward, value, raw_reward, self.suc_signal_scale * self.suc_scale",
            "consume(value)\n        return reward, value, raw_reward, self.suc_signal_scale * self.suc_scale",
        ),
    ],
    ids=(
        "reordering",
        "early-return",
        "attribute-action-use",
        "decoder-intermediate",
        "outer-overwrite",
        "extra-value-use",
    ),
)
def test_audit_rejects_each_previously_uncovered_escape_class(
    source_fixture: tuple[Path, Path, Path],
    upstream_path: str,
    old: str,
    new: str,
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    replace_verified_source(
        manifest_path, receipt_path, source_root, upstream_path, old, new
    )

    with pytest.raises(ValueError, match="span hash"):
        audit_sources(manifest_path, receipt_path, source_root)


def test_audit_rejects_crlf_conversion_without_newline_normalization(
    source_fixture: tuple[Path, Path, Path],
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    upstream_path = "models/clip_withhead.py"
    source_path = source_root / "TimeRewarder" / upstream_path
    payload = source_path.read_bytes()
    assert b"\r\n" not in payload
    source_path.write_bytes(payload.replace(b"\n", b"\r\n"))
    refresh_verified_source(manifest_path, receipt_path, source_path, upstream_path)

    with pytest.raises(ValueError, match="span hash"):
        audit_sources(manifest_path, receipt_path, source_root)


@pytest.mark.parametrize(
    ("old", "new", "function_name"),
    [
        (
            "return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1) + 0.001 * x",
            "return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1) + 0.002 * x",
            "transform_one",
        ),
        (
            "return logits[..., LEGACY_ORDER_CLS_DIM:]",
            "return logits[..., LEGACY_ORDER_CLS_DIM + 1:]",
            "slice_regression_logits",
        ),
    ],
)
def test_audit_rejects_semantic_helper_mutation(
    source_fixture: tuple[Path, Path, Path],
    old: str,
    new: str,
    function_name: str,
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    replace_verified_source(
        manifest_path,
        receipt_path,
        source_root,
        "models/discretesupport.py",
        old,
        new,
    )

    with pytest.raises(ValueError, match=function_name):
        audit_sources(manifest_path, receipt_path, source_root)


@pytest.mark.parametrize("missing", ["RL/replay_buffer.py", "RL/utils.py"])
def test_audit_requires_concrete_replay_provenance_sources(
    source_fixture: tuple[Path, Path, Path], missing: str
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    remove_verified_source(manifest_path, receipt_path, source_root, missing)

    with pytest.raises(ValueError, match=missing):
        audit_sources(manifest_path, receipt_path, source_root)


@pytest.mark.parametrize(
    ("upstream_path", "old", "new", "function_name"),
    [
        (
            "RL/replay_buffer.py",
            "value = time_step[spec.name]",
            "value = 0 if spec.name == 'reward' else time_step[spec.name]",
            "ReplayBufferStorage.add",
        ),
        (
            "RL/replay_buffer.py",
            "step_reward = episode['reward'][idx + i]",
            "step_reward = episode['action'][idx + i]",
            "ReplayBuffer._sample",
        ),
        (
            "RL/replay_buffer.py",
            "yield self._sample()",
            "yield self._sample_episode()",
            "ReplayBuffer.__iter__",
        ),
        (
            "RL/utils.py",
            "return tuple(torch.as_tensor(x, device=device) for x in xs)",
            "return tuple(torch.as_tensor(x, device=device) for x in reversed(xs))",
            "to_torch",
        ),
    ],
)
def test_audit_rejects_concrete_replay_path_mutation(
    source_fixture: tuple[Path, Path, Path],
    upstream_path: str,
    old: str,
    new: str,
    function_name: str,
) -> None:
    manifest_path, receipt_path, source_root = source_fixture
    replace_verified_source(
        manifest_path, receipt_path, source_root, upstream_path, old, new
    )

    with pytest.raises(ValueError, match=function_name):
        audit_sources(manifest_path, receipt_path, source_root)


def command_receipts(records: list[dict[str, object]]) -> list[dict[str, object]]:
    commands = [
        ["git", "clone", "--no-checkout", records[0]["url"], "<checkout>"],
        ["git", "-C", "<checkout>", "checkout", "--detach", REVISION],
        ["git", "-C", "<checkout>", "rev-parse", "HEAD"],
        ["git", "-C", "<checkout>", "status", "--porcelain"],
        ["git", "-C", "<checkout>", "fsck", "--full"],
    ]
    for record in records:
        commands.extend(
            [
                [
                    "git",
                    "-C",
                    "<checkout>",
                    "rev-parse",
                    f"{REVISION}:{record['upstream_path']}",
                ],
                [
                    "git",
                    "-C",
                    "<checkout>",
                    "show",
                    f"{REVISION}:{record['upstream_path']}",
                ],
            ]
        )
    return [{"command": command, "status": 0} for command in commands]


def remove_verified_source(
    manifest_path: Path,
    receipt_path: Path,
    source_root: Path,
    upstream_path: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest["sources"] = [
        item for item in manifest["sources"] if item["upstream_path"] != upstream_path
    ]
    receipt["sources"] = [
        item for item in receipt["sources"] if item["upstream_path"] != upstream_path
    ]
    receipt["commands"] = command_receipts(receipt["sources"])
    path = source_root / "TimeRewarder" / upstream_path
    path.unlink()
    for parent in path.parents:
        if parent == source_root or any(parent.iterdir()):
            break
        parent.rmdir()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def replace_verified_source(
    manifest_path: Path,
    receipt_path: Path,
    source_root: Path,
    upstream_path: str,
    old: str,
    new: str,
) -> None:
    source_path = source_root / "TimeRewarder" / upstream_path
    text = source_path.read_text(encoding="utf-8")
    if old not in text:
        raise AssertionError(f"mutation target not found: {old}")
    source_path.write_text(
        text.replace(old, new),
        encoding="utf-8",
    )
    refresh_verified_source(manifest_path, receipt_path, source_path, upstream_path)


def add_trailing_space_to_method_definition(
    manifest_path: Path,
    receipt_path: Path,
    source_root: Path,
    upstream_path: str,
    class_name: str,
    method_name: str,
) -> None:
    source_path = source_root / "TimeRewarder" / upstream_path
    text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    scope = tree.body
    if class_name is not None:
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        scope = class_node.body
    method_node = next(
        node
        for node in scope
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    lines = text.splitlines(keepends=True)
    definition = lines[method_node.lineno - 1]
    ending = "\n" if definition.endswith("\n") else ""
    lines[method_node.lineno - 1] = definition.removesuffix("\n") + " " + ending
    source_path.write_text("".join(lines), encoding="utf-8")
    refresh_verified_source(manifest_path, receipt_path, source_path, upstream_path)


def refresh_verified_source(
    manifest_path: Path,
    receipt_path: Path,
    source_path: Path,
    upstream_path: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload = source_path.read_bytes()
    for document in (manifest, receipt):
        record = next(
            item
            for item in document["sources"]
            if item["upstream_path"] == upstream_path
        )
        record["sha256"] = hashlib.sha256(payload).hexdigest()
        record["byte_size"] = len(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
