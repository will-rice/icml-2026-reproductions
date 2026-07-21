"""Persistent state for the ICML reproduction loop."""

import argparse
import copy
import json
import math
import os
import re
import tempfile
from pathlib import Path


PHASES = {
    "idle",
    "selected",
    "design-pending",
    "implementing",
    "validated",
    "deployed",
    "submitted",
    "judging",
    "improving",
    "complete",
    "blocked",
}
STATE_KEYS = {"version", "phase", "current", "history", "total_api_cost_usd"}
IMMUTABLE_PAPER_FIELDS = {"paper_id", "title", "slug", "project_path"}
PAPER_COST_FIELDS = {"estimated_api_cost_usd", "actual_api_cost_usd"}
SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
ALLOWED = {
    "idle": {"selected"},
    "selected": {"design-pending", "blocked"},
    "design-pending": {"implementing", "blocked"},
    "implementing": {"validated", "blocked"},
    "validated": {"deployed", "blocked"},
    "deployed": {"submitted", "blocked"},
    "submitted": {"judging", "blocked"},
    "judging": {"improving", "complete", "blocked"},
    "improving": {"validated", "blocked"},
    "complete": {"idle"},
    "blocked": {"idle"},
}


def main() -> None:
    """Run the state management command-line interface."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "show"):
        subparser = commands.add_parser(command)
        subparser.add_argument("path", type=Path)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("path", type=Path)
    select_parser.add_argument("paper_json")
    transition_parser = commands.add_parser("transition")
    transition_parser.add_argument("path", type=Path)
    transition_parser.add_argument("phase", choices=sorted(PHASES))
    transition_parser.add_argument("updates_json")
    arguments = parser.parse_args()

    if arguments.command == "init":
        if arguments.path.exists():
            raise FileExistsError(arguments.path)
        state = new_state()
        save_state(arguments.path, state)
    elif arguments.command == "show":
        state = load_state(arguments.path)
    elif arguments.command == "select":
        state = select_paper(load_state(arguments.path), json.loads(arguments.paper_json))
        save_state(arguments.path, state)
    else:
        state = transition(
            load_state(arguments.path),
            arguments.phase,
            **json.loads(arguments.updates_json),
        )
        save_state(arguments.path, state)
    print(json.dumps(state, indent=2, sort_keys=True))


def new_state() -> dict:
    """Return an empty reproduction loop state."""
    return {
        "version": 1,
        "phase": "idle",
        "current": None,
        "history": [],
        "total_api_cost_usd": 0.0,
    }


def load_state(path: Path) -> dict:
    """Load and validate state from a JSON file."""
    with path.open(encoding="utf-8") as file:
        state = json.load(file)
    validate_state(state)
    return state


def save_state(path: Path, state: dict) -> None:
    """Validate and atomically save state as JSON."""
    validate_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(state, file, allow_nan=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def select_paper(state: dict, paper: dict) -> dict:
    """Select a previously uncompleted paper for the loop."""
    validate_state(state)
    if state["phase"] != "idle":
        raise ValueError("phase")
    if not isinstance(paper, dict):
        raise ValueError("paper")
    for field in ("paper_id", "title", "slug"):
        if not paper.get(field):
            raise ValueError(field)
    if any(
        isinstance(completed, dict)
        and completed.get("paper_id") == paper["paper_id"]
        for completed in state["history"]
    ):
        raise ValueError("paper_id")

    current = copy.deepcopy(paper)
    current.setdefault("estimated_api_cost_usd", 0.0)
    current["project_path"] = f"submissions/{current['slug']}"
    validate_paper_costs(current)
    if any(
        completed.get("project_path") == current["project_path"]
        for completed in state["history"]
    ):
        raise ValueError("project_path")

    selected = copy.deepcopy(state)
    selected["phase"] = "selected"
    selected["current"] = current
    validate_state(selected)
    return selected


def transition(state: dict, phase: str, **updates: object) -> dict:
    """Return a copied state after a valid phase transition."""
    validate_state(state)
    if (
        type(phase) is not str
        or phase not in PHASES
        or phase not in ALLOWED[state["phase"]]
    ):
        raise ValueError("phase")
    if state["phase"] == "idle":
        return select_paper(state, updates)

    transitioned = copy.deepcopy(state)
    current = transitioned["current"]
    for field in IMMUTABLE_PAPER_FIELDS:
        if field in updates and updates[field] != current.get(field):
            raise ValueError(field)
    if (
        "estimated_api_cost_usd" in updates
        and updates["estimated_api_cost_usd"] != current.get("estimated_api_cost_usd")
    ):
        raise ValueError("estimated_api_cost_usd")
    if (
        "space_id" in updates
        and "space_id" in current
        and updates["space_id"] != current["space_id"]
    ):
        raise ValueError("space_id")
    current.update(updates)
    validate_paper_costs(current)
    if (
        "actual_api_cost_usd" in updates
        and "actual_api_cost_usd" in state["current"]
        and current["actual_api_cost_usd"] < state["current"]["actual_api_cost_usd"]
    ):
        raise ValueError("actual_api_cost_usd")

    if phase == "implementing" and updates.get("design_approved") is not True:
        raise ValueError("design_approved")
    if phase == "deployed" and not updates.get("deployed_sha"):
        raise ValueError("deployed_sha")
    if phase == "submitted" and not updates.get("space_id"):
        raise ValueError("space_id")
    if phase == "complete" and not updates.get("verdict"):
        raise ValueError("verdict")
    if phase == "submitted" and any(
        completed.get("space_id") == current["space_id"]
        for completed in transitioned["history"]
    ):
        raise ValueError("space_id")

    if phase == "idle":
        transitioned["history"].append(copy.deepcopy(current))
        transitioned["total_api_cost_usd"] += current.get("actual_api_cost_usd", 0.0)
        transitioned["current"] = None
    transitioned["phase"] = phase
    validate_state(transitioned)
    return transitioned


def validate_state(state: dict) -> None:
    """Raise ValueError when state does not satisfy the persisted schema."""
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise ValueError("keys")
    if type(state["version"]) is not int or state["version"] != 1:
        raise ValueError("version")
    if type(state["phase"]) is not str or state["phase"] not in PHASES:
        raise ValueError("phase")
    validate_cost(state["total_api_cost_usd"], "total_api_cost_usd")
    if not isinstance(state["history"], list):
        raise ValueError("history")
    project_paths = set()
    space_ids = set()
    for completed in state["history"]:
        if not isinstance(completed, dict):
            raise ValueError("history")
        validate_paper_record(completed, project_paths, space_ids)
    if state["phase"] == "idle" and state["current"] is not None:
        raise ValueError("current")
    if state["phase"] != "idle" and not isinstance(state["current"], dict):
        raise ValueError("current")
    if state["current"] is not None:
        validate_paper_record(state["current"], project_paths, space_ids)


def validate_cost(value: object, field: str) -> None:
    """Raise ValueError when a cost is not a nonnegative number."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(field)


def validate_paper_costs(paper: dict) -> None:
    """Raise ValueError when a per-paper cost is outside its allowed range."""
    for field in PAPER_COST_FIELDS:
        if field in paper:
            validate_cost(paper[field], field)
            if paper[field] > 10.0:
                raise ValueError(field)


def validate_paper_record(
    paper: dict, project_paths: set[str], space_ids: set[str]
) -> None:
    """Validate persistent paper identity and cost invariants."""
    validate_paper_costs(paper)
    if type(paper.get("slug")) is not str or not SLUG_PATTERN.fullmatch(paper["slug"]):
        raise ValueError("slug")
    if (
        type(paper.get("project_path")) is not str
        or paper["project_path"] != f"submissions/{paper['slug']}"
    ):
        raise ValueError("project_path")
    if paper["project_path"] in project_paths:
        raise ValueError("project_path")
    project_paths.add(paper["project_path"])
    if "space_id" in paper:
        if type(paper["space_id"]) is not str or not paper["space_id"]:
            raise ValueError("space_id")
        if paper["space_id"] in space_ids:
            raise ValueError("space_id")
        space_ids.add(paper["space_id"])


if __name__ == "__main__":
    main()
