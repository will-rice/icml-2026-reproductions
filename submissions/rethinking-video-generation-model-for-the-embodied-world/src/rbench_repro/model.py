from hashlib import sha256
import json


CLAIMS: tuple[str, str, str] = (
    "RBench evaluates robot-oriented video generation across five task domains and four embodiments with task-level and visual-fidelity sub-metrics (Figure 1).",
    "The benchmark evaluates 25 open-source, commercial, and robotics-specific video models across task and embodiment dimensions (Table 2).",
    "RBench captures robotic video failure modes including structural distortion, floating components, and key-action omission (Figure 2).",
)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()
