from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

UPSTREAM_DIR = Path(__file__).resolve().parents[2] / "evidence" / "inputs" / "upstream"
UPSTREAM_FILE = UPSTREAM_DIR / "logit_processor_w1.py"


def load_upstream_module() -> types.ModuleType:
    """Load the byte-exact vendored official Top-W logits processor.

    The vendored module imports ``transformers`` only to subclass
    ``LogitsProcessor``; the audited functions are pure NumPy. When the
    heavyweight runtime is absent a stub satisfies the import for the
    duration of module execution and is removed afterwards, so the
    process-wide module table is never polluted.
    """
    stubbed = "transformers" not in sys.modules
    if stubbed:
        stub = types.ModuleType("transformers")
        stub.LogitsProcessor = object
        sys.modules["transformers"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "top_w_upstream_reference", UPSTREAM_FILE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if stubbed:
            del sys.modules["transformers"]
    return module
