"""Record the adaptation go/no-go decision without training an adapter.

The plan treats parameter-efficient fine-tuning as optional. It must not block
the retrieval comparison. On a laptop, a 10-example adapter would not be a
credible test, so this phase writes a deferral with the measurements it used.
"""

from __future__ import annotations

import platform
import sys

from evidence_rag.config import OUTPUTS, ensure_dirs, load_config
from evidence_rag.utils import write_json


def decide() -> dict:
    ensure_dirs()
    config = load_config()
    try:
        import psutil  # optional, not required

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        ram_gb = None
    decision = {
        "decision": "defer",
        "date_rule": "Record the gate before any adapter training. Training did not run.",
        "model_considered": config["llm_model"],
        "reason": (
            "The comparison that the project needs is the same local model with and "
            "without retrieval. Ten development cases are too few to fit an adapter "
            "without memorizing them, and CPU fine-tuning is not assumed to be feasible. "
            "Held-out cases must stay out of any later training."
        ),
        "what_continues": "Untuned local model, lexical plus embedding retrieval, citation checks.",
        "hardware": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "ram_gb": ram_gb,
        },
        "license_note": "Confirm the model card license before redistribution. Weights are downloaded at runtime and are not vendored in this repo.",
    }
    write_json(OUTPUTS / "adaptation_decision.json", decision)
    return decision
