"""Three-way agreement report for the dictionary content-analysis validation.

Run after run_06 and the blind labelling passes:
    python verification_agreement.py

Compares the dictionary labels (verification_sample.csv) against the blind
automated re-labelling (verification_auto.csv) and, when present, the author's
blind read (verification_human.csv). Prints exact-match and per-label
agreement/kappa; writes verification_report.csv and
verification_disagreements.csv next to the inputs. If the human file is not
there yet, reports dictionary-vs-auto only.
"""

from __future__ import annotations

import logging

from ksai import config
from ksai.verify import run_report

CONTENT = config.RESULTS / "content"  # committed: the safe agreement report
CONTENT_PRIVATE = config.RESULTS_PRIVATE / "content"  # gitignored: blind inputs + disagreements

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_report(
        sample_csv=CONTENT_PRIVATE / "verification_sample.csv",
        auto_csv=CONTENT_PRIVATE / "verification_auto.csv",
        human_csv=CONTENT_PRIVATE / "verification_human.csv",
        out_dir=CONTENT,
        private_dir=CONTENT_PRIVATE,
    )
