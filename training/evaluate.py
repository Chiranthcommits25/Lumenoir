"""Thin wrapper so evaluation is launched from the training tree as well.

    python -m training.evaluate --predictions out/ --reference gt/
"""

from __future__ import annotations

from backend.evaluation.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
