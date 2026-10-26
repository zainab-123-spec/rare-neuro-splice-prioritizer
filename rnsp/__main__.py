"""Allow ``python -m rnsp``."""

from __future__ import annotations

from rnsp.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
