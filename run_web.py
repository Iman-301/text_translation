from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))
    runpy.run_path(str(src_dir / "app.py"), run_name="__main__")


if __name__ == "__main__":
    main()

