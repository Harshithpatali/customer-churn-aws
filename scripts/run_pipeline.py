from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    run("train.py")
    run("statistical_analysis.py")
    run("explainability.py")
