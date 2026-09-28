"""Run the controlled simulation, primary analyses, and sensitivity analyses."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    "01_synthetic.py",
    "02_beijing_main.py",
    "03_gas_main.py",
    "04_beijing_horizon_sensitivity.py",
    "05_correlation_threshold_sensitivity.py",
    "06_gas_model_sensitivity.py",
    "07_gas_shap_dependence_sensitivity.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()
    for name in EXPERIMENTS:
        command = [
            sys.executable,
            str(ROOT / "experiments" / name),
            "--permutations",
            str(args.permutations),
        ]
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
