from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from zeta_modelling.model_1.phase3_advanced_forecast import save_artifacts, train_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and persist Phase 3 forecast artifacts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO / "zeta_modelling" / "data_1" / "phase1_final_monthly_2024_2025.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO / "zeta_modelling" / "model_1" / "phase3_artifacts.joblib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    artifacts = train_artifacts(df)
    save_artifacts(artifacts, args.output)
    print(f"Saved artifacts to {args.output}")


if __name__ == "__main__":
    main()
