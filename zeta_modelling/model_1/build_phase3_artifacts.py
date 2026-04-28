from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.phase3_advanced_forecast import save_artifacts, train_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and persist Phase 3 forecast artifacts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("zeta_modelling/data/phase1_final_monthly_2024_2025.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("zeta_modelling/model/phase3_artifacts.joblib"),
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
