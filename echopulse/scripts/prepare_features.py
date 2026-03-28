from __future__ import annotations

import argparse

from echopulse.config import load_config
from echopulse.dataset import build_feature_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tabular EchoPulse features from audio.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_csv", type=str, default="artifacts/features.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    df = build_feature_dataframe(args.data_dir, config)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved features to {args.output_csv}")


if __name__ == "__main__":
    main()
