from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.response_dataset import aggregate_response_directory
else:
    from .response_dataset import aggregate_response_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-sample FEM responses into response_dataset.npz.",
    )
    parser.add_argument(
        "--response-dir",
        default="data/fem_responses",
        help="Directory that contains sample_XXXX_response.npz files.",
    )
    parser.add_argument(
        "--output",
        default="data/response_dataset.npz",
        help="Output NPZ path for the aggregated response dataset.",
    )
    parser.add_argument(
        "--manifest",
        default="data/samples.csv",
        help="Optional manifest CSV to update with FEM success flags.",
    )
    args = parser.parse_args()

    aggregate_response_directory(
        response_dir=args.response_dir,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(f"Saved response dataset to {args.output}")


if __name__ == "__main__":
    main()
