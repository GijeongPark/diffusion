from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def _infer_sample_count(data: np.lib.npyio.NpzFile) -> int:
    if "grf" in data.files:
        return int(np.asarray(data["grf"]).shape[0])
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim >= 1:
            return int(arr.shape[0])
    raise ValueError("Could not infer the sample dimension from the input dataset.")


def _parse_sample_ids(sample_ids: str, n_total: int, data: np.lib.npyio.NpzFile | None = None) -> np.ndarray:
    values = [int(item.strip()) for item in sample_ids.split(",") if item.strip()]
    if not values:
        raise ValueError("--sample-ids was provided but no valid ids were parsed.")

    if data is not None and "sample_id" in data.files:
        catalog = np.asarray(data["sample_id"], dtype=np.int64).reshape(-1)
        index_by_sample_id = {int(sample_id): int(idx) for idx, sample_id in enumerate(catalog.tolist())}
        if all(int(value) in index_by_sample_id for value in values):
            return np.asarray([index_by_sample_id[int(value)] for value in values], dtype=np.int32)

    sample_id_array = np.asarray(values, dtype=np.int32)
    if int(np.min(sample_id_array)) < 0 or int(np.max(sample_id_array)) >= n_total:
        raise ValueError(f"sample ids must match the dataset sample_id field or fall in [0, {n_total - 1}].")
    return sample_id_array


def _parse_source_indices(
    source_indices: str | Iterable[int] | np.ndarray,
    n_total: int,
) -> np.ndarray:
    if isinstance(source_indices, str):
        values = [int(item.strip()) for item in source_indices.split(",") if item.strip()]
    else:
        values = [int(value) for value in source_indices]
    if not values:
        raise ValueError("source_indices was provided but no valid indices were parsed.")
    selected = np.asarray(values, dtype=np.int32)
    if int(np.min(selected)) < 0 or int(np.max(selected)) >= n_total:
        raise ValueError(f"source indices must be in [0, {n_total - 1}].")
    return selected


def subset_unit_cell_dataset(
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    sample_ids: str = "",
    source_indices: str | Iterable[int] | np.ndarray | None = None,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    data = np.load(input_path, allow_pickle=True)
    n_total = _infer_sample_count(data)
    if source_indices is not None:
        selected = _parse_source_indices(source_indices, n_total)
    elif sample_ids.strip():
        selected = _parse_sample_ids(sample_ids, n_total, data=data)
    else:
        n_keep = n_total if limit is None else min(int(limit), n_total)
        selected = np.arange(n_keep, dtype=np.int32)

    subset: dict[str, np.ndarray] = {}
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim >= 1 and int(arr.shape[0]) == n_total:
            subset[key] = arr[selected]
        else:
            subset[key] = arr

    if "sample_id" in data.files:
        subset["source_sample_id"] = np.asarray(data["sample_id"], dtype=np.int32)[selected]
    else:
        subset["source_sample_id"] = np.asarray(selected, dtype=np.int32)
    subset["source_index"] = np.asarray(selected, dtype=np.int32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **subset)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a smaller unit-cell dataset NPZ for quick test runs.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Source unit-cell dataset NPZ.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output NPZ path for the subset dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Take the first N samples from the source dataset.",
    )
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Comma-separated sample ids to keep. Overrides --limit if provided.",
    )
    parser.add_argument(
        "--source-indices",
        default="",
        help="Comma-separated row indices to keep from the source dataset.",
    )
    args = parser.parse_args()

    output_path = subset_unit_cell_dataset(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        sample_ids=args.sample_ids,
        source_indices=args.source_indices or None,
    )
    print(f"Saved subset dataset to {output_path}")


if __name__ == "__main__":
    main()
