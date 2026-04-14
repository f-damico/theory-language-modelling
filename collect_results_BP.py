#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

FILENAME_RE = re.compile(r"^(local|global)__sr(\d+)__ss(\d+)\.npz$")


def _load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        out: Dict[str, Any] = {}
        for k in data.files:
            v = data[k]
            if v.shape == ():
                out[k] = v.item()
            else:
                out[k] = v.copy()
        return out


def _parse_params(params_json: Any) -> Dict[str, Any]:
    if params_json is None:
        return {}
    if isinstance(params_json, bytes):
        params_json = params_json.decode("utf-8")
    if isinstance(params_json, np.ndarray):
        params_json = params_json.item()
    if isinstance(params_json, str):
        return json.loads(params_json)
    return dict(params_json)


def _discover_files(run_dir: Path) -> List[Tuple[Path, str, int, int]]:
    found: List[Tuple[Path, str, int, int]] = []
    for path in sorted(run_dir.glob("*.npz")):
        m = FILENAME_RE.match(path.name)
        if m is None:
            continue
        mode = m.group(1)
        seed_rules = int(m.group(2))
        seed_sample = int(m.group(3))
        found.append((path, mode, seed_rules, seed_sample))
    return found


def _stack_metric(files_matrix: List[List[Dict[str, Any] | None]], key: str) -> np.ndarray:
    sample = None
    for row in files_matrix:
        for item in row:
            if item is not None and key in item:
                sample = np.asarray(item[key])
                break
        if sample is not None:
            break
    if sample is None:
        raise KeyError(key)

    out_shape = (len(files_matrix), len(files_matrix[0])) + sample.shape
    out = np.full(out_shape, np.nan, dtype=np.float64)
    for i, row in enumerate(files_matrix):
        for j, item in enumerate(row):
            if item is None or key not in item:
                continue
            arr = np.asarray(item[key], dtype=np.float64)
            out[i, j] = arr
    return out


def _aggregate_mode(items: List[Tuple[Path, int, int]]) -> Dict[str, Any]:
    seed_rules_values = sorted({sr for _, sr, _ in items})
    seed_sample_values = sorted({ss for _, _, ss in items})

    data_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for path, sr, ss in items:
        data_map[(sr, ss)] = _load_npz(path)

    files_matrix: List[List[Dict[str, Any] | None]] = []
    file_paths_matrix: List[List[str | None]] = []
    missing_mask = np.zeros((len(seed_rules_values), len(seed_sample_values)), dtype=bool)
    for i, sr in enumerate(seed_rules_values):
        row: List[Dict[str, Any] | None] = []
        row_paths: List[str | None] = []
        for j, ss in enumerate(seed_sample_values):
            item = data_map.get((sr, ss))
            row.append(item)
            row_paths.append(None if item is None else str(next(p for p, xsr, xss in items if xsr == sr and xss == ss)))
            missing_mask[i, j] = item is None
        files_matrix.append(row)
        file_paths_matrix.append(row_paths)

    first = next(item for row in files_matrix for item in row if item is not None)
    params_reference = _parse_params(first.get("params_json"))
    note = first.get("note", "")
    lambda_values = np.asarray(first["lambda_values"], dtype=np.float64)

    metric_keys = []
    for key, value in first.items():
        if key in {"params_json", "note", "lambda_values"}:
            continue
        if isinstance(value, (str, bytes, dict)):
            continue
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            metric_keys.append(key)

    by_sample_seed: Dict[str, np.ndarray] = {}
    mean_over_sample_seed: Dict[str, np.ndarray] = {}
    std_over_sample_seed: Dict[str, np.ndarray] = {}
    for key in metric_keys:
        stacked = _stack_metric(files_matrix, key)
        by_sample_seed[key] = stacked
        mean_over_sample_seed[key] = np.nanmean(stacked, axis=1)
        std_over_sample_seed[key] = np.nanstd(stacked, axis=1)

    return {
        "seed_rules_values": np.asarray(seed_rules_values, dtype=np.int64),
        "seed_sample_values": np.asarray(seed_sample_values, dtype=np.int64),
        "lambda_values": lambda_values,
        "missing_mask": missing_mask,
        "file_paths": np.array(file_paths_matrix, dtype=object),
        "params_reference": params_reference,
        "note": note,
        "metrics_by_seed_sample": by_sample_seed,
        "metrics_mean_over_seed_sample": mean_over_sample_seed,
        "metrics_std_over_seed_sample": std_over_sample_seed,
    }


def collect_run(run_name: str, raw_root: Path, output_root: Path) -> Path:
    run_dir = raw_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    discovered = _discover_files(run_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No raw .npz files matching local/global seed pattern were found in {run_dir}"
        )

    grouped: Dict[str, List[Tuple[Path, int, int]]] = {"local": [], "global": []}
    for path, mode, sr, ss in discovered:
        grouped[mode].append((path, sr, ss))

    modes_out: Dict[str, Any] = {}
    for mode, items in grouped.items():
        if not items:
            continue
        modes_out[mode] = _aggregate_mode(items)

    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"{run_name}.npy"
    payload = {
        "run_name": run_name,
        "raw_root": str(raw_root),
        "run_dir": str(run_dir),
        "modes": modes_out,
    }
    np.save(out_path, payload, allow_pickle=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw BP sweep results over seed_sample, keeping seed_rules separate.")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--raw_root", type=Path, default=Path("data") / "BP_raw")
    parser.add_argument("--output_root", type=Path, default=Path("data") / "BP")
    args = parser.parse_args()

    out_path = collect_run(args.run_name, raw_root=args.raw_root, output_root=args.output_root)
    print(f"Saved collected results to {out_path}")


if __name__ == "__main__":
    main()
