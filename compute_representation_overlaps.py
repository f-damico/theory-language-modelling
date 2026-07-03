#!/usr/bin/env python3
"""
Compute time-time representation overlaps for one trained RHM/transformer run.

Expected training-output layout, with small fallbacks:

    data/<RUN_NAME>/<RUN_ID>/
        checkpoints/checkpoint_epoch_*.pt
        data/dataset_full.npz

or pass the leaf folder directly with --run_dir.

The script selects a logarithmic subset of saved checkpoints, computes hidden
representations on random train/test subsets saved with the run, builds centered
Gram matrices, and saves same-layer linear-CKA overlaps for every checkpoint pair.

Default representation is the last-token hidden state, which is the state used by
last-token prediction readout in this repository.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-12


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _jsonify(vars(obj))
    return repr(obj)


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _as_namespace(x: Any) -> argparse.Namespace:
    if isinstance(x, argparse.Namespace):
        return copy.deepcopy(x)
    if isinstance(x, SimpleNamespace):
        return argparse.Namespace(**vars(x))
    if isinstance(x, dict):
        return argparse.Namespace(**x)
    if hasattr(x, "__dict__"):
        return argparse.Namespace(**vars(x))
    raise TypeError(f"Cannot convert object of type {type(x)} to argparse.Namespace")


def _ensure_arg_defaults(args: argparse.Namespace, *, device: str) -> argparse.Namespace:
    """Add harmless defaults needed by init.init_model if older checkpoints miss them."""
    args = copy.deepcopy(args)
    args.device = device
    if not hasattr(args, "input_size") and hasattr(args, "tuple_size") and hasattr(args, "num_layers"):
        args.input_size = int(args.tuple_size) ** int(args.num_layers)
    defaults = {
        "bias": False,
        "width": None,
        "filter_size": None,
        "init_scale": 1.0,
        "model": "transformer_mla",
        "lr": 0.0,
        "momentum": 0.9,
        "optim": "adam",
        "scheduler": None,
        "scheduler_time": None,
        "max_epochs": 1,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def _load_args_from_pkl(run_dir: Path) -> Optional[argparse.Namespace]:
    pkl_files = sorted(run_dir.glob("*.pkl"))
    if not pkl_files:
        pkl_files = sorted(run_dir.parent.glob("*.pkl"))
    for p in pkl_files:
        try:
            with open(p, "rb") as f:
                args = pickle.load(f)
            return _as_namespace(args)
        except Exception:
            continue
    return None


def _extract_args_from_checkpoint(ckpt: Any, run_dir: Path, *, device: str) -> argparse.Namespace:
    args_obj = None
    if isinstance(ckpt, dict):
        for key in ("args", "train_args", "hparams", "hyperparameters"):
            if key in ckpt:
                args_obj = ckpt[key]
                break
    if args_obj is None:
        args_obj = _load_args_from_pkl(run_dir)
    if args_obj is None:
        manifest = run_dir / "manifest.json"
        if manifest.exists():
            with open(manifest, "r") as f:
                data = json.load(f)
            for key in ("args", "train_args", "hparams", "hyperparameters"):
                if key in data:
                    args_obj = data[key]
                    break
    if args_obj is None:
        raise RuntimeError(
            "Could not find training args in the checkpoint, .pkl, or manifest.json. "
            "The model cannot be reconstructed safely."
        )
    return _ensure_arg_defaults(_as_namespace(args_obj), device=device)


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Raw state dict saved as a dictionary of tensors.
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise RuntimeError("Could not find a model state_dict inside checkpoint.")


def _checkpoint_epoch_from_name(path: Path) -> Optional[int]:
    m = re.search(r"checkpoint_epoch_(\d+)", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"epoch[_-]?(\d+)", path.name)
    if m:
        return int(m.group(1))
    return None


def _checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    ep = _checkpoint_epoch_from_name(path)
    return (ep if ep is not None else 10**18, str(path))


def _find_leaf_run_dir(data_root: Path, run_name: Optional[str], run_dir: Optional[Path], run_id: Optional[str]) -> Path:
    if run_dir is not None:
        out = run_dir.expanduser().resolve()
        if not out.exists():
            raise FileNotFoundError(f"run_dir does not exist: {out}")
        return out

    if run_name is None:
        raise ValueError("Provide either --run_dir or both --data_root and --run_name.")

    root = (data_root.expanduser().resolve() / run_name)
    if not root.exists():
        raise FileNotFoundError(f"Run-name folder does not exist: {root}")

    if run_id is not None:
        out = root / run_id
        if not out.exists():
            raise FileNotFoundError(f"Requested run_id folder does not exist: {out}")
        return out

    if (root / "checkpoints").exists():
        return root

    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "checkpoints").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        # Fallback: maybe the checkpoint files are nested one level deeper with a different name.
        nested = sorted(root.glob("*/checkpoints/checkpoint_epoch_*.pt"))
        parents = sorted({p.parent.parent for p in nested})
        if len(parents) == 1:
            return parents[0]
        raise FileNotFoundError(f"No single run folder with checkpoints found under {root}")

    raise RuntimeError(
        f"Found {len(candidates)} run folders under {root}. Pass --run_id or --run_dir. "
        f"Examples: {[p.name for p in candidates[:5]]}"
    )


def _find_checkpoints(run_dir: Path, max_epoch: Optional[int]) -> List[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints folder: {ckpt_dir}")
    paths = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"), key=_checkpoint_sort_key)
    if not paths:
        paths = sorted([p for p in ckpt_dir.glob("*.pt") if p.name != "latest.pt"], key=_checkpoint_sort_key)
    if not paths:
        raise FileNotFoundError(f"No .pt checkpoints found in {ckpt_dir}")

    if max_epoch is not None:
        kept = []
        for p in paths:
            ep = _checkpoint_epoch_from_name(p)
            if ep is None or ep <= int(max_epoch):
                kept.append(p)
        paths = kept
    if not paths:
        raise RuntimeError("No checkpoint remains after applying --max_epoch.")
    return paths


def _select_log_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    k = int(k)
    if k <= 0 or k >= n:
        return np.arange(n, dtype=int)
    # 1-based log positions: n=1000,k=4 -> 1,10,100,1000.
    raw = np.rint(np.logspace(0.0, math.log10(n), num=k)).astype(int) - 1
    raw = np.clip(raw, 0, n - 1)
    selected: List[int] = []
    for idx in raw.tolist():
        if idx not in selected:
            selected.append(idx)
    if len(selected) < k:
        # Fill possible duplicates with approximately log/linear-spaced missing indices.
        for idx in np.rint(np.linspace(0, n - 1, num=k * 4)).astype(int).tolist():
            if idx not in selected:
                selected.append(idx)
            if len(selected) == k:
                break
    return np.array(sorted(selected[:k]), dtype=int)


def _find_dataset_npz(run_dir: Path) -> Path:
    candidates = [
        run_dir / "data" / "dataset_full.npz",
        run_dir / "dataset_full.npz",
        run_dir / "data" / "dataset_reference_subset.npz",
        run_dir / "dataset_reference_subset.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    recursive = sorted(run_dir.glob("**/dataset_full.npz"))
    if recursive:
        return recursive[0]
    recursive = sorted(run_dir.glob("**/dataset_reference_subset.npz"))
    if recursive:
        return recursive[0]
    raise FileNotFoundError(f"Could not find dataset_full.npz or dataset_reference_subset.npz under {run_dir}")


def _get_npz_array(npz: np.lib.npyio.NpzFile, names: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for name in names:
        if name in npz.files:
            return npz[name], name
    return None, None


def _onehot_masked_from_sequences(sequences: np.ndarray, args: argparse.Namespace) -> torch.Tensor:
    seq = np.asarray(sequences)
    if seq.ndim != 2:
        raise ValueError(f"Expected raw token sequences with shape [N,T], got {seq.shape}")
    v = int(args.num_features)
    t = int(args.num_tokens)
    seq = seq[:, -t:].astype(np.int64, copy=False)
    if np.any(seq < 0) or np.any(seq >= v):
        raise ValueError("Raw token sequences contain values outside [0,num_features).")

    eye = np.eye(v, dtype=np.float32)
    x = eye[seq]  # [N,T,V]

    if int(getattr(args, "whitening", 0)):
        inv_sqrt_norm = (1.0 - 1.0 / v) ** -0.5
        x = (x - 1.0 / v) * inv_sqrt_norm

    # Training code replaces the masked last token by a constant unit-norm vector.
    x[:, -1, :] = v ** -0.5
    return torch.from_numpy(x.astype(np.float32, copy=False))


def _processed_inputs_from_array(arr: np.ndarray, args: argparse.Namespace, *, force_mask_last: bool) -> torch.Tensor:
    x = np.asarray(arr)
    if x.ndim == 2:
        return _onehot_masked_from_sequences(x, args)
    if x.ndim != 3:
        raise ValueError(f"Expected inputs/features with shape [N,T,V] or [N,V,T], got {x.shape}")

    t = int(args.num_tokens)
    v = int(args.num_features)

    # Accept [N,T,V] or [N,V,T].
    if x.shape[1] == t and x.shape[2] == v:
        out = x
    elif x.shape[1] == v and x.shape[2] == t:
        out = np.transpose(x, (0, 2, 1))
    elif x.shape[2] == v:
        out = x[:, -t:, :]
    elif x.shape[1] == v:
        out = np.transpose(x[:, :, -t:], (0, 2, 1))
    else:
        raise ValueError(
            f"Cannot infer feature layout for shape {x.shape}; expected num_tokens={t}, num_features={v}."
        )

    out = out.astype(np.float32, copy=True)
    if force_mask_last:
        out[:, -1, :] = v ** -0.5
    return torch.from_numpy(out)


def _load_split_inputs(
    npz: np.lib.npyio.NpzFile,
    split: str,
    args: argparse.Namespace,
    *,
    force_mask_last: bool,
) -> torch.Tensor:
    input_names = [
        f"{split}_inputs",
        f"{split}_processed_inputs",
        f"{split}_features",
        f"{split}_x",
        f"{split}_X",
    ]
    arr, key = _get_npz_array(npz, input_names)
    if arr is not None:
        return _processed_inputs_from_array(arr, args, force_mask_last=force_mask_last)

    seq_names = [
        f"{split}_sequences",
        f"{split}_rhm_sequences",
        f"{split}_tokens",
        f"{split}_raw_sequences",
    ]
    arr, key = _get_npz_array(npz, seq_names)
    if arr is not None:
        return _onehot_masked_from_sequences(arr, args)

    raise KeyError(
        f"Could not find {split} inputs/sequences in dataset npz. Available keys: {npz.files}"
    )


def _choose_subset_indices(n: int, requested: int, rng: np.random.Generator) -> np.ndarray:
    if requested is None or int(requested) <= 0 or int(requested) >= n:
        return np.arange(n, dtype=np.int64)
    idx = rng.choice(n, size=int(requested), replace=False)
    return np.sort(idx.astype(np.int64))


def _make_loader(x: torch.Tensor, batch_size: int) -> torch.utils.data.DataLoader:
    ds = torch.utils.data.TensorDataset(x)
    return torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0)


def _layer_names(model: torch.nn.Module, include_final_norm: bool) -> List[str]:
    names = ["embedding"]
    n_blocks = len(model.blocks) if hasattr(model, "blocks") else 0
    names.extend([f"block_{i+1}" for i in range(n_blocks)])
    if include_final_norm and hasattr(model, "ln_f"):
        names.append("final_norm")
    return names


def _select_representation(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "last":
        return x[:, -1, :]
    if mode == "mean_pool":
        return x.mean(dim=1)
    if mode == "all_flat":
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    raise ValueError(f"Unknown representation_mode={mode!r}")


@torch.no_grad()
def _forward_representations(
    model: torch.nn.Module,
    batch_inputs: torch.Tensor,
    *,
    representation_mode: str,
    include_final_norm: bool,
) -> List[torch.Tensor]:
    if not all(hasattr(model, attr) for attr in ("token_embedding", "position_embedding", "blocks")):
        raise TypeError("This representation extractor currently supports transformer_mla / transformer_bert models.")

    x = batch_inputs
    B, T, C = x.shape
    token_emb = F.linear(x, model.token_embedding, bias=None) * (C ** -0.5)
    pos_emb = model.position_embedding(torch.arange(T, device=x.device))
    h = token_emb + pos_emb

    reps = [_select_representation(h, representation_mode).detach().cpu()]
    for block in model.blocks:
        h = block(h)
        reps.append(_select_representation(h, representation_mode).detach().cpu())

    if include_final_norm and hasattr(model, "ln_f"):
        h_norm = model.ln_f(h)
        reps.append(_select_representation(h_norm, representation_mode).detach().cpu())
    return reps


def _reconstruct_model(
    checkpoint_path: Path,
    run_dir: Path,
    *,
    device: str,
    repo_dir: Path,
) -> Tuple[torch.nn.Module, argparse.Namespace, Dict[str, Any]]:
    # Make sure repo imports use the target repository, not the location of this script.
    repo_dir = repo_dir.expanduser().resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    import init  # imported after sys.path is set

    ckpt = _safe_torch_load(checkpoint_path, map_location="cpu")
    args = _extract_args_from_checkpoint(ckpt, run_dir, device=device)
    if "transformer" not in str(getattr(args, "model", "")):
        raise ValueError(f"This script currently expects a transformer model, got args.model={args.model!r}")

    state = _extract_state_dict(ckpt)
    model = init.init_model(args)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARNING] load_state_dict non-strict for {checkpoint_path.name}: missing={missing}, unexpected={unexpected}")
    model.eval()

    metadata = {}
    if isinstance(ckpt, dict):
        for key in ("epoch", "epoch_done", "global_update", "update", "train_loss", "test_loss"):
            if key in ckpt:
                metadata[key] = _jsonify(ckpt[key])
    if "epoch" not in metadata:
        ep = _checkpoint_epoch_from_name(checkpoint_path)
        if ep is not None:
            metadata["epoch"] = ep
    return model, args, metadata


def _compute_normalized_grams_for_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    inputs: torch.Tensor,
    *,
    device: str,
    repo_dir: Path,
    batch_size: int,
    representation_mode: str,
    include_final_norm: bool,
    max_gram_rows: int,
) -> Tuple[List[np.ndarray], List[float], argparse.Namespace, List[str], Dict[str, Any]]:
    model, args, ckpt_meta = _reconstruct_model(checkpoint_path, run_dir, device=device, repo_dir=repo_dir)
    layer_names = _layer_names(model, include_final_norm=include_final_norm)

    loader = _make_loader(inputs, batch_size=batch_size)
    chunks: Optional[List[List[torch.Tensor]]] = None

    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        reps = _forward_representations(
            model,
            xb,
            representation_mode=representation_mode,
            include_final_norm=include_final_norm,
        )
        if chunks is None:
            chunks = [[] for _ in reps]
        for i, r in enumerate(reps):
            chunks[i].append(r.to(dtype=torch.float32))

    if chunks is None:
        raise RuntimeError("No representation batch was produced.")

    grams: List[np.ndarray] = []
    norms: List[float] = []
    for name, parts in zip(layer_names, chunks):
        H = torch.cat(parts, dim=0).to(torch.float32)  # [rows, E]
        if int(H.shape[0]) > int(max_gram_rows):
            raise RuntimeError(
                f"Gram matrix would have {H.shape[0]} rows for layer {name}. "
                f"This exceeds --max_gram_rows={max_gram_rows}. Reduce subset size or use representation_mode=last."
            )
        H = H - H.mean(dim=0, keepdim=True)
        G = H @ H.T
        norm = float(torch.linalg.vector_norm(G).item())
        if math.isfinite(norm) and norm > EPS:
            G = G / norm
        else:
            G = torch.zeros_like(G)
            norm = 0.0
        grams.append(G.cpu().numpy().astype(np.float32, copy=False))
        norms.append(norm)
        del H, G

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return grams, norms, args, layer_names, ckpt_meta


def _compute_cka_from_normalized_grams(grams_by_checkpoint: List[List[np.ndarray]]) -> np.ndarray:
    K = len(grams_by_checkpoint)
    L = len(grams_by_checkpoint[0])
    out = np.zeros((L, K, K), dtype=np.float32)
    for l in range(L):
        for i in range(K):
            Gi = grams_by_checkpoint[i][l]
            for j in range(i, K):
                Gj = grams_by_checkpoint[j][l]
                val = float(np.sum(Gi.astype(np.float64, copy=False) * Gj.astype(np.float64, copy=False)))
                val = max(0.0, min(1.0, val))
                out[l, i, j] = val
                out[l, j, i] = val
    return out


def _estimate_gram_memory_gb(num_ckpts: int, num_layers: int, n_rows: int) -> float:
    return (num_ckpts * num_layers * n_rows * n_rows * 4.0) / (1024.0 ** 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute same-layer representation CKA overlaps for one trained run.")

    parser.add_argument("--repo_dir", type=Path, default=Path.cwd(), help="repository root containing init.py/models")
    parser.add_argument("--data_root", type=Path, default=None, help="root containing run-name folders; default: repo_dir/data")
    parser.add_argument("--run_name", type=str, default=None, help="folder name under data_root")
    parser.add_argument("--run_id", type=str, default=None, help="optional child folder inside data_root/run_name")
    parser.add_argument("--run_dir", type=Path, default=None, help="direct path to the single leaf run folder")

    parser.add_argument("--results_root", type=Path, default=None, help="default: repo_dir/results")
    parser.add_argument("--output_name", type=str, required=True, help="folder created inside results_root")

    parser.add_argument("--num_savings", type=int, default=8, help="number of logarithmically selected checkpoints")
    parser.add_argument("--max_epoch", type=int, default=None, help="ignore checkpoints with epoch greater than this value")

    parser.add_argument("--subset_train_size", type=int, default=1024, help="random train examples; <=0 means all")
    parser.add_argument("--subset_test_size", type=int, default=1024, help="random test examples; <=0 means all")
    parser.add_argument("--subset_seed", type=int, default=None, help="default: seed_sample from checkpoint args, else 0")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--representation_mode", type=str, default="last", choices=["last", "mean_pool", "all_flat"])
    parser.add_argument("--include_final_norm", action="store_true", default=True)
    parser.add_argument("--no_include_final_norm", dest="include_final_norm", action="store_false")
    parser.add_argument("--force_mask_last", action="store_true", default=True)
    parser.add_argument("--no_force_mask_last", dest="force_mask_last", action="store_false")
    parser.add_argument("--max_gram_rows", type=int, default=4096)

    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    repo_dir = cli.repo_dir.expanduser().resolve()
    data_root = (cli.data_root.expanduser().resolve() if cli.data_root is not None else repo_dir / "data")
    results_root = (cli.results_root.expanduser().resolve() if cli.results_root is not None else repo_dir / "results")

    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False.")

    run_dir = _find_leaf_run_dir(data_root, cli.run_name, cli.run_dir, cli.run_id)
    ckpt_paths_all = _find_checkpoints(run_dir, cli.max_epoch)
    selected_idx = _select_log_indices(len(ckpt_paths_all), cli.num_savings)
    ckpt_paths = [ckpt_paths_all[int(i)] for i in selected_idx]

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] checkpoints_total_after_filter={len(ckpt_paths_all)}")
    print(f"[INFO] selected_checkpoints={len(ckpt_paths)}")
    for p in ckpt_paths:
        print(f"[INFO]   {p.name}")

    # Load first selected checkpoint only to recover args needed for dataset parsing.
    first_ckpt = _safe_torch_load(ckpt_paths[0], map_location="cpu")
    first_args = _extract_args_from_checkpoint(first_ckpt, run_dir, device=cli.device)
    if cli.subset_seed is None:
        subset_seed = int(getattr(first_args, "seed_sample", 0))
    else:
        subset_seed = int(cli.subset_seed)

    dataset_path = _find_dataset_npz(run_dir)
    print(f"[INFO] dataset_path={dataset_path}")
    npz = np.load(dataset_path, allow_pickle=True)

    train_inputs_all = _load_split_inputs(npz, "train", first_args, force_mask_last=cli.force_mask_last)
    test_inputs_all = _load_split_inputs(npz, "test", first_args, force_mask_last=cli.force_mask_last)

    rng = np.random.default_rng(subset_seed)
    train_idx = _choose_subset_indices(len(train_inputs_all), cli.subset_train_size, rng)
    test_idx = _choose_subset_indices(len(test_inputs_all), cli.subset_test_size, rng)
    train_inputs = train_inputs_all[torch.as_tensor(train_idx, dtype=torch.long)].contiguous()
    test_inputs = test_inputs_all[torch.as_tensor(test_idx, dtype=torch.long)].contiguous()

    print(f"[INFO] subset_seed={subset_seed}")
    print(f"[INFO] train_subset={len(train_idx)} / {len(train_inputs_all)}")
    print(f"[INFO] test_subset={len(test_idx)} / {len(test_inputs_all)}")
    print(f"[INFO] representation_mode={cli.representation_mode}")

    run_label = run_dir.name
    if cli.run_name is not None and run_label == cli.run_name:
        run_label = cli.run_name
    out_dir = results_root / cli.output_name / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    result_arrays: Dict[str, Any] = {}
    layer_names: Optional[List[str]] = None
    selected_epochs: List[int] = []
    selected_global_updates: List[int] = []
    args_dict: Optional[Dict[str, Any]] = None

    for split, x in [("train", train_inputs), ("test", test_inputs)]:
        print(f"[INFO] Processing split={split}")
        grams_by_checkpoint: List[List[np.ndarray]] = []
        gram_norms: List[List[float]] = []
        split_meta: List[Dict[str, Any]] = []

        for k, p in enumerate(ckpt_paths):
            print(f"[INFO] [{split}] checkpoint {k+1}/{len(ckpt_paths)}: {p.name}")
            grams, norms, ckpt_args, names, meta = _compute_normalized_grams_for_checkpoint(
                p,
                run_dir,
                x,
                device=cli.device,
                repo_dir=repo_dir,
                batch_size=cli.batch_size,
                representation_mode=cli.representation_mode,
                include_final_norm=cli.include_final_norm,
                max_gram_rows=cli.max_gram_rows,
            )
            if layer_names is None:
                layer_names = names
                args_dict = vars(ckpt_args).copy()
                eff_rows = int(grams[0].shape[0])
                est_gb = _estimate_gram_memory_gb(len(ckpt_paths), len(layer_names), eff_rows)
                print(f"[INFO] effective_gram_rows={eff_rows}")
                print(f"[INFO] estimated stored Gram RAM per split ~ {est_gb:.3f} GB")
            elif layer_names != names:
                raise RuntimeError(f"Layer names changed across checkpoints: {layer_names} vs {names}")

            grams_by_checkpoint.append(grams)
            gram_norms.append(norms)
            split_meta.append(meta)

        cka = _compute_cka_from_normalized_grams(grams_by_checkpoint)
        result_arrays[f"{split}_cka"] = cka
        result_arrays[f"{split}_gram_fro_norms"] = np.asarray(gram_norms, dtype=np.float64)  # [K,L]
        result_arrays[f"{split}_subset_indices"] = train_idx if split == "train" else test_idx
        result_arrays[f"{split}_effective_gram_rows"] = np.array(grams_by_checkpoint[0][0].shape[0], dtype=np.int64)

        if split == "train":
            selected_epochs = [int(m.get("epoch", _checkpoint_epoch_from_name(p) or -1)) for m, p in zip(split_meta, ckpt_paths)]
            selected_global_updates = [int(m.get("global_update", -1)) for m in split_meta]

        del grams_by_checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if layer_names is None:
        raise RuntimeError("No layers were processed.")

    result_arrays.update(
        {
            "layer_names": np.asarray(layer_names),
            "selected_checkpoint_paths": np.asarray([str(p) for p in ckpt_paths]),
            "selected_checkpoint_files": np.asarray([p.name for p in ckpt_paths]),
            "selected_checkpoint_indices_in_filtered_list": selected_idx.astype(np.int64),
            "selected_epochs": np.asarray(selected_epochs, dtype=np.int64),
            "selected_global_updates": np.asarray(selected_global_updates, dtype=np.int64),
            "representation_mode": np.asarray(cli.representation_mode),
            "metric": np.asarray("centered_linear_CKA_from_normalized_Gram"),
            "run_dir": np.asarray(str(run_dir)),
            "dataset_path": np.asarray(str(dataset_path)),
            "subset_seed": np.array(subset_seed, dtype=np.int64),
            "num_savings_requested": np.array(cli.num_savings, dtype=np.int64),
            "max_epoch": np.array(-1 if cli.max_epoch is None else cli.max_epoch, dtype=np.int64),
        }
    )

    out_npz = out_dir / "representation_overlaps.npz"
    np.savez_compressed(out_npz, **result_arrays)

    metadata = {
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "output_npz": str(out_npz),
        "metric": "centered_linear_CKA_from_normalized_Gram",
        "representation_mode": cli.representation_mode,
        "layer_names": layer_names,
        "num_checkpoints_total_after_filter": len(ckpt_paths_all),
        "num_checkpoints_selected": len(ckpt_paths),
        "selected_checkpoint_files": [p.name for p in ckpt_paths],
        "selected_epochs": selected_epochs,
        "selected_global_updates": selected_global_updates,
        "subset_seed": subset_seed,
        "train_subset_size": int(len(train_idx)),
        "test_subset_size": int(len(test_idx)),
        "force_mask_last": bool(cli.force_mask_last),
        "include_final_norm": bool(cli.include_final_norm),
        "args": _jsonify(args_dict or {}),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(out_dir / "selected_checkpoints.txt", "w") as f:
        for ep, p in zip(selected_epochs, ckpt_paths):
            f.write(f"{ep}\t{p}\n")

    print(f"[DONE] Saved {out_npz}")
    print(f"[DONE] Metadata {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
