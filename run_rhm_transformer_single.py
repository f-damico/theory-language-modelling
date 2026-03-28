#!/usr/bin/env python3
"""Small wrapper around main.py for one RHM transformer run.

Defaults follow a typical paper setting for Fig. 2:
- RHM with L=3, s=2, v=32, m=8
- simple multi-head attention stack (MLA)
- depth=3
- num_heads=16, embedding_dim=num_heads*vocab_size=512
- Adam + warmup to lr=1e-2 in the first 10 epochs

The training set size is left configurable so the same script can be used to build a curve.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


def choose_batch_size(train_size: int, requested_batch_size: int) -> int:
    """Return a batch size <= train_size that divides train_size."""
    if requested_batch_size >= train_size:
        return train_size

    batch_size = requested_batch_size
    while batch_size > 1 and train_size % batch_size != 0:
        batch_size //= 2

    if batch_size <= 0:
        return 1
    return batch_size


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run one paper-style RHM transformer training")

    # Paper-like RHM defaults (Fig. 2 left)
    parser.add_argument("--train_size", type=int, default=32768)
    parser.add_argument("--test_size", type=int, default=32768,
                        help="Paper uses a validation set of size 2^15 for model selection.")
    parser.add_argument("--num_features", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=32)
    parser.add_argument("--a", type=float, default=-1.0,
                    help="dataset switch: a<0 current dataset, a>=0 power-law last-layer dataset")
    parser.add_argument("--num_synonyms", type=int, default=8)
    parser.add_argument("--tuple_size", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_tokens", type=int, default=8,
                        help="Full RHM sequence length s^L. For L=3, s=2 this is 8.")

    # Transformer defaults from Appendix A.2
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Default is num_heads * num_features.")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--scheduler_time", type=int, default=10,
                        help="Warmup length in epochs.")

    # Practical training defaults
    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--accumulation", action="store_true", default=False)
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=25)
    parser.add_argument("--loss_threshold", type=float, default=1e-3)
    parser.add_argument(
        "--compute_margin_stats",
        default=False,
        action="store_true",
        help="compute training margin statistics on a random subset of the training set",
    )
    parser.add_argument(
        "--margin_stats_max_samples",
        type=int,
        default=4096,
        help="maximum number of random training examples used for margin statistics",
    )

    # Seeds and runtime
    parser.add_argument("--seed_rules", type=int, default=0)
    parser.add_argument("--seed_sample", type=int, default=0)
    parser.add_argument("--seed_model", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--repo_dir", type=Path, default=repo_dir)
    parser.add_argument("--output_dir", type=Path, default=repo_dir / "results" / "rhm_transformer_single")
    parser.add_argument("--tag", type=str, default="paper_L3_s2_v32_m8")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedding_dim = args.embedding_dim or (args.num_heads * args.num_features)
    batch_size = choose_batch_size(args.train_size, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outname = args.output_dir / (
        f"{args.tag}_P{args.train_size}_sr{args.seed_rules}_ss{args.seed_sample}_sm{args.seed_model}.pkl"
    )

    cmd = [
        args.python_bin,
        "-u",
        str(args.repo_dir / "main.py"),
        "--device", args.device,
        "--dataset", "rhm",
        "--mode", "masked",
        "--num_features", str(args.num_features),
        "--num_classes", str(args.num_classes),
        "--a", str(args.a),
        "--num_synonyms", str(args.num_synonyms),
        "--tuple_size", str(args.tuple_size),
        "--num_layers", str(args.num_layers),
        "--seed_rules", str(args.seed_rules),
        "--num_tokens", str(args.num_tokens),
        "--train_size", str(args.train_size),
        "--batch_size", str(batch_size),
        "--init_scale", str(args.init_scale),
        "--test_size", str(args.test_size),
        "--seed_sample", str(args.seed_sample),
        "--input_format", "onehot",
        "--whitening", "0",
        "--model", "transformer_mla",
        "--depth", str(args.depth),
        "--num_heads", str(args.num_heads),
        "--embedding_dim", str(embedding_dim),
        "--seed_model", str(args.seed_model),
        "--lr", str(args.lr),
        "--optim", "adam",
        "--scheduler", "warmup",
        "--scheduler_time", str(args.scheduler_time),
        "--max_epochs", str(args.max_epochs),
        "--print_freq", str(args.print_freq),
        "--save_freq", str(args.save_freq),
        "--loss_threshold", str(args.loss_threshold),
        "--outname", str(outname),
    ]

    if args.compute_margin_stats:
        cmd.extend([
            "--compute_margin_stats",
            "--margin_stats_max_samples", str(args.margin_stats_max_samples),
        ])
    if args.accumulation:
        cmd.append("--accumulation")

    print("[INFO] Running one RHM transformer training")
    print(f"[INFO] repo_dir={args.repo_dir}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] outname={outname}")
    print(f"[INFO] effective_batch_size={batch_size}")
    print("[CMD] " + " ".join(cmd))

    env = dict(**__import__("os").environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print(f"[INFO] OMP_NUM_THREADS={env['OMP_NUM_THREADS']}")
    print(f"[INFO] MKL_NUM_THREADS={env['MKL_NUM_THREADS']}")

    subprocess.run(cmd, cwd=args.repo_dir, check=True, env=env)


if __name__ == "__main__":
    main()
