#!/usr/bin/env python3
"""Small wrapper around main.py for one RHM convolutional run.

Defaults are chosen to be the closest convolutional baseline to compare with the
paper's RHM transformer runs:
- RHM with L=3, s=2, v=32, m=8
- hierarchical CNN (hcnn) by default
- depth matched to the hierarchy depth
- filter size matched to tuple_size
- width H=256 (paper-style CNN baseline for RHM)
- SGD + cosine schedule, with the effective learning rate scaled by H inside init.py

Use --model hcnn or --model hlcn to choose the exact convolutional architecture.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Run one paper-style RHM convolutional training")

    # Paper-like RHM defaults (close to Fig. 2 / Fig. 3 style settings)
    parser.add_argument("--train_size", type=int, default=32768)
    parser.add_argument("--test_size", type=int, default=32768,
                        help="Paper uses a validation set of size 2^15 for model selection.")
    parser.add_argument("--num_features", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=32)
    parser.add_argument("--a", type=float, default=None,
                        help="Optional dataset switch forwarded only if requested.")
    parser.add_argument("--num_synonyms", type=int, default=8)
    parser.add_argument("--tuple_size", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_tokens", type=int, default=8,
                        help="Full RHM sequence length s^L. For L=3, s=2 this is 8.")

    # Convolutional defaults from Appendix A.1 spirit
    parser.add_argument("--model", type=str, default="hcnn", choices=["hcnn", "hlcn"],
                        help="Convolutional architecture to train.")
    parser.add_argument("--depth", type=int, default=None,
                        help="Default is num_layers, i.e. matched to the RHM depth.")
    parser.add_argument("--width", type=int, default=256,
                        help="Number of channels H. Default is 256 as a paper-style CNN baseline.")
    parser.add_argument("--filter_size", type=int, default=None,
                        help="Default is tuple_size, matching the RHM branching factor.")
    parser.add_argument("--lr", type=float, default=1.0,
                        help="Base LR before the width multiplication done in init.py.")
    parser.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "warmup"])
    parser.add_argument("--scheduler_time", type=int, default=100,
                        help="Cosine schedule horizon. Paper-style CNN baseline uses 100.")
    parser.add_argument("--bias", action="store_true", default=False)

    # Practical training defaults
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--accumulation", action="store_true", default=False)
    parser.add_argument("--init_scale", type=float, default=1.0,
                        help="Forwarded for interface compatibility; current hcnn/hlcn init ignores it.")
    parser.add_argument("--max_epochs", type=int, default=128)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=25)
    parser.add_argument("--loss_threshold", type=float, default=1e-3)
    parser.add_argument(
        "--compute_margin_stats",
        default=False,
        action="store_true",
        help="Forward only if your main.py supports it.",
    )
    parser.add_argument(
        "--margin_stats_max_samples",
        type=int,
        default=4096,
        help="Maximum number of random training examples used for margin statistics.",
    )
    parser.add_argument(
        "--save_trainstep_epochs",
        type=int,
        default=None,
        help="Forward only if your main.py supports it.",
    )

    # Seeds and runtime
    parser.add_argument("--seed_rules", type=int, default=0)
    parser.add_argument("--seed_sample", type=int, default=0)
    parser.add_argument("--seed_model", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--repo_dir", type=Path, default=repo_dir)
    parser.add_argument("--output_dir", type=Path, default=repo_dir / "results" / "rhm_convolutional_single")
    parser.add_argument("--tag", type=str, default="paper_L3_s2_v32_m8_hcnn")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    depth = args.depth if args.depth is not None else args.num_layers
    filter_size = args.filter_size if args.filter_size is not None else args.tuple_size
    batch_size = choose_batch_size(args.train_size, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outname = args.output_dir / (
        f"{args.tag}_{args.model}_P{args.train_size}_sr{args.seed_rules}_ss{args.seed_sample}_sm{args.seed_model}.pkl"
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
        "--model", args.model,
        "--depth", str(depth),
        "--width", str(args.width),
        "--filter_size", str(filter_size),
        "--seed_model", str(args.seed_model),
        "--lr", str(args.lr),
        "--optim", args.optim,
        "--max_epochs", str(args.max_epochs),
        "--print_freq", str(args.print_freq),
        "--save_freq", str(args.save_freq),
        "--loss_threshold", str(args.loss_threshold),
        "--outname", str(outname),
    ]

    if args.bias:
        cmd.append("--bias")

    if args.scheduler != "none":
        cmd.extend(["--scheduler", args.scheduler, "--scheduler_time", str(args.scheduler_time)])

    if args.accumulation:
        cmd.append("--accumulation")

    if args.a is not None:
        cmd.extend(["--a", str(args.a)])

    if args.compute_margin_stats:
        cmd.extend([
            "--compute_margin_stats",
            "--margin_stats_max_samples", str(args.margin_stats_max_samples),
        ])

    if args.save_trainstep_epochs is not None:
        cmd.extend(["--save_trainstep_epochs", str(args.save_trainstep_epochs)])

    print("[INFO] Running one RHM convolutional training")
    print(f"[INFO] repo_dir={args.repo_dir}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] outname={outname}")
    print(f"[INFO] model={args.model}")
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
