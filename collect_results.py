#!/usr/bin/env python3
from pathlib import Path
import argparse
import pickle
import numpy as np


SEED_KEYS = ("seed_rules", "seed_sample", "seed_model")

IGNORE_COMPARE_KEYS = {
    "outname",
    "device",
    "run_name",
    "data_root",
    "train_size",
    *SEED_KEYS,
}


def load_one_file(path):
    with open(path, "rb") as f:
        args = pickle.load(f)
        output = pickle.load(f)

    if hasattr(args, "__dict__"):
        args_dict = vars(args).copy()
    elif isinstance(args, dict):
        args_dict = dict(args)
    else:
        raise TypeError(f"Unsupported args type in {path}: {type(args)}")

    return args_dict, output


def dynamics_to_arrays(output):
    dyn = output.get("dynamics", [])
    if len(dyn) == 0:
        return {
            "epochs": np.array([], dtype=int),
            "trainloss": np.array([], dtype=float),
            "testloss": np.array([], dtype=float),
            "testacc": np.array([], dtype=float),
        }

    return {
        "epochs": np.array([d["t"] for d in dyn], dtype=int),
        "trainloss": np.array([d["trainloss"] for d in dyn], dtype=float),
        "testloss": np.array([d["testloss"] for d in dyn], dtype=float),
        "testacc": np.array([d["testacc"] for d in dyn], dtype=float),
    }


def comparable_params(args_dict):
    return {k: v for k, v in args_dict.items() if k not in IGNORE_COMPARE_KEYS}


def nanmean_std_with_flag(x, axis):
    valid = ~np.isnan(x)
    counts = valid.sum(axis=axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(x, axis=axis)
        std = np.nanstd(x, axis=axis, ddof=1)

    std = np.where(counts < 2, -1.0, std)
    mean = np.where(counts == 0, np.nan, mean)

    return mean, std, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True,
                        help="folder name inside data_root containing raw .pkl results")
    parser.add_argument("--data_root", type=str, default="data",
                        help="root folder containing run folders")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="where to save the aggregated .npy")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="name of the saved .npy file; default = run_name")
    args = parser.parse_args()

    run_dir = Path(args.data_root).expanduser().resolve() / args.run_name
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = args.experiment_name or args.run_name
    save_path = results_dir / f"{experiment_name}.npy"

    files = sorted(run_dir.glob("*.pkl"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl files found in {run_dir}")

    # Read all files once
    entries = []
    for path in files:
        args_dict, output = load_one_file(path)
        dyn = dynamics_to_arrays(output)

        best = output.get("best", {})
        entry = {
            "path": str(path),
            "train_size": int(args_dict["train_size"]),
            "seed_rules": int(args_dict.get("seed_rules", -1)),
            "seed_sample": int(args_dict.get("seed_sample", -1)),
            "seed_model": int(args_dict.get("seed_model", -1)),
            "args": args_dict,
            "params_compare": comparable_params(args_dict),
            "epochs": dyn["epochs"],
            "trainloss": dyn["trainloss"],
            "testloss": dyn["testloss"],
            "testacc": dyn["testacc"],
            "best_loss": float(best.get("loss", np.nan)),
            "best_acc": float(best.get("acc", np.nan)),
            "best_epoch": float(best.get("epoch", np.nan)),
            "last_saved_epoch": float(output.get("epoch", np.nan)),
        }
        entries.append(entry)

    # Check that all non-seed, non-train_size params are identical
    ref = entries[0]["params_compare"]
    for e in entries[1:]:
        if e["params_compare"] != ref:
            raise ValueError(
                "Files in this run folder do not belong to one single experiment family.\n"
                "At least one non-seed/non-train_size hyperparameter differs.\n"
                f"Reference params:\n{ref}\n\nDifferent params:\n{e['params_compare']}\n\n"
                f"Problematic file: {e['path']}"
            )

    # Sorted unique P and global union of epochs
    P_values = np.array(sorted({e["train_size"] for e in entries}), dtype=int)
    epoch_values = np.array(sorted({int(t) for e in entries for t in e["epochs"]}), dtype=int)

    # Group entries by P
    entries_by_P = {int(P): [] for P in P_values}
    for e in entries:
        entries_by_P[int(e["train_size"])].append(e)

    for P in P_values:
        entries_by_P[int(P)] = sorted(
            entries_by_P[int(P)],
            key=lambda e: (e["seed_rules"], e["seed_sample"], e["seed_model"], e["path"])
        )

    max_seeds = max(len(v) for v in entries_by_P.values())
    nP = len(P_values)
    nT = len(epoch_values)

    # Raw arrays: shape (nP, max_seeds, nT)
    trainloss_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)
    testloss_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)
    testacc_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)

    # Raw scalar summaries: shape (nP, max_seeds)
    best_loss_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    best_acc_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    best_epoch_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    last_saved_epoch_raw = np.full((nP, max_seeds), np.nan, dtype=float)

    # Seeds: shape (nP, max_seeds, 3)
    seed_triplets = np.full((nP, max_seeds, 3), -1, dtype=int)

    # Counts
    num_seeds = np.zeros(nP, dtype=int)

    # Optional traceability
    file_index = np.full((nP, max_seeds), "", dtype=object)

    epoch_to_idx = {int(t): i for i, t in enumerate(epoch_values)}

    for iP, P in enumerate(P_values):
        plist = entries_by_P[int(P)]
        num_seeds[iP] = len(plist)

        for iseed, e in enumerate(plist):
            seed_triplets[iP, iseed, 0] = e["seed_rules"]
            seed_triplets[iP, iseed, 1] = e["seed_sample"]
            seed_triplets[iP, iseed, 2] = e["seed_model"]
            file_index[iP, iseed] = e["path"]

            for local_i, ep in enumerate(e["epochs"]):
                j = epoch_to_idx[int(ep)]
                trainloss_raw[iP, iseed, j] = e["trainloss"][local_i]
                testloss_raw[iP, iseed, j] = e["testloss"][local_i]
                testacc_raw[iP, iseed, j] = e["testacc"][local_i]

            best_loss_raw[iP, iseed] = e["best_loss"]
            best_acc_raw[iP, iseed] = e["best_acc"]
            best_epoch_raw[iP, iseed] = e["best_epoch"]
            last_saved_epoch_raw[iP, iseed] = e["last_saved_epoch"]

    # Aggregate over seed axis=1
    trainloss_mean, trainloss_std, trainloss_n = nanmean_std_with_flag(trainloss_raw, axis=1)
    testloss_mean, testloss_std, testloss_n = nanmean_std_with_flag(testloss_raw, axis=1)
    testacc_mean, testacc_std, testacc_n = nanmean_std_with_flag(testacc_raw, axis=1)

    best_loss_mean, best_loss_std, best_loss_n = nanmean_std_with_flag(best_loss_raw, axis=1)
    best_acc_mean, best_acc_std, best_acc_n = nanmean_std_with_flag(best_acc_raw, axis=1)
    best_epoch_mean, best_epoch_std, best_epoch_n = nanmean_std_with_flag(best_epoch_raw, axis=1)
    last_saved_epoch_mean, last_saved_epoch_std, last_saved_epoch_n = nanmean_std_with_flag(last_saved_epoch_raw, axis=1)

    result = {
        # identifiers
        "run_name": np.array(args.run_name),
        "experiment_name": np.array(experiment_name),

        # fixed experiment parameters
        "fixed_params": np.array(ref, dtype=object),

        # axes
        "P_values": P_values,                 # shape (nP,)
        "epoch_values": epoch_values,         # shape (nT,)
        "num_seeds": num_seeds,               # shape (nP,)
        "seed_triplets": seed_triplets,       # shape (nP, max_seeds, 3)

        # raw curves per seed
        "trainloss_raw": trainloss_raw,       # shape (nP, max_seeds, nT)
        "testloss_raw": testloss_raw,         # shape (nP, max_seeds, nT)
        "testacc_raw": testacc_raw,           # shape (nP, max_seeds, nT)

        # aggregated curves
        "trainloss_mean": trainloss_mean,     # shape (nP, nT)
        "trainloss_std": trainloss_std,       # shape (nP, nT)
        "trainloss_n": trainloss_n,           # shape (nP, nT)

        "testloss_mean": testloss_mean,       # shape (nP, nT)
        "testloss_std": testloss_std,         # shape (nP, nT)
        "testloss_n": testloss_n,             # shape (nP, nT)

        "testacc_mean": testacc_mean,         # shape (nP, nT)
        "testacc_std": testacc_std,           # shape (nP, nT)
        "testacc_n": testacc_n,               # shape (nP, nT)

        # raw scalar summaries per seed
        "best_loss_raw": best_loss_raw,               # shape (nP, max_seeds)
        "best_acc_raw": best_acc_raw,                 # shape (nP, max_seeds)
        "best_epoch_raw": best_epoch_raw,             # shape (nP, max_seeds)
        "last_saved_epoch_raw": last_saved_epoch_raw, # shape (nP, max_seeds)

        # aggregated scalar summaries over seeds
        "best_loss_mean": best_loss_mean,             # shape (nP,)
        "best_loss_std": best_loss_std,               # shape (nP,)
        "best_loss_n": best_loss_n,                   # shape (nP,)

        "best_acc_mean": best_acc_mean,               # shape (nP,)
        "best_acc_std": best_acc_std,                 # shape (nP,)
        "best_acc_n": best_acc_n,                     # shape (nP,)

        "best_epoch_mean": best_epoch_mean,           # shape (nP,)
        "best_epoch_std": best_epoch_std,             # shape (nP,)
        "best_epoch_n": best_epoch_n,                 # shape (nP,)

        "last_saved_epoch_mean": last_saved_epoch_mean,   # shape (nP,)
        "last_saved_epoch_std": last_saved_epoch_std,     # shape (nP,)
        "last_saved_epoch_n": last_saved_epoch_n,         # shape (nP,)

        # optional traceability
        "file_index": file_index,                     # shape (nP, max_seeds)
    }

    np.save(save_path, result, allow_pickle=True)

    print(f"Saved aggregated numpy dict to:\n  {save_path}")
    print(f"Found {len(files)} raw files")
    print(f"P values: {P_values.tolist()}")
    print(f"Global epoch checkpoints: {epoch_values.tolist()}")
    print(f"num_seeds per P: {num_seeds.tolist()}")


if __name__ == "__main__":
    main()