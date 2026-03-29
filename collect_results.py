#!/usr/bin/env python3
from pathlib import Path
import argparse
import pickle
import numpy as np
import torch
import io


SEED_KEYS = ("seed_rules", "seed_sample", "seed_model")

IGNORE_COMPARE_KEYS = {
    "outname",
    "device",
    "run_name",
    "data_root",
    "train_size",
    "save_trainstep_epochs",   # only changes extra saving/logging, not the base experiment
    *SEED_KEYS,
}

SPECTRAL_KEY = "specnorm"
L2_KEY = "l2norm"
SPECTRAL_NO_QK_KEY = "specnorm_no_qk"

MARGIN_MIN_KEY = "margin_min"
MARGIN_MEAN_KEY = "margin_mean"
MARGIN_MAX_KEY = "margin_max"
MARGIN_STD_KEY = "margin_std"


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_one_file(path):
    old_load_from_bytes = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
    try:
        with open(path, "rb") as f:
            args = pickle.load(f)
            output = pickle.load(f)
    finally:
        torch.storage._load_from_bytes = old_load_from_bytes

    if hasattr(args, "__dict__"):
        args_dict = vars(args).copy()
    elif isinstance(args, dict):
        args_dict = dict(args)
    else:
        raise TypeError(f"Unsupported args type in {path}: {type(args)}")

    return args_dict, output


def dynamics_to_arrays(output, dyn_key="dynamics", time_name="epochs"):
    dyn = output.get(dyn_key, [])
    if dyn is None:
        dyn = []

    if len(dyn) == 0:
        return {
            time_name: np.array([], dtype=int),
            "trainloss": np.array([], dtype=float),
            "testloss": np.array([], dtype=float),
            "testacc": np.array([], dtype=float),
            "err": np.array([], dtype=float),
            "spectral": np.array([], dtype=float),
            "spectral_no_qk": np.array([], dtype=float),
            "l2": np.array([], dtype=float),
            "margin_min": np.array([], dtype=float),
            "margin_mean": np.array([], dtype=float),
            "margin_max": np.array([], dtype=float),
            "margin_std": np.array([], dtype=float),
        }

    # For epoch dynamics, "t" is epoch index.
    # For timestep dynamics, allow fallback to "global_update" if needed.
    times = np.array(
        [int(d.get("t", d.get("global_update", -1))) for d in dyn],
        dtype=int
    )

    trainloss = np.array([d["trainloss"] for d in dyn], dtype=float)
    testloss = np.array([d["testloss"] for d in dyn], dtype=float)
    testacc = np.array([d["testacc"] for d in dyn], dtype=float)

    err = 1.0 - testacc

    spectral = np.array([d.get(SPECTRAL_KEY, np.nan) for d in dyn], dtype=float)
    spectral_no_qk = np.array([d.get(SPECTRAL_NO_QK_KEY, np.nan) for d in dyn], dtype=float)
    l2 = np.array([d.get(L2_KEY, np.nan) for d in dyn], dtype=float)

    margin_min = np.array([d.get(MARGIN_MIN_KEY, np.nan) for d in dyn], dtype=float)
    margin_mean = np.array([d.get(MARGIN_MEAN_KEY, np.nan) for d in dyn], dtype=float)
    margin_max = np.array([d.get(MARGIN_MAX_KEY, np.nan) for d in dyn], dtype=float)
    margin_std = np.array([d.get(MARGIN_STD_KEY, np.nan) for d in dyn], dtype=float)

    return {
        time_name: times,
        "trainloss": trainloss,
        "testloss": testloss,
        "testacc": testacc,
        "err": err,
        "spectral": spectral,
        "spectral_no_qk": spectral_no_qk,
        "l2": l2,
        "margin_min": margin_min,
        "margin_mean": margin_mean,
        "margin_max": margin_max,
        "margin_std": margin_std,
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


def nanmean_std_with_flag_optional(raw):
    """
    raw is assumed to have shape (nP, max_seeds, nT).
    If nT == 0, return empty arrays with shape (nP, 0)
    instead of doing reductions on empty slices.
    """
    if raw.shape[-1] == 0:
        out_shape = (raw.shape[0], 0)
        mean = np.full(out_shape, np.nan, dtype=float)
        std = np.full(out_shape, -1.0, dtype=float)
        counts = np.zeros(out_shape, dtype=int)
        return mean, std, counts

    return nanmean_std_with_flag(raw, axis=1)


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

        dyn = dynamics_to_arrays(output, dyn_key="dynamics", time_name="epochs")
        dyn_timestep = dynamics_to_arrays(output, dyn_key="dynamics_timestep", time_name="timesteps")

        best = output.get("best", {})
        entry = {
            "path": str(path),
            "train_size": int(args_dict["train_size"]),
            "seed_rules": int(args_dict.get("seed_rules", -1)),
            "seed_sample": int(args_dict.get("seed_sample", -1)),
            "seed_model": int(args_dict.get("seed_model", -1)),
            "args": args_dict,
            "params_compare": comparable_params(args_dict),

            # epoch-level dynamics
            "epochs": dyn["epochs"],
            "trainloss": dyn["trainloss"],
            "testloss": dyn["testloss"],
            "testacc": dyn["testacc"],
            "err": dyn["err"],
            "spectral": dyn["spectral"],
            "spectral_no_qk": dyn["spectral_no_qk"],
            "l2": dyn["l2"],
            "margin_min": dyn["margin_min"],
            "margin_mean": dyn["margin_mean"],
            "margin_max": dyn["margin_max"],
            "margin_std": dyn["margin_std"],

            # timestep-level dynamics (optional)
            "timesteps": dyn_timestep["timesteps"],
            "trainloss_timestep": dyn_timestep["trainloss"],
            "testloss_timestep": dyn_timestep["testloss"],
            "testacc_timestep": dyn_timestep["testacc"],
            "err_timestep": dyn_timestep["err"],
            "spectral_timestep": dyn_timestep["spectral"],
            "spectral_no_qk_timestep": dyn_timestep["spectral_no_qk"],
            "l2_timestep": dyn_timestep["l2"],
            "margin_min_timestep": dyn_timestep["margin_min"],
            "margin_mean_timestep": dyn_timestep["margin_mean"],
            "margin_max_timestep": dyn_timestep["margin_max"],
            "margin_std_timestep": dyn_timestep["margin_std"],

            # summaries
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

    # Sorted unique P and global unions of saved checkpoints
    P_values = np.array(sorted({e["train_size"] for e in entries}), dtype=int)
    epoch_values = np.array(sorted({int(t) for e in entries for t in e["epochs"]}), dtype=int)
    timestep_values = np.array(sorted({int(t) for e in entries for t in e["timesteps"]}), dtype=int)

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
    nTs = len(timestep_values)

    # -----------------------------
    # Raw epoch arrays: (nP, max_seeds, nT)
    # -----------------------------
    trainloss_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)
    testloss_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)
    testacc_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)

    spectral_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)
    spectral_no_qk_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)
    l2_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)

    margin_min_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)
    margin_mean_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)
    margin_max_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)
    margin_std_raw = np.full((nP, max_seeds, nT), np.nan, dtype=float)

    # -----------------------------
    # Raw timestep arrays: (nP, max_seeds, nTs)
    # -----------------------------
    trainloss_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)
    testloss_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)
    testacc_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)

    spectral_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)
    spectral_no_qk_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)
    l2_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)

    margin_min_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)
    margin_mean_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)
    margin_max_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)
    margin_std_timestep_raw = np.full((nP, max_seeds, nTs), np.nan, dtype=float)

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
    timestep_to_idx = {int(t): i for i, t in enumerate(timestep_values)}

    for iP, P in enumerate(P_values):
        plist = entries_by_P[int(P)]
        num_seeds[iP] = len(plist)

        for iseed, e in enumerate(plist):
            seed_triplets[iP, iseed, 0] = e["seed_rules"]
            seed_triplets[iP, iseed, 1] = e["seed_sample"]
            seed_triplets[iP, iseed, 2] = e["seed_model"]
            file_index[iP, iseed] = e["path"]

            # epoch-level fills
            for local_i, ep in enumerate(e["epochs"]):
                j = epoch_to_idx[int(ep)]
                trainloss_raw[iP, iseed, j] = e["trainloss"][local_i]
                testloss_raw[iP, iseed, j] = e["testloss"][local_i]
                testacc_raw[iP, iseed, j] = e["testacc"][local_i]

                spectral_raw[iP, iseed, j] = e["spectral"][local_i]
                spectral_no_qk_raw[iP, iseed, j] = e["spectral_no_qk"][local_i]
                l2_raw[iP, iseed, j] = e["l2"][local_i]

                margin_min_raw[iP, iseed, j] = e["margin_min"][local_i]
                margin_mean_raw[iP, iseed, j] = e["margin_mean"][local_i]
                margin_max_raw[iP, iseed, j] = e["margin_max"][local_i]
                margin_std_raw[iP, iseed, j] = e["margin_std"][local_i]

            # timestep-level fills (optional)
            for local_i, ts in enumerate(e["timesteps"]):
                j = timestep_to_idx[int(ts)]
                trainloss_timestep_raw[iP, iseed, j] = e["trainloss_timestep"][local_i]
                testloss_timestep_raw[iP, iseed, j] = e["testloss_timestep"][local_i]
                testacc_timestep_raw[iP, iseed, j] = e["testacc_timestep"][local_i]

                spectral_timestep_raw[iP, iseed, j] = e["spectral_timestep"][local_i]
                spectral_no_qk_timestep_raw[iP, iseed, j] = e["spectral_no_qk_timestep"][local_i]
                l2_timestep_raw[iP, iseed, j] = e["l2_timestep"][local_i]

                margin_min_timestep_raw[iP, iseed, j] = e["margin_min_timestep"][local_i]
                margin_mean_timestep_raw[iP, iseed, j] = e["margin_mean_timestep"][local_i]
                margin_max_timestep_raw[iP, iseed, j] = e["margin_max_timestep"][local_i]
                margin_std_timestep_raw[iP, iseed, j] = e["margin_std_timestep"][local_i]

            best_loss_raw[iP, iseed] = e["best_loss"]
            best_acc_raw[iP, iseed] = e["best_acc"]
            best_epoch_raw[iP, iseed] = e["best_epoch"]
            last_saved_epoch_raw[iP, iseed] = e["last_saved_epoch"]

    # Derived raw errors
    err_raw = 1.0 - testacc_raw
    err_timestep_raw = 1.0 - testacc_timestep_raw

    # -----------------------------
    # Aggregate epoch arrays over seed axis=1
    # -----------------------------
    trainloss_mean, trainloss_std, trainloss_n = nanmean_std_with_flag_optional(trainloss_raw)
    testloss_mean, testloss_std, testloss_n = nanmean_std_with_flag_optional(testloss_raw)
    testacc_mean, testacc_std, testacc_n = nanmean_std_with_flag_optional(testacc_raw)
    err_mean, err_std, err_n = nanmean_std_with_flag_optional(err_raw)

    spectral_mean, spectral_std, spectral_n = nanmean_std_with_flag_optional(spectral_raw)
    spectral_no_qk_mean, spectral_no_qk_std, spectral_no_qk_n = nanmean_std_with_flag_optional(spectral_no_qk_raw)
    l2_mean, l2_std, l2_n = nanmean_std_with_flag_optional(l2_raw)

    margin_min_mean, margin_min_std, margin_min_n = nanmean_std_with_flag_optional(margin_min_raw)
    margin_mean_mean, margin_mean_std, margin_mean_n = nanmean_std_with_flag_optional(margin_mean_raw)
    margin_max_mean, margin_max_std, margin_max_n = nanmean_std_with_flag_optional(margin_max_raw)
    margin_std_mean, margin_std_std, margin_std_n = nanmean_std_with_flag_optional(margin_std_raw)

    # -----------------------------
    # Aggregate timestep arrays over seed axis=1
    # -----------------------------
    trainloss_timestep_mean, trainloss_timestep_std, trainloss_timestep_n = nanmean_std_with_flag_optional(trainloss_timestep_raw)
    testloss_timestep_mean, testloss_timestep_std, testloss_timestep_n = nanmean_std_with_flag_optional(testloss_timestep_raw)
    testacc_timestep_mean, testacc_timestep_std, testacc_timestep_n = nanmean_std_with_flag_optional(testacc_timestep_raw)
    err_timestep_mean, err_timestep_std, err_timestep_n = nanmean_std_with_flag_optional(err_timestep_raw)

    spectral_timestep_mean, spectral_timestep_std, spectral_timestep_n = nanmean_std_with_flag_optional(spectral_timestep_raw)
    spectral_no_qk_timestep_mean, spectral_no_qk_timestep_std, spectral_no_qk_timestep_n = nanmean_std_with_flag_optional(spectral_no_qk_timestep_raw)
    l2_timestep_mean, l2_timestep_std, l2_timestep_n = nanmean_std_with_flag_optional(l2_timestep_raw)

    margin_min_timestep_mean, margin_min_timestep_std, margin_min_timestep_n = nanmean_std_with_flag_optional(margin_min_timestep_raw)
    margin_mean_timestep_mean, margin_mean_timestep_std, margin_mean_timestep_n = nanmean_std_with_flag_optional(margin_mean_timestep_raw)
    margin_max_timestep_mean, margin_max_timestep_std, margin_max_timestep_n = nanmean_std_with_flag_optional(margin_max_timestep_raw)
    margin_std_timestep_mean, margin_std_timestep_std, margin_std_timestep_n = nanmean_std_with_flag_optional(margin_std_timestep_raw)

    # Scalar summaries
    best_loss_mean, best_loss_std, best_loss_n = nanmean_std_with_flag(best_loss_raw, axis=1)
    best_acc_mean, best_acc_std, best_acc_n = nanmean_std_with_flag(best_acc_raw, axis=1)
    best_epoch_mean, best_epoch_std, best_epoch_n = nanmean_std_with_flag(best_epoch_raw, axis=1)
    last_saved_epoch_mean, last_saved_epoch_std, last_saved_epoch_n = nanmean_std_with_flag(last_saved_epoch_raw, axis=1)

    # -----------------------------
    # Seed-resolved arrays: shape (nP, nT, max_seeds)
    # -----------------------------
    trainloss_seeds = np.transpose(trainloss_raw, (0, 2, 1))
    testloss_seeds = np.transpose(testloss_raw, (0, 2, 1))
    testacc_seeds = np.transpose(testacc_raw, (0, 2, 1))
    err_seeds = np.transpose(err_raw, (0, 2, 1))

    spectral_seeds = np.transpose(spectral_raw, (0, 2, 1))
    spectral_no_qk_seeds = np.transpose(spectral_no_qk_raw, (0, 2, 1))
    l2_seeds = np.transpose(l2_raw, (0, 2, 1))

    margin_min_seeds = np.transpose(margin_min_raw, (0, 2, 1))
    margin_mean_seeds = np.transpose(margin_mean_raw, (0, 2, 1))
    margin_max_seeds = np.transpose(margin_max_raw, (0, 2, 1))
    margin_std_seeds = np.transpose(margin_std_raw, (0, 2, 1))

    # -----------------------------
    # Timestep seed-resolved arrays: shape (nP, nTs, max_seeds)
    # -----------------------------
    trainloss_seeds_timestep = np.transpose(trainloss_timestep_raw, (0, 2, 1))
    testloss_seeds_timestep = np.transpose(testloss_timestep_raw, (0, 2, 1))
    testacc_seeds_timestep = np.transpose(testacc_timestep_raw, (0, 2, 1))
    err_seeds_timestep = np.transpose(err_timestep_raw, (0, 2, 1))

    spectral_seeds_timestep = np.transpose(spectral_timestep_raw, (0, 2, 1))
    spectral_no_qk_seeds_timestep = np.transpose(spectral_no_qk_timestep_raw, (0, 2, 1))
    l2_seeds_timestep = np.transpose(l2_timestep_raw, (0, 2, 1))

    margin_min_seeds_timestep = np.transpose(margin_min_timestep_raw, (0, 2, 1))
    margin_mean_seeds_timestep = np.transpose(margin_mean_timestep_raw, (0, 2, 1))
    margin_max_seeds_timestep = np.transpose(margin_max_timestep_raw, (0, 2, 1))
    margin_std_seeds_timestep = np.transpose(margin_std_timestep_raw, (0, 2, 1))

    T_arr = epoch_values.copy()
    T_arr_timestep = timestep_values.copy()

    result = {
        # identifiers
        "run_name": np.array(args.run_name),
        "experiment_name": np.array(experiment_name),

        # fixed experiment parameters
        "fixed_params": np.array(ref, dtype=object),

        # axes
        "P_values": P_values,                       # shape (nP,)
        "epoch_values": epoch_values,               # shape (nT,)
        "T_arr": T_arr,                             # shape (nT,)
        "timestep_values": timestep_values,         # shape (nTs,)
        "T_arr_timestep": T_arr_timestep,           # shape (nTs,)
        "num_seeds": num_seeds,                     # shape (nP,)
        "seed_triplets": seed_triplets,             # shape (nP, max_seeds, 3)

        # raw epoch curves per seed
        "trainloss_raw": trainloss_raw,             # shape (nP, max_seeds, nT)
        "testloss_raw": testloss_raw,
        "testacc_raw": testacc_raw,
        "err_raw": err_raw,

        "spectral_raw": spectral_raw,
        "spectral_no_qk_raw": spectral_no_qk_raw,
        "l2_raw": l2_raw,

        "margin_min_raw": margin_min_raw,
        "margin_mean_raw": margin_mean_raw,
        "margin_max_raw": margin_max_raw,
        "margin_std_raw": margin_std_raw,

        # raw timestep curves per seed
        "trainloss_timestep_raw": trainloss_timestep_raw,   # shape (nP, max_seeds, nTs)
        "testloss_timestep_raw": testloss_timestep_raw,
        "testacc_timestep_raw": testacc_timestep_raw,
        "err_timestep_raw": err_timestep_raw,

        "spectral_timestep_raw": spectral_timestep_raw,
        "spectral_no_qk_timestep_raw": spectral_no_qk_timestep_raw,
        "l2_timestep_raw": l2_timestep_raw,

        "margin_min_timestep_raw": margin_min_timestep_raw,
        "margin_mean_timestep_raw": margin_mean_timestep_raw,
        "margin_max_timestep_raw": margin_max_timestep_raw,
        "margin_std_timestep_raw": margin_std_timestep_raw,

        # seed-resolved epoch arrays
        "trainloss_seeds": trainloss_seeds,               # shape (nP, nT, max_seeds)
        "testloss_seeds": testloss_seeds,
        "testacc_seeds": testacc_seeds,
        "err_seeds": err_seeds,

        "spectral_seeds": spectral_seeds,
        "spectral_no_qk_seeds": spectral_no_qk_seeds,
        "l2_seeds": l2_seeds,

        "margin_min_seeds": margin_min_seeds,
        "margin_mean_seeds": margin_mean_seeds,
        "margin_max_seeds": margin_max_seeds,
        "margin_std_seeds": margin_std_seeds,

        # seed-resolved timestep arrays
        "trainloss_seeds_timestep": trainloss_seeds_timestep,   # shape (nP, nTs, max_seeds)
        "testloss_seeds_timestep": testloss_seeds_timestep,
        "testacc_seeds_timestep": testacc_seeds_timestep,
        "err_seeds_timestep": err_seeds_timestep,

        "spectral_seeds_timestep": spectral_seeds_timestep,
        "spectral_no_qk_seeds_timestep": spectral_no_qk_seeds_timestep,
        "l2_seeds_timestep": l2_seeds_timestep,

        "margin_min_seeds_timestep": margin_min_seeds_timestep,
        "margin_mean_seeds_timestep": margin_mean_seeds_timestep,
        "margin_max_seeds_timestep": margin_max_seeds_timestep,
        "margin_std_seeds_timestep": margin_std_seeds_timestep,

        # aggregated epoch curves
        "trainloss_mean": trainloss_mean,           # shape (nP, nT)
        "trainloss_std": trainloss_std,
        "trainloss_n": trainloss_n,

        "testloss_mean": testloss_mean,
        "testloss_std": testloss_std,
        "testloss_n": testloss_n,

        "testacc_mean": testacc_mean,
        "testacc_std": testacc_std,
        "testacc_n": testacc_n,

        "err_mean": err_mean,
        "err_std": err_std,
        "err_n": err_n,

        "spectral_mean": spectral_mean,
        "spectral_std": spectral_std,
        "spectral_n": spectral_n,

        "spectral_no_qk_mean": spectral_no_qk_mean,
        "spectral_no_qk_std": spectral_no_qk_std,
        "spectral_no_qk_n": spectral_no_qk_n,

        "l2_mean": l2_mean,
        "l2_std": l2_std,
        "l2_n": l2_n,

        "margin_min_mean": margin_min_mean,
        "margin_min_std": margin_min_std,
        "margin_min_n": margin_min_n,

        "margin_mean_mean": margin_mean_mean,
        "margin_mean_std": margin_mean_std,
        "margin_mean_n": margin_mean_n,

        "margin_max_mean": margin_max_mean,
        "margin_max_std": margin_max_std,
        "margin_max_n": margin_max_n,

        "margin_std_mean": margin_std_mean,
        "margin_std_std": margin_std_std,
        "margin_std_n": margin_std_n,

        # aggregated timestep curves
        "trainloss_timestep_mean": trainloss_timestep_mean,   # shape (nP, nTs)
        "trainloss_timestep_std": trainloss_timestep_std,
        "trainloss_timestep_n": trainloss_timestep_n,

        "testloss_timestep_mean": testloss_timestep_mean,
        "testloss_timestep_std": testloss_timestep_std,
        "testloss_timestep_n": testloss_timestep_n,

        "testacc_timestep_mean": testacc_timestep_mean,
        "testacc_timestep_std": testacc_timestep_std,
        "testacc_timestep_n": testacc_timestep_n,

        "err_timestep_mean": err_timestep_mean,
        "err_timestep_std": err_timestep_std,
        "err_timestep_n": err_timestep_n,

        "spectral_timestep_mean": spectral_timestep_mean,
        "spectral_timestep_std": spectral_timestep_std,
        "spectral_timestep_n": spectral_timestep_n,

        "spectral_no_qk_timestep_mean": spectral_no_qk_timestep_mean,
        "spectral_no_qk_timestep_std": spectral_no_qk_timestep_std,
        "spectral_no_qk_timestep_n": spectral_no_qk_timestep_n,

        "l2_timestep_mean": l2_timestep_mean,
        "l2_timestep_std": l2_timestep_std,
        "l2_timestep_n": l2_timestep_n,

        "margin_min_timestep_mean": margin_min_timestep_mean,
        "margin_min_timestep_std": margin_min_timestep_std,
        "margin_min_timestep_n": margin_min_timestep_n,

        "margin_mean_timestep_mean": margin_mean_timestep_mean,
        "margin_mean_timestep_std": margin_mean_timestep_std,
        "margin_mean_timestep_n": margin_mean_timestep_n,

        "margin_max_timestep_mean": margin_max_timestep_mean,
        "margin_max_timestep_std": margin_max_timestep_std,
        "margin_max_timestep_n": margin_max_timestep_n,

        "margin_std_timestep_mean": margin_std_timestep_mean,
        "margin_std_timestep_std": margin_std_timestep_std,
        "margin_std_timestep_n": margin_std_timestep_n,

        # raw scalar summaries per seed
        "best_loss_raw": best_loss_raw,                 # shape (nP, max_seeds)
        "best_acc_raw": best_acc_raw,
        "best_epoch_raw": best_epoch_raw,
        "last_saved_epoch_raw": last_saved_epoch_raw,

        # aggregated scalar summaries over seeds
        "best_loss_mean": best_loss_mean,               # shape (nP,)
        "best_loss_std": best_loss_std,
        "best_loss_n": best_loss_n,

        "best_acc_mean": best_acc_mean,
        "best_acc_std": best_acc_std,
        "best_acc_n": best_acc_n,

        "best_epoch_mean": best_epoch_mean,
        "best_epoch_std": best_epoch_std,
        "best_epoch_n": best_epoch_n,

        "last_saved_epoch_mean": last_saved_epoch_mean,
        "last_saved_epoch_std": last_saved_epoch_std,
        "last_saved_epoch_n": last_saved_epoch_n,

        # optional traceability
        "file_index": file_index,                       # shape (nP, max_seeds)
    }

    np.save(save_path, result, allow_pickle=True)

    print(f"Saved aggregated numpy dict to:\n  {save_path}")
    print(f"Found {len(files)} raw files")
    print(f"P values: {P_values.tolist()}")
    print(f"Global epoch checkpoints: {epoch_values.tolist()}")
    print(f"Global timestep checkpoints: {timestep_values.tolist()}")
    print(f"num_seeds per P: {num_seeds.tolist()}")

    print(f"trainloss_seeds shape: {trainloss_seeds.shape}")
    print(f"testloss_seeds shape: {testloss_seeds.shape}")
    print(f"testacc_seeds shape: {testacc_seeds.shape}")
    print(f"err_seeds shape: {err_seeds.shape}")
    print(f"spectral_seeds shape: {spectral_seeds.shape}")
    print(f"spectral_no_qk_seeds shape: {spectral_no_qk_seeds.shape}")
    print(f"l2_seeds shape: {l2_seeds.shape}")
    print(f"margin_min_seeds shape: {margin_min_seeds.shape}")
    print(f"margin_mean_seeds shape: {margin_mean_seeds.shape}")
    print(f"margin_max_seeds shape: {margin_max_seeds.shape}")
    print(f"margin_std_seeds shape: {margin_std_seeds.shape}")

    print(f"trainloss_seeds_timestep shape: {trainloss_seeds_timestep.shape}")
    print(f"testloss_seeds_timestep shape: {testloss_seeds_timestep.shape}")
    print(f"testacc_seeds_timestep shape: {testacc_seeds_timestep.shape}")
    print(f"err_seeds_timestep shape: {err_seeds_timestep.shape}")
    print(f"spectral_seeds_timestep shape: {spectral_seeds_timestep.shape}")
    print(f"spectral_no_qk_seeds_timestep shape: {spectral_no_qk_seeds_timestep.shape}")
    print(f"l2_seeds_timestep shape: {l2_seeds_timestep.shape}")
    print(f"margin_min_seeds_timestep shape: {margin_min_seeds_timestep.shape}")
    print(f"margin_mean_seeds_timestep shape: {margin_mean_seeds_timestep.shape}")
    print(f"margin_max_seeds_timestep shape: {margin_max_seeds_timestep.shape}")
    print(f"margin_std_seeds_timestep shape: {margin_std_seeds_timestep.shape}")


if __name__ == "__main__":
    main()