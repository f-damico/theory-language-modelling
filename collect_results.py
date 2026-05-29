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
    "batch_size",              # can differ across jobs; warn but still collect
    "save_freq",               # only changes checkpoint/write cadence; warn but still collect
    "save_trainstep_epochs",
    "weight_save_every",        # only changes extra checkpoint artifacts
    "save_run_data",
    "save_processed_dataset_inputs",
    "save_data_subset_train_size",
    "save_data_subset_test_size",
    "save_data_subset_seed",
    *SEED_KEYS,
}

WARNING_ONLY_COMPARE_KEYS = ("batch_size", "save_freq")

SPECTRAL_KEY = "specnorm"
L2_KEY = "l2norm"
SPECTRAL_NO_QK_KEY = "specnorm_no_qk"

MARGIN_MIN_KEY = "margin_min"
MARGIN_MEAN_KEY = "margin_mean"
MARGIN_MAX_KEY = "margin_max"
MARGIN_STD_KEY = "margin_std"

# Scalar diagnostics already present in the old collector.
SCALAR_DYNAMICS_KEYS = (
    "trainloss",
    "trainacc",
    "trainerr",
    "testloss",
    "testacc",
    "err",
    "spectral",
    "spectral_no_qk",
    "l2",
    "margin_min",
    "margin_mean",
    "margin_max",
    "margin_std",
)

# New RHM level-wise M_l diagnostics saved by the modified transformer code.
# For each split, the raw training files contain keys like:
#   train_rhm_M_mean, test_rhm_M_mean, ...
# Each value is a vector of length L = num_layers.
RHM_ML_VECTOR_KEYS = (
    "rhm_M_mean",
    "rhm_M_pos_frac",
    "rhm_survival_mean",
    "rhm_level_penalty_mean",
)
RHM_SPLITS = ("train", "test")

# Logit-cloud effective dimension diagnostics saved by main.py at checkpoints.
LOGIT_EFFDIM_SCALAR_KEYS = (
    "logit_energy_mean",
    "logit_input_variance",
    "logit_effdim_entropy",
    "logit_effdim_pr",
    "logit_effdim_entropy_norm",
    "logit_effdim_pr_norm",
    "logit_effdim_num_samples",
)

SCALAR_DYNAMICS_KEYS = SCALAR_DYNAMICS_KEYS + tuple(
    f"{split}_{key}"
    for split in RHM_SPLITS
    for key in LOGIT_EFFDIM_SCALAR_KEYS
)


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


def _to_1d_float(value, length=None):
    if value is None:
        if length is None:
            return np.array([], dtype=float)
        return np.full(int(length), np.nan, dtype=float)

    arr = np.asarray(value, dtype=float).reshape(-1)
    if length is None:
        return arr

    out = np.full(int(length), np.nan, dtype=float)
    n = min(out.size, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _to_1d_int(value, length=None):
    if value is None:
        if length is None:
            return np.array([], dtype=int)
        return np.full(int(length), -1, dtype=int)

    arr = np.asarray(value, dtype=int).reshape(-1)
    if length is None:
        return arr

    out = np.full(int(length), -1, dtype=int)
    n = min(out.size, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _infer_num_rhm_levels(dyn):
    """
    Infer L from one raw dynamics list.

    Priority:
    1. train_rhm_levels / test_rhm_levels;
    2. any vector diagnostic such as train_rhm_M_mean;
    3. zero if not available.
    """
    for d in dyn:
        for split in RHM_SPLITS:
            levels = d.get(f"{split}_rhm_levels")
            if levels is not None:
                arr = np.asarray(levels).reshape(-1)
                if arr.size > 0:
                    return int(arr.size)

        for split in RHM_SPLITS:
            for key in RHM_ML_VECTOR_KEYS:
                value = d.get(f"{split}_{key}")
                if value is not None:
                    arr = np.asarray(value).reshape(-1)
                    if arr.size > 0:
                        return int(arr.size)

    return 0


def dynamics_to_arrays(output, dyn_key="dynamics", time_name="epochs"):
    dyn = output.get(dyn_key, [])
    if dyn is None:
        dyn = []

    n_levels = _infer_num_rhm_levels(dyn)

    if len(dyn) == 0:
        out = {
            time_name: np.array([], dtype=int),
            "trainloss": np.array([], dtype=float),
            "trainacc": np.array([], dtype=float),
            "trainerr": np.array([], dtype=float),
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
            "rhm_num_levels": np.array(0, dtype=int),
        }
        for split in RHM_SPLITS:
            for key in RHM_ML_VECTOR_KEYS:
                out[f"{split}_{key}"] = np.full((0, 0), np.nan, dtype=float)
            out[f"{split}_rhm_levels"] = np.full((0, 0), -1, dtype=int)
            out[f"{split}_rhm_margin_num_samples"] = np.array([], dtype=float)
            for key in LOGIT_EFFDIM_SCALAR_KEYS:
                out[f"{split}_{key}"] = np.array([], dtype=float)
        return out

    times = np.array(
        [int(d.get("t", d.get("global_update", -1))) for d in dyn],
        dtype=int,
    )

    trainloss = np.array([d["trainloss"] for d in dyn], dtype=float)

    trainacc = np.array([d.get("trainacc", np.nan) for d in dyn], dtype=float)
    trainerr = np.array(
        [
            d.get(
                "trainerr",
                (1.0 - d["trainacc"]) if ("trainacc" in d and np.isfinite(d["trainacc"])) else np.nan,
            )
            for d in dyn
        ],
        dtype=float,
    )

    testloss = np.array([d["testloss"] for d in dyn], dtype=float)
    testacc = np.array([d["testacc"] for d in dyn], dtype=float)
    err = np.array([d.get("err", 1.0 - d["testacc"]) for d in dyn], dtype=float)

    out = {
        time_name: times,
        "trainloss": trainloss,
        "trainacc": trainacc,
        "trainerr": trainerr,
        "testloss": testloss,
        "testacc": testacc,
        "err": err,
        "spectral": np.array([d.get(SPECTRAL_KEY, np.nan) for d in dyn], dtype=float),
        "spectral_no_qk": np.array([d.get(SPECTRAL_NO_QK_KEY, np.nan) for d in dyn], dtype=float),
        "l2": np.array([d.get(L2_KEY, np.nan) for d in dyn], dtype=float),
        "margin_min": np.array([d.get(MARGIN_MIN_KEY, np.nan) for d in dyn], dtype=float),
        "margin_mean": np.array([d.get(MARGIN_MEAN_KEY, np.nan) for d in dyn], dtype=float),
        "margin_max": np.array([d.get(MARGIN_MAX_KEY, np.nan) for d in dyn], dtype=float),
        "margin_std": np.array([d.get(MARGIN_STD_KEY, np.nan) for d in dyn], dtype=float),
        "rhm_num_levels": np.array(n_levels, dtype=int),
    }

    for split in RHM_SPLITS:
        for key in LOGIT_EFFDIM_SCALAR_KEYS:
            full_key = f"{split}_{key}"
            out[full_key] = np.array([d.get(full_key, np.nan) for d in dyn], dtype=float)

    for split in RHM_SPLITS:
        for key in RHM_ML_VECTOR_KEYS:
            full_key = f"{split}_{key}"
            out[full_key] = np.stack(
                [_to_1d_float(d.get(full_key), length=n_levels) for d in dyn],
                axis=0,
            ) if n_levels > 0 else np.full((len(dyn), 0), np.nan, dtype=float)

        levels_key = f"{split}_rhm_levels"
        out[levels_key] = np.stack(
            [
                _to_1d_int(d.get(levels_key, np.arange(1, n_levels + 1)), length=n_levels)
                for d in dyn
            ],
            axis=0,
        ) if n_levels > 0 else np.full((len(dyn), 0), -1, dtype=int)

        n_key = f"{split}_rhm_margin_num_samples"
        out[n_key] = np.array([d.get(n_key, np.nan) for d in dyn], dtype=float)

    return out


def comparable_params(args_dict):
    return {k: v for k, v in args_dict.items() if k not in IGNORE_COMPARE_KEYS}


def _warn_if_warning_only_params_differ(entries):
    """Print a non-fatal warning for parameters that are allowed to differ."""
    for key in WARNING_ONLY_COMPARE_KEYS:
        values_to_paths = {}
        for e in entries:
            value_repr = repr(e["args"].get(key, None))
            values_to_paths.setdefault(value_repr, []).append(e["path"])

        if len(values_to_paths) <= 1:
            continue

        print(
            f"[WARNING] Different values of {key!r} found across result files. "
            "This is allowed and aggregation will continue, but runs were not "
            "produced with exactly the same batch size."
        )
        for value_repr, paths in sorted(values_to_paths.items()):
            example = paths[0]
            print(
                f"[WARNING]   {key}={value_repr}: {len(paths)} file(s); "
                f"example: {example}"
            )


def nanmean_std_with_flag(x, axis):
    valid = ~np.isnan(x)
    counts = valid.sum(axis=axis)

    if x.size == 0 or x.shape[axis] == 0:
        out_shape = tuple(s for i, s in enumerate(x.shape) if i != axis)
        return (
            np.full(out_shape, np.nan, dtype=float),
            np.full(out_shape, -1.0, dtype=float),
            np.zeros(out_shape, dtype=int),
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(x, axis=axis)
        std = np.nanstd(x, axis=axis, ddof=1)

    std = np.where(counts < 2, -1.0, std)
    mean = np.where(counts == 0, np.nan, mean)

    return mean, std, counts


def nanmean_std_with_flag_optional(raw):
    """
    raw has shape (nP, max_seeds, nT, optional_level_dim...).
    Reduces over seed axis=1.
    """
    return nanmean_std_with_flag(raw, axis=1)


def _allocate_raw(nP, max_seeds, nT, n_levels=None, dtype=float, fill_value=np.nan):
    if n_levels is None:
        shape = (nP, max_seeds, nT)
    else:
        shape = (nP, max_seeds, nT, n_levels)
    return np.full(shape, fill_value, dtype=dtype)


def _fill_vector(raw, iP, iseed, j, value):
    if raw.ndim != 4:
        return
    arr = np.asarray(value, dtype=float).reshape(-1)
    n = min(raw.shape[-1], arr.size)
    if n > 0:
        raw[iP, iseed, j, :n] = arr[:n]


def _fill_int_vector(raw, iP, iseed, j, value):
    if raw.ndim != 4:
        return
    arr = np.asarray(value, dtype=int).reshape(-1)
    n = min(raw.shape[-1], arr.size)
    if n > 0:
        raw[iP, iseed, j, :n] = arr[:n]


def _infer_global_num_levels(entries, suffix=""):
    max_L = 0
    for e in entries:
        for split in RHM_SPLITS:
            for key in RHM_ML_VECTOR_KEYS:
                arr = np.asarray(e.get(f"{split}_{key}{suffix}", np.empty((0, 0))))
                if arr.ndim >= 2:
                    max_L = max(max_L, int(arr.shape[-1]))
    return max_L


def _transpose_seed_axis(raw):
    if raw.ndim == 3:
        return np.transpose(raw, (0, 2, 1))
    if raw.ndim == 4:
        return np.transpose(raw, (0, 2, 1, 3))
    raise ValueError(f"Unsupported raw ndim: {raw.ndim}")


def _add_metric_result(result, name, raw, *, timestep=False):
    """
    Add raw, seed-resolved, and aggregated versions of one metric.

    For epoch metrics:
        name_raw, name_seeds, name_mean, name_std, name_n
    For timestep metrics:
        name_timestep_raw, name_seeds_timestep, name_timestep_mean, ...
    """
    mean, std, n = nanmean_std_with_flag_optional(raw)
    seeds = _transpose_seed_axis(raw)

    if timestep:
        result[f"{name}_timestep_raw"] = raw
        result[f"{name}_seeds_timestep"] = seeds
        result[f"{name}_timestep_mean"] = mean
        result[f"{name}_timestep_std"] = std
        result[f"{name}_timestep_n"] = n
    else:
        result[f"{name}_raw"] = raw
        result[f"{name}_seeds"] = seeds
        result[f"{name}_mean"] = mean
        result[f"{name}_std"] = std
        result[f"{name}_n"] = n

    return seeds, mean, std, n


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

    # New runs are stored as one subfolder per run, with the old-style metrics
    # .pkl inside that folder.  Keep compatibility with the old flat layout by
    # searching recursively.  Checkpoints are .pt files, so they are ignored here.
    files = sorted(run_dir.rglob("*.pkl"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl files found recursively in {run_dir}")

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
            "epochs": dyn["epochs"],
            "timesteps": dyn_timestep["timesteps"],
            "best_loss": float(best.get("loss", np.nan)),
            "best_acc": float(best.get("acc", np.nan)),
            "best_epoch": float(best.get("epoch", np.nan)),
            "last_saved_epoch": float(output.get("epoch", np.nan)),
        }

        for key in SCALAR_DYNAMICS_KEYS:
            entry[key] = dyn[key]
            entry[f"{key}_timestep"] = dyn_timestep[key]

        for split in RHM_SPLITS:
            for key in RHM_ML_VECTOR_KEYS:
                entry[f"{split}_{key}"] = dyn[f"{split}_{key}"]
                entry[f"{split}_{key}_timestep"] = dyn_timestep[f"{split}_{key}"]
            entry[f"{split}_rhm_levels"] = dyn[f"{split}_rhm_levels"]
            entry[f"{split}_rhm_levels_timestep"] = dyn_timestep[f"{split}_rhm_levels"]
            entry[f"{split}_rhm_margin_num_samples"] = dyn[f"{split}_rhm_margin_num_samples"]
            entry[f"{split}_rhm_margin_num_samples_timestep"] = dyn_timestep[f"{split}_rhm_margin_num_samples"]

        entries.append(entry)

    # Check that all non-seed, non-train_size, warning-only params are identical.
    # In particular, batch_size is allowed to differ: it changes optimization
    # details, but the collector can still aggregate the files on the same grids.
    _warn_if_warning_only_params_differ(entries)

    ref = entries[0]["params_compare"]
    for e in entries[1:]:
        if e["params_compare"] != ref:
            raise ValueError(
                "Files in this run folder do not belong to one single experiment family.\n"
                "At least one non-seed/non-train_size/non-warning-only hyperparameter differs.\n"
                f"Warning-only keys: {WARNING_ONLY_COMPARE_KEYS}\n"
                f"Reference params:\n{ref}\n\nDifferent params:\n{e['params_compare']}\n\n"
                f"Problematic file: {e['path']}"
            )

    P_values = np.array(sorted({e["train_size"] for e in entries}), dtype=int)
    epoch_values = np.array(sorted({int(t) for e in entries for t in e["epochs"]}), dtype=int)
    timestep_values = np.array(sorted({int(t) for e in entries for t in e["timesteps"]}), dtype=int)

    entries_by_P = {int(P): [] for P in P_values}
    for e in entries:
        entries_by_P[int(e["train_size"])].append(e)

    for P in P_values:
        entries_by_P[int(P)] = sorted(
            entries_by_P[int(P)],
            key=lambda e: (e["seed_rules"], e["seed_sample"], e["seed_model"], e["path"]),
        )

    max_seeds = max(len(v) for v in entries_by_P.values())
    nP = len(P_values)
    nT = len(epoch_values)
    nTs = len(timestep_values)

    n_ml_levels = _infer_global_num_levels(entries, suffix="")
    n_ml_levels_timestep = _infer_global_num_levels(entries, suffix="_timestep")

    epoch_to_idx = {int(t): i for i, t in enumerate(epoch_values)}
    timestep_to_idx = {int(t): i for i, t in enumerate(timestep_values)}

    # Raw epoch arrays for scalar diagnostics.
    raw = {key: _allocate_raw(nP, max_seeds, nT) for key in SCALAR_DYNAMICS_KEYS}

    # Raw timestep arrays for scalar diagnostics.
    raw_timestep = {key: _allocate_raw(nP, max_seeds, nTs) for key in SCALAR_DYNAMICS_KEYS}

    # Raw epoch arrays for M_l diagnostics.
    ml_raw = {}
    ml_raw_timestep = {}
    for split in RHM_SPLITS:
        for key in RHM_ML_VECTOR_KEYS:
            ml_raw[f"{split}_{key}"] = _allocate_raw(nP, max_seeds, nT, n_levels=n_ml_levels)
            ml_raw_timestep[f"{split}_{key}"] = _allocate_raw(nP, max_seeds, nTs, n_levels=n_ml_levels_timestep)
        ml_raw[f"{split}_rhm_levels"] = _allocate_raw(nP, max_seeds, nT, n_levels=n_ml_levels, dtype=int, fill_value=-1)
        ml_raw_timestep[f"{split}_rhm_levels"] = _allocate_raw(nP, max_seeds, nTs, n_levels=n_ml_levels_timestep, dtype=int, fill_value=-1)
        ml_raw[f"{split}_rhm_margin_num_samples"] = _allocate_raw(nP, max_seeds, nT)
        ml_raw_timestep[f"{split}_rhm_margin_num_samples"] = _allocate_raw(nP, max_seeds, nTs)

    best_loss_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    best_acc_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    best_epoch_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    last_saved_epoch_raw = np.full((nP, max_seeds), np.nan, dtype=float)

    seed_triplets = np.full((nP, max_seeds, 3), -1, dtype=int)
    num_seeds = np.zeros(nP, dtype=int)
    file_index = np.full((nP, max_seeds), "", dtype=object)

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
                for key in SCALAR_DYNAMICS_KEYS:
                    raw[key][iP, iseed, j] = e[key][local_i]

                for split in RHM_SPLITS:
                    for key in RHM_ML_VECTOR_KEYS:
                        _fill_vector(ml_raw[f"{split}_{key}"], iP, iseed, j, e[f"{split}_{key}"][local_i])
                    _fill_int_vector(ml_raw[f"{split}_rhm_levels"], iP, iseed, j, e[f"{split}_rhm_levels"][local_i])
                    ml_raw[f"{split}_rhm_margin_num_samples"][iP, iseed, j] = e[f"{split}_rhm_margin_num_samples"][local_i]

            for local_i, ts in enumerate(e["timesteps"]):
                j = timestep_to_idx[int(ts)]
                for key in SCALAR_DYNAMICS_KEYS:
                    raw_timestep[key][iP, iseed, j] = e[f"{key}_timestep"][local_i]

                for split in RHM_SPLITS:
                    for key in RHM_ML_VECTOR_KEYS:
                        _fill_vector(ml_raw_timestep[f"{split}_{key}"], iP, iseed, j, e[f"{split}_{key}_timestep"][local_i])
                    _fill_int_vector(ml_raw_timestep[f"{split}_rhm_levels"], iP, iseed, j, e[f"{split}_rhm_levels_timestep"][local_i])
                    ml_raw_timestep[f"{split}_rhm_margin_num_samples"][iP, iseed, j] = e[f"{split}_rhm_margin_num_samples_timestep"][local_i]

            best_loss_raw[iP, iseed] = e["best_loss"]
            best_acc_raw[iP, iseed] = e["best_acc"]
            best_epoch_raw[iP, iseed] = e["best_epoch"]
            last_saved_epoch_raw[iP, iseed] = e["last_saved_epoch"]

    # Keep the original convention: derive errors from test accuracy.
    raw["err"] = 1.0 - raw["testacc"]
    raw_timestep["err"] = 1.0 - raw_timestep["testacc"]

    result = {
        "run_name": np.array(args.run_name),
        "experiment_name": np.array(experiment_name),
        "fixed_params": np.array(ref, dtype=object),
        "P_values": P_values,
        "epoch_values": epoch_values,
        "T_arr": epoch_values.copy(),
        "timestep_values": timestep_values,
        "T_arr_timestep": timestep_values.copy(),
        "num_seeds": num_seeds,
        "seed_triplets": seed_triplets,
        "rhm_M_l_num_levels": np.array(n_ml_levels, dtype=int),
        "rhm_M_l_num_levels_timestep": np.array(n_ml_levels_timestep, dtype=int),
    }

    # Old scalar epoch and timestep outputs, with the same key names as before.
    saved_shapes = {}
    for key in SCALAR_DYNAMICS_KEYS:
        seeds, mean, std, n = _add_metric_result(result, key, raw[key], timestep=False)
        saved_shapes[f"{key}_seeds"] = seeds.shape
        seeds_ts, mean_ts, std_ts, n_ts = _add_metric_result(result, key, raw_timestep[key], timestep=True)
        saved_shapes[f"{key}_seeds_timestep"] = seeds_ts.shape

    # New M_l epoch and timestep outputs.
    for split in RHM_SPLITS:
        for key in RHM_ML_VECTOR_KEYS:
            name = f"{split}_{key}"
            seeds, mean, std, n = _add_metric_result(result, name, ml_raw[name], timestep=False)
            saved_shapes[f"{name}_seeds"] = seeds.shape

            seeds_ts, mean_ts, std_ts, n_ts = _add_metric_result(result, name, ml_raw_timestep[name], timestep=True)
            saved_shapes[f"{name}_seeds_timestep"] = seeds_ts.shape

        # Metadata: levels and number of evaluated samples.
        levels_name = f"{split}_rhm_levels"
        result[f"{levels_name}_raw"] = ml_raw[levels_name]
        result[f"{levels_name}_seeds"] = _transpose_seed_axis(ml_raw[levels_name])
        result[f"{levels_name}_timestep_raw"] = ml_raw_timestep[levels_name]
        result[f"{levels_name}_seeds_timestep"] = _transpose_seed_axis(ml_raw_timestep[levels_name])

        ns_name = f"{split}_rhm_margin_num_samples"
        result[f"{ns_name}_raw"] = ml_raw[ns_name]
        result[f"{ns_name}_seeds"] = _transpose_seed_axis(ml_raw[ns_name])
        result[f"{ns_name}_timestep_raw"] = ml_raw_timestep[ns_name]
        result[f"{ns_name}_seeds_timestep"] = _transpose_seed_axis(ml_raw_timestep[ns_name])

    # Scalar summaries.
    best_loss_mean, best_loss_std, best_loss_n = nanmean_std_with_flag(best_loss_raw, axis=1)
    best_acc_mean, best_acc_std, best_acc_n = nanmean_std_with_flag(best_acc_raw, axis=1)
    best_epoch_mean, best_epoch_std, best_epoch_n = nanmean_std_with_flag(best_epoch_raw, axis=1)
    last_saved_epoch_mean, last_saved_epoch_std, last_saved_epoch_n = nanmean_std_with_flag(last_saved_epoch_raw, axis=1)

    result.update({
        "best_loss_raw": best_loss_raw,
        "best_acc_raw": best_acc_raw,
        "best_epoch_raw": best_epoch_raw,
        "last_saved_epoch_raw": last_saved_epoch_raw,
        "best_loss_mean": best_loss_mean,
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
        "file_index": file_index,
    })

    np.save(save_path, result, allow_pickle=True)

    print(f"Saved aggregated numpy dict to:\n  {save_path}")
    print(f"Found {len(files)} raw files")
    print(f"P values: {P_values.tolist()}")
    print(f"Global epoch checkpoints: {epoch_values.tolist()}")
    print(f"Global timestep checkpoints: {timestep_values.tolist()}")
    print(f"num_seeds per P: {num_seeds.tolist()}")

    print(f"trainloss_seeds shape: {result['trainloss_seeds'].shape}")
    print(f"trainacc_seeds shape: {result['trainacc_seeds'].shape}")
    print(f"trainerr_seeds shape: {result['trainerr_seeds'].shape}")
    print(f"testloss_seeds shape: {result['testloss_seeds'].shape}")
    print(f"testacc_seeds shape: {result['testacc_seeds'].shape}")
    print(f"err_seeds shape: {result['err_seeds'].shape}")
    print(f"spectral_seeds shape: {result['spectral_seeds'].shape}")
    print(f"spectral_no_qk_seeds shape: {result['spectral_no_qk_seeds'].shape}")
    print(f"l2_seeds shape: {result['l2_seeds'].shape}")
    print(f"margin_min_seeds shape: {result['margin_min_seeds'].shape}")
    print(f"margin_mean_seeds shape: {result['margin_mean_seeds'].shape}")
    print(f"margin_max_seeds shape: {result['margin_max_seeds'].shape}")
    print(f"margin_std_seeds shape: {result['margin_std_seeds'].shape}")
    print(f"train_logit_effdim_entropy_seeds shape: {result['train_logit_effdim_entropy_seeds'].shape}")
    print(f"test_logit_effdim_entropy_seeds shape: {result['test_logit_effdim_entropy_seeds'].shape}")

    if n_ml_levels > 0:
        print(f"RHM M_l diagnostics detected with L={n_ml_levels}")
        for split in RHM_SPLITS:
            print(f"{split}_rhm_M_mean_seeds shape: {result[f'{split}_rhm_M_mean_seeds'].shape}")
            print(f"{split}_rhm_M_pos_frac_seeds shape: {result[f'{split}_rhm_M_pos_frac_seeds'].shape}")
            print(f"{split}_rhm_survival_mean_seeds shape: {result[f'{split}_rhm_survival_mean_seeds'].shape}")
            print(f"{split}_rhm_level_penalty_mean_seeds shape: {result[f'{split}_rhm_level_penalty_mean_seeds'].shape}")
    else:
        print("RHM M_l diagnostics not detected in epoch dynamics; saved empty arrays with L=0.")

    print(f"trainloss_seeds_timestep shape: {result['trainloss_seeds_timestep'].shape}")
    print(f"trainacc_seeds_timestep shape: {result['trainacc_seeds_timestep'].shape}")
    print(f"trainerr_seeds_timestep shape: {result['trainerr_seeds_timestep'].shape}")
    print(f"testloss_seeds_timestep shape: {result['testloss_seeds_timestep'].shape}")
    print(f"testacc_seeds_timestep shape: {result['testacc_seeds_timestep'].shape}")
    print(f"err_seeds_timestep shape: {result['err_seeds_timestep'].shape}")
    print(f"spectral_seeds_timestep shape: {result['spectral_seeds_timestep'].shape}")
    print(f"spectral_no_qk_seeds_timestep shape: {result['spectral_no_qk_seeds_timestep'].shape}")
    print(f"l2_seeds_timestep shape: {result['l2_seeds_timestep'].shape}")
    print(f"margin_min_seeds_timestep shape: {result['margin_min_seeds_timestep'].shape}")
    print(f"margin_mean_seeds_timestep shape: {result['margin_mean_seeds_timestep'].shape}")
    print(f"margin_max_seeds_timestep shape: {result['margin_max_seeds_timestep'].shape}")
    print(f"margin_std_seeds_timestep shape: {result['margin_std_seeds_timestep'].shape}")

    if n_ml_levels_timestep > 0:
        print(f"RHM M_l timestep diagnostics detected with L={n_ml_levels_timestep}")
        for split in RHM_SPLITS:
            print(f"{split}_rhm_M_mean_seeds_timestep shape: {result[f'{split}_rhm_M_mean_seeds_timestep'].shape}")


if __name__ == "__main__":
    main()
