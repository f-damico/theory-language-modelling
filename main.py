import argparse
import copy
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import datasets
import init
import measures
import models
import training

# -----------------------------------------------------------------------------
# Run-folder saving utilities: compact data files + resume-ready checkpoints
# -----------------------------------------------------------------------------


def _none_or_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value_str = str(value).strip().lower()
    if value_str in {'none', 'null', 'no', 'false', '-1', '0'}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def _args_dict(args):
    out = vars(args).copy() if hasattr(args, '__dict__') else dict(args)
    for k, v in list(out.items()):
        if isinstance(v, Path):
            out[k] = str(v)
    return out


def _unwrap_subset(dataset):
    """Return base dataset and the base-dataset indices represented by a Subset."""
    if isinstance(dataset, torch.utils.data.Subset):
        base, parent_indices = _unwrap_subset(dataset.dataset)
        child_indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return base, parent_indices[child_indices]
    return dataset, torch.arange(len(dataset), dtype=torch.long)


def _tensor_to_numpy(x, indices=None):
    if x is None:
        return None
    if torch.is_tensor(x):
        if indices is not None:
            idx = indices.to(device=x.device, dtype=torch.long) if torch.is_tensor(indices) else torch.as_tensor(indices, dtype=torch.long, device=x.device)
            x = x[idx]
        return x.detach().cpu().numpy()
    arr = np.asarray(x)
    if indices is not None:
        idx = indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices)
        arr = arr[np.asarray(idx, dtype=np.int64)]
    return arr


def _compact_int_array(arr):
    arr = np.asarray(arr)
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.integer):
        return arr
    mn = int(arr.min())
    mx = int(arr.max())
    if mn >= 0:
        if mx <= np.iinfo(np.uint8).max:
            return arr.astype(np.uint8, copy=False)
        if mx <= np.iinfo(np.uint16).max:
            return arr.astype(np.uint16, copy=False)
        if mx <= np.iinfo(np.uint32).max:
            return arr.astype(np.uint32, copy=False)
    if mn >= np.iinfo(np.int16).min and mx <= np.iinfo(np.int16).max:
        return arr.astype(np.int16, copy=False)
    if mn >= np.iinfo(np.int32).min and mx <= np.iinfo(np.int32).max:
        return arr.astype(np.int32, copy=False)
    return arr.astype(np.int64, copy=False)


def _safe_take_attr(base_dataset, attr_name, indices):
    if not hasattr(base_dataset, attr_name):
        return None
    value = getattr(base_dataset, attr_name)
    try:
        return _tensor_to_numpy(value, indices)
    except Exception:
        return None


def _select_reference_indices(num_items, requested, seed):
    num_items = int(num_items)
    if requested is None or int(requested) < 0 or int(requested) >= num_items:
        return np.arange(num_items, dtype=np.int64)
    requested = int(requested)
    if requested == 0 or num_items == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(num_items, size=requested, replace=False)).astype(np.int64)


def _build_dataset_npz_payload(train_loader, test_loader, args, *, subset=False):
    train_base, train_indices = _unwrap_subset(train_loader.dataset)
    if test_loader is not None:
        test_base, test_indices = _unwrap_subset(test_loader.dataset)
    else:
        test_base, test_indices = train_base, torch.empty(0, dtype=torch.long)

    if subset:
        seed = int(getattr(args, 'save_data_subset_seed', -1))
        if seed < 0:
            seed = int(getattr(args, 'seed_sample', 0))
        tr_sel = _select_reference_indices(
            len(train_indices),
            getattr(args, 'save_data_subset_train_size', 1024),
            seed,
        )
        te_sel = _select_reference_indices(
            len(test_indices),
            getattr(args, 'save_data_subset_test_size', 1024),
            seed + 1,
        )
        train_indices_use = train_indices[torch.as_tensor(tr_sel, dtype=torch.long)]
        test_indices_use = test_indices[torch.as_tensor(te_sel, dtype=torch.long)]
    else:
        tr_sel = np.arange(len(train_indices), dtype=np.int64)
        te_sel = np.arange(len(test_indices), dtype=np.int64)
        train_indices_use = train_indices
        test_indices_use = test_indices

    payload = {
        'format_version': np.array(1, dtype=np.int64),
        'subset': np.array(bool(subset)),
        'args_json': np.array(json.dumps(_args_dict(args), sort_keys=True, default=_json_default)),
        'train_indices_in_split': _compact_int_array(tr_sel),
        'test_indices_in_split': _compact_int_array(te_sel),
        'train_global_indices': _compact_int_array(train_indices_use.cpu().numpy()),
        'test_global_indices': _compact_int_array(test_indices_use.cpu().numpy()),
    }

    # Always save the exact targets used by the loss.
    train_targets = _safe_take_attr(train_base, 'labels', train_indices_use)
    test_targets = _safe_take_attr(test_base, 'labels', test_indices_use) if test_loader is not None else np.empty(0, dtype=np.int64)
    if train_targets is not None:
        payload['train_targets'] = _compact_int_array(train_targets)
    if test_targets is not None:
        payload['test_targets'] = _compact_int_array(test_targets)

    # For RHM this is the compact exact dataset: terminal token sequences + rules.
    for attr in ('rhm_sequences', 'rhm_root_labels', 'sample_ids'):
        tr = _safe_take_attr(train_base, attr, train_indices_use)
        te = _safe_take_attr(test_base, attr, test_indices_use) if test_loader is not None else None
        if tr is not None:
            payload[f'train_{attr}'] = _compact_int_array(tr)
        if te is not None:
            payload[f'test_{attr}'] = _compact_int_array(te)

    if hasattr(train_base, 'rules'):
        rules = getattr(train_base, 'rules')
        if isinstance(rules, dict):
            for level, rule in sorted(rules.items(), key=lambda kv: int(kv[0])):
                payload[f'rules_level_{int(level)}'] = _compact_int_array(_tensor_to_numpy(rule))

    # Optional fallback/exact processed model inputs. This can be very large, so it is off by default.
    if getattr(args, 'save_processed_dataset_inputs', False):
        tr_x = _safe_take_attr(train_base, 'features', train_indices_use)
        te_x = _safe_take_attr(test_base, 'features', test_indices_use) if test_loader is not None else None
        if tr_x is not None:
            payload['train_inputs'] = tr_x
        if te_x is not None:
            payload['test_inputs'] = te_x

    return payload


def _save_npz_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **payload)
    tmp.replace(path)


def _save_run_datasets(run_dir, train_loader, test_loader, args):
    data_dir = Path(run_dir) / 'data'
    full_path = data_dir / 'dataset_full.npz'
    subset_path = data_dir / 'dataset_reference_subset.npz'

    if getattr(args, 'save_run_data', True):
        _save_npz_atomic(full_path, _build_dataset_npz_payload(train_loader, test_loader, args, subset=False))
        print(f'[INFO] saved full compact dataset to {full_path}')

    if getattr(args, 'save_data_subset_train_size', 1024) != 0 or getattr(args, 'save_data_subset_test_size', 1024) != 0:
        _save_npz_atomic(subset_path, _build_dataset_npz_payload(train_loader, test_loader, args, subset=True))
        print(f'[INFO] saved reference data subset to {subset_path}')

    return {'dataset_full': str(full_path), 'dataset_reference_subset': str(subset_path)}


def _state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def _optimizer_state_to_cpu(state):
    if torch.is_tensor(state):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {k: _optimizer_state_to_cpu(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_optimizer_state_to_cpu(v) for v in state]
    if isinstance(state, tuple):
        return tuple(_optimizer_state_to_cpu(v) for v in state)
    return state


def _rng_state_dict():
    state = {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    return state


def _save_resume_checkpoint(path, *, model, optimizer, scheduler, args, epoch, global_update, best, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'format_version': 1,
        'epoch': int(epoch),
        'global_update': int(global_update),
        'model_state_dict': _state_dict_to_cpu(model.state_dict()),
        'optimizer_state_dict': _optimizer_state_to_cpu(optimizer.state_dict()),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'args': _args_dict(args),
        'best_epoch': int(best.get('epoch', -1)) if isinstance(best, dict) else -1,
        'best_loss': float(best.get('loss', float('nan'))) if isinstance(best, dict) else float('nan'),
        'best_acc': float(best.get('acc', float('nan'))) if isinstance(best, dict) else float('nan'),
        'rng_state': _rng_state_dict(),
        'extra': extra or {},
    }
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, tmp)
    tmp.replace(path)
    return path


def _write_manifest(run_dir, args, paths=None):
    run_dir = Path(run_dir)
    manifest_path = run_dir / 'manifest.json'
    manifest = {
        'format_version': 1,
        'args': _args_dict(args),
        'paths': paths or {},
        'notes': {
            'metrics_pkl': 'Same pickle format as the original training output: pickle.dump(args), pickle.dump(output).',
            'checkpoints': 'Resume-ready torch files with model, optimizer, scheduler, epoch, args and RNG state.',
            'dataset_full': 'Compressed npz. For RHM, terminal sequences and rules are saved compactly as integers.',
        },
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=_json_default)
    return manifest_path



def _add_rhm_margin_measures(entry, model, train_loader, test_loader, args):
    if getattr(args, 'compute_rhm_margins', False):
        entry.update(measures.get_rhm_margin_measures_for_splits(model, train_loader, test_loader, args))
    return entry


def _add_logit_effective_dimension_measures(entry, model, train_loader, test_loader, args):
    # Always computed at the same checkpoints as the other saved dynamics.
    entry.update(measures.get_logit_effective_dimension_measures_for_splits(model, train_loader, test_loader, args))
    return entry


def run(args):
    # Reduce batch_size when larger than train_size.
    if args.batch_size >= args.train_size:
        args.batch_size = args.train_size

    args.outname = str(Path(args.outname))
    run_dir = Path(args.outname).expanduser().resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / 'checkpoints'
    data_paths = {}
    _write_manifest(run_dir, args, paths={'metrics_pkl': args.outname})

    if args.compute_rhm_margins and 'transformer' not in args.model:
        raise ValueError('--compute_rhm_margins is currently implemented only for transformer models.')

    if args.accumulation:
        accumulation = args.train_size // args.batch_size
    else:
        accumulation = 1

    train_loader, test_loader = init.init_data(args)
    data_paths = _save_run_datasets(run_dir, train_loader, test_loader, args)
    _write_manifest(run_dir, args, paths={'metrics_pkl': args.outname, **data_paths})

    model = init.init_model(args)
    model0 = copy.deepcopy(model)

    if args.scheduler_time is None:
        args.scheduler_time = args.max_epochs

    criterion, optimizer, scheduler = init.init_training(model, args)
    dynamics, best = init.init_output(model, criterion, train_loader, test_loader, args)
    dynamics_timestep = []

    if args.save_trainstep_epochs is None:
        save_trainstep_epochs = 0
    else:
        save_trainstep_epochs = args.save_trainstep_epochs

    if save_trainstep_epochs < 0:
        raise ValueError('--save_trainstep_epochs must be >= 0 or None')

    if save_trainstep_epochs > 0:
        print(
            f"[INFO] save_trainstep_epochs={save_trainstep_epochs}: "
            f"will save measures after every optimizer update for epochs "
            f"0..{save_trainstep_epochs - 1}"
        )

    if args.compute_rhm_margins:
        print(
            '[INFO] compute_rhm_margins=True: saving train/test arrays '
            'rhm_M_mean, rhm_M_pos_frac, rhm_survival_mean, '
            'rhm_level_penalty_mean at validation checkpoints.'
        )
        print(
            f"[INFO] rhm_margins_max_train_samples={args.rhm_margins_max_train_samples}, "
            f"rhm_margins_max_test_samples={args.rhm_margins_max_test_samples}, "
            f"rhm_margins_batch_size={args.rhm_margins_batch_size}"
        )

    print(
        '[INFO] logit effective dimension=True: saving train/test logit_energy_mean, '
        'logit_input_variance, logit_effdim_entropy, logit_effdim_pr and normalized versions '
        'at validation checkpoints.'
    )
    print(
        f"[INFO] logit_effdim_max_train_samples={args.logit_effdim_max_train_samples}, "
        f"logit_effdim_max_test_samples={args.logit_effdim_max_test_samples}, "
        f"logit_effdim_batch_size={args.logit_effdim_batch_size}"
    )

    def build_timestep_entry(epoch, update, global_update):
        train_loss_step, _ = measures.test(model, train_loader)
        test_loss_step, test_acc_step = measures.test(model, test_loader)
        entry = {
            't': global_update,
            'epoch': epoch + 1,
            'epoch0': epoch,
            'update': update,
            'global_update': global_update,
            'trainloss': train_loss_step,
            'testloss': test_loss_step,
            'testacc': test_acc_step,
        }
        entry.update(measures.get_norm_measures(model))
        _add_logit_effective_dimension_measures(entry, model, train_loader, test_loader, args)
        return entry

    def make_output(epoch_done):
        return {
            'init': model0.state_dict(),
            'best': best,
            'model': copy.deepcopy(model.state_dict()),
            'dynamics': dynamics,
            'dynamics_timestep': dynamics_timestep,
            'epoch': epoch_done,
            'run_dir': str(run_dir),
            'checkpoint_dir': str(checkpoint_dir),
            'data_paths': data_paths,
            'weight_save_every': args.weight_save_every,
            'save_trainstep_epochs': args.save_trainstep_epochs,
            'compute_rhm_margins': args.compute_rhm_margins,
            'compute_logit_effdim': True,
        }

    def save_weight_checkpoint(epoch_done, reason='periodic'):
        if args.weight_save_every is None:
            return None
        ckpt_path = checkpoint_dir / f'checkpoint_epoch_{int(epoch_done):06d}.pt'
        saved = _save_resume_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=epoch_done,
            global_update=global_update,
            best=best,
            extra={
                'reason': reason,
                'metrics_pkl': args.outname,
                **data_paths,
            },
        )
        latest_path = checkpoint_dir / 'latest.pt'
        latest_txt = checkpoint_dir / 'latest_checkpoint.txt'
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(saved.name)
        except Exception:
            latest_txt.write_text(str(saved) + '\n')
        latest_txt.write_text(str(saved) + '\n')
        print(f'[INFO] saved model checkpoint to {saved}')
        return saved

    if args.print_freq >= 10:
        print_ckpts = init.init_loglinckpt(args.print_freq, args.max_epochs, fill=True)
    else:
        print_ckpts = init.init_loglinckpt(args.print_freq, args.max_epochs, fill=False)

    save_ckpts = init.init_loglinckpt(args.save_freq, args.max_epochs, fill=False)
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    start_time = time.time()
    global_update = 0

    if args.weight_save_every is not None:
        save_weight_checkpoint(0, reason='initial')

    for epoch in range(args.max_epochs):
        def post_update_callback(epoch, update, num_updates_epoch, batch_idx):
            nonlocal global_update
            global_update += 1
            if epoch < save_trainstep_epochs:
                entry = build_timestep_entry(
                    epoch=epoch,
                    update=update,
                    global_update=global_update,
                )
                dynamics_timestep.append(entry)

        loss = training.train(
            model,
            train_loader,
            accumulation,
            criterion,
            optimizer,
            scheduler,
            epoch=epoch,
            post_update_callback=post_update_callback,
        )

        if (epoch + 1) == print_ckpt:
            avg_epoch_time = (time.time() - start_time) / (epoch + 1)
            test_loss, test_acc = measures.test(model, test_loader)

            if test_loss < best['loss']:
                best['epoch'] = epoch + 1
                best['loss'] = test_loss
                best['acc'] = test_acc
                best['model'] = copy.deepcopy(model.state_dict())

            norm_measures = measures.get_norm_measures(model)
            entry = {
                't': epoch + 1,
                'trainloss': loss,
                'testloss': test_loss,
                'testacc': test_acc,
            }
            entry.update(norm_measures)

            if args.compute_margin_stats:
                entry.update(
                    measures.get_margin_stats(
                        model,
                        train_loader,
                        max_samples=args.margin_stats_max_samples,
                        batch_size=args.batch_size,
                    )
                )

            _add_rhm_margin_measures(entry, model, train_loader, test_loader, args)
            _add_logit_effective_dimension_measures(entry, model, train_loader, test_loader, args)
            dynamics.append(entry)

            log_message = (
                'Epoch : {}\t train loss: {:06.4f}, test loss: {:06.4f}, '
                'test acc.: {:04.2f}, epoch time: {:5f}'
            ).format(epoch + 1, loss, test_loss, test_acc, avg_epoch_time)

            if 'specnorm' in norm_measures:
                log_message += ', spectral complexity: {:.6e}'.format(norm_measures['specnorm'])
            if 'specnorm_no_qk' in norm_measures:
                log_message += ', spectral complexity no QK: {:.6e}'.format(norm_measures['specnorm_no_qk'])
            if 'l2norm' in norm_measures:
                log_message += ', l2 norm: {:.6e}'.format(norm_measures['l2norm'])

            if args.compute_margin_stats:
                log_message += ', margin mean: {:.6e}, margin std: {:.6e}, margin min: {:.6e}, margin max: {:.6e}'.format(
                    entry['margin_mean'],
                    entry['margin_std'],
                    entry['margin_min'],
                    entry['margin_max'],
                )

            if args.compute_rhm_margins:
                log_message += ', train M_l mean: {}'.format(
                    '[' + ', '.join('{:.3e}'.format(x) for x in entry['train_rhm_M_mean']) + ']'
                )
                log_message += ', test M_l mean: {}'.format(
                    '[' + ', '.join('{:.3e}'.format(x) for x in entry['test_rhm_M_mean']) + ']'
                )

            log_message += ', train logit Deff(ent/PR): {:.3f}/{:.3f}'.format(
                entry.get('train_logit_effdim_entropy', float('nan')),
                entry.get('train_logit_effdim_pr', float('nan')),
            )
            log_message += ', test logit Deff(ent/PR): {:.3f}/{:.3f}'.format(
                entry.get('test_logit_effdim_entropy', float('nan')),
                entry.get('test_logit_effdim_pr', float('nan')),
            )

            print(log_message)
            print_ckpt = next(print_ckpts)

        if (epoch + 1) == save_ckpt:
            print(f'Checkpoint at epoch {epoch + 1}, saving data ...')
            output = make_output(epoch + 1)
            with open(args.outname, 'wb') as handle:
                pickle.dump(args, handle)
                pickle.dump(output, handle)
            save_ckpt = next(save_ckpts)

        if args.weight_save_every is not None and (epoch + 1) % int(args.weight_save_every) == 0:
            save_weight_checkpoint(epoch + 1, reason='periodic')

        if loss <= args.loss_threshold:
            output = make_output(epoch + 1)
            with open(args.outname, 'wb') as handle:
                pickle.dump(args, handle)
                pickle.dump(output, handle)
            if args.weight_save_every is not None:
                save_weight_checkpoint(epoch + 1, reason='loss_threshold')
            break

    return None


torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(
    description='Supervised Learning of the Random Hierarchy Model with deep neural networks'
)
parser.add_argument('--device', type=str, default='cuda')

# Dataset args
parser.add_argument('--dataset', type=str)
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--num_features', metavar='v', type=int, help='number of features')
parser.add_argument('--num_classes', metavar='n', type=int, help='number of classes')
parser.add_argument(
    '--a',
    type=float,
    default=-1.0,
    help='if a<0 use the current RHM dataset; if a>=0 use power_law RHM with zipf=a on the last layer',
)
parser.add_argument('--num_synonyms', metavar='m', type=int, help='multiplicity of low-level representations')
parser.add_argument('--tuple_size', metavar='s', type=int, help='size of low-level representations')
parser.add_argument('--num_layers', metavar='L', type=int, help='number of layers')
parser.add_argument('--seed_rules', type=int, help='seed for the dataset')
parser.add_argument('--path', type=str, help='path of the text')
parser.add_argument('--num_tokens', type=int, help='number of input tokens (spatial size)')
parser.add_argument('--train_size', metavar='Ptr', type=int, help='training set size')
parser.add_argument('--batch_size', metavar='B', type=int, help='batch size')
parser.add_argument('--init_scale', type=float, default=1.0, help='multiplicative factor for random weight initialization')
parser.add_argument('--test_size', metavar='Pte', type=int, help='test set size')
parser.add_argument('--seed_sample', type=int, help='seed for the sampling of train and testset')
parser.add_argument('--input_format', type=str, default='onehot')
parser.add_argument('--whitening', type=int, default=0)

# Architecture args
parser.add_argument('--model', type=str, help='architecture (fcn, hcnn, hlcn, transformer implemented)')
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--width', type=int, help='width of the network')
parser.add_argument('--filter_size', type=int, default=None)
parser.add_argument('--num_heads', type=int, help='number of heads (transformer only)')
parser.add_argument('--embedding_dim', type=int, help='embedding dimension (transformer only)')
parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument('--seed_model', type=int, help='seed for model initialization')

# Training args
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--accumulation', default=False, action='store_true')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--scheduler_time', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument(
    '--save_trainstep_epochs',
    type=int,
    default=None,
    help='if > 0, save measures after every optimizer update during the first N epochs',
)

# Output args
parser.add_argument('--print_freq', type=int, help='frequency of prints', default=10)
parser.add_argument('--save_freq', type=int, help='frequency of saves', default=10)
parser.add_argument('--loss_threshold', type=float, default=1e-3)
parser.add_argument('--outname', type=str, required=True, help='path of the output file')
parser.add_argument(
    '--compute_margin_stats',
    default=False,
    action='store_true',
    help='compute min/max/mean/std of ordinary training margins on a deterministic subset of the training set',
)
parser.add_argument(
    '--margin_stats_max_samples',
    type=int,
    default=4096,
    help='maximum number of training examples used to compute ordinary margin statistics',
)

# RHM last-token level margin diagnostics.  Only enabled for transformers.
parser.add_argument(
    '--compute_rhm_margins',
    default=False,
    action='store_true',
    help='compute level-wise RHM last-token margins M_l for train and test splits at validation checkpoints',
)
parser.add_argument(
    '--rhm_margins_max_train_samples',
    type=int,
    default=4096,
    help='max train examples for RHM M_l diagnostics; set <=0 to use the full train split',
)
parser.add_argument(
    '--rhm_margins_max_test_samples',
    type=int,
    default=4096,
    help='max test examples for RHM M_l diagnostics; set <=0 to use the full test split',
)
parser.add_argument(
    '--rhm_margins_batch_size',
    type=int,
    default=1024,
    help='batch size used for RHM M_l diagnostics',
)

# Logit-cloud effective dimension diagnostics. These are always computed at
# validation checkpoints, with deterministic train/test subsets to keep the
# overhead bounded. Set max samples <= 0 to use the full split.
parser.add_argument(
    '--logit_effdim_max_train_samples',
    type=int,
    default=4096,
    help='max train examples for logit effective-dimension diagnostics; set <=0 to use the full train split',
)
parser.add_argument(
    '--logit_effdim_max_test_samples',
    type=int,
    default=4096,
    help='max test examples for logit effective-dimension diagnostics; set <=0 to use the full test split',
)
parser.add_argument(
    '--logit_effdim_batch_size',
    type=int,
    default=1024,
    help='batch size used for logit effective-dimension diagnostics',
)


# Run-folder data/checkpoint saving.  Use --weight_save_every none to disable
# model checkpoint files.  The .pkl metrics file is still written as before.
parser.add_argument(
    '--weight_save_every',
    type=_none_or_int,
    default=1,
    help='save a resume-ready model checkpoint every N epochs; use none/None/-1 to disable',
)
parser.add_argument(
    '--save_run_data',
    type=int,
    default=1,
    help='if 1, save the full compact train/test dataset once in run_dir/data/dataset_full.npz',
)
parser.add_argument(
    '--save_processed_dataset_inputs',
    type=int,
    default=0,
    help='if 1, also save processed model inputs; this can be very large and is off by default',
)
parser.add_argument(
    '--save_data_subset_train_size',
    type=int,
    default=1024,
    help='number of train examples saved in the reference subset; set -1 for full train split, 0 to skip train subset',
)
parser.add_argument(
    '--save_data_subset_test_size',
    type=int,
    default=1024,
    help='number of test examples saved in the reference subset; set -1 for full test split, 0 to skip test subset',
)
parser.add_argument(
    '--save_data_subset_seed',
    type=int,
    default=-1,
    help='seed used to choose the saved reference subset; default -1 reuses seed_sample',
)

args = parser.parse_args()
args.save_run_data = bool(args.save_run_data)
args.save_processed_dataset_inputs = bool(args.save_processed_dataset_inputs)
run(args)
