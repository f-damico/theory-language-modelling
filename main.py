import os
import sys
import time
import copy

import numpy as np
import math
import random

import functools
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import pickle

import datasets
import models
import init
import training
import measures
from pathlib import Path

def build_auto_outname(args):
    data_root = Path(args.data_root).expanduser().resolve()
    run_dir = data_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"seed_rules_{args.seed_rules}"
        f"__seed_sample_{args.seed_sample}"
        f"__seed_model_{args.seed_model}.pkl"
    )

    return str(run_dir / filename)

def run( args):

    # reduce batch_size when larger than train_size
    if (args.batch_size >= args.train_size):
        args.batch_size = args.train_size
    
    assert (args.train_size%args.batch_size)==0, 'batch_size must divide train_size!'

    if args.accumulation:
        accumulation = args.train_size // args.batch_size
    else:
        accumulation = 1

    train_loader, test_loader = init.init_data( args)

    model = init.init_model( args)
    model0 = copy.deepcopy( model)

    if args.scheduler_time is None:
        args.scheduler_time = args.max_epochs
    criterion, optimizer, scheduler = init.init_training( model, args)

    dynamics, best = init.init_output( model, criterion, train_loader, test_loader, args)
    if args.print_freq >= 10:
        print_ckpts = init.init_loglinckpt( args.print_freq, args.max_epochs, fill=True)
    else:
        print_ckpts = init.init_loglinckpt( args.print_freq, args.max_epochs, fill=False)
    save_ckpts = init.init_loglinckpt( args.save_freq, args.max_epochs, fill=False)

    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    start_time = time.time()

    for epoch in range(args.max_epochs):

        loss = training.train(model, train_loader, accumulation, criterion, optimizer, scheduler)

        # evaluate / print when needed
        if (epoch + 1) == print_ckpt:

            avg_epoch_time = (time.time() - start_time) / (epoch + 1)
            test_loss, test_acc = measures.test(model, test_loader)

            if test_loss < best['loss']:
                best['epoch'] = epoch + 1
                best['loss'] = test_loss
                best['acc'] = test_acc
                best['model'] = copy.deepcopy(model.state_dict())

            dynamics.append({
                't': epoch + 1,
                'trainloss': loss,
                'testloss': test_loss,
                'testacc': test_acc
            })

            print(
                'Epoch : ', epoch + 1,
                '\t train loss: {:06.4f}'.format(loss),
                ',test loss: {:06.4f}'.format(test_loss),
                ', test acc.: {:04.2f}'.format(test_acc),
                ', epoch time: {:5f}'.format(avg_epoch_time)
            )

            print_ckpt = next(print_ckpts)

        # save independently from print checkpoints
        if (epoch + 1) == save_ckpt:

            # if this epoch was not evaluated above, evaluate now so dynamics are up to date
            if len(dynamics) == 0 or dynamics[-1]['t'] != (epoch + 1):
                test_loss, test_acc = measures.test(model, test_loader)

                if test_loss < best['loss']:
                    best['epoch'] = epoch + 1
                    best['loss'] = test_loss
                    best['acc'] = test_acc
                    best['model'] = copy.deepcopy(model.state_dict())

                dynamics.append({
                    't': epoch + 1,
                    'trainloss': loss,
                    'testloss': test_loss,
                    'testacc': test_acc
                })

            print(f'Checkpoint at epoch {epoch+1}, saving data ...')

            output = {
                'init': model0.state_dict(),
                'best': best,
                'model': copy.deepcopy(model.state_dict()),
                'dynamics': dynamics,
                'epoch': epoch + 1
            }

            with open(args.outname, "wb") as handle:
                pickle.dump(args, handle)
                pickle.dump(output, handle)

            print(f"Saved file to: {args.outname}")
            save_ckpt = next(save_ckpts)

        # early stop save
        if loss <= args.loss_threshold:

            if len(dynamics) == 0 or dynamics[-1]['t'] != (epoch + 1):
                test_loss, test_acc = measures.test(model, test_loader)

                if test_loss < best['loss']:
                    best['epoch'] = epoch + 1
                    best['loss'] = test_loss
                    best['acc'] = test_acc
                    best['model'] = copy.deepcopy(model.state_dict())

                dynamics.append({
                    't': epoch + 1,
                    'trainloss': loss,
                    'testloss': test_loss,
                    'testacc': test_acc
                })

            output = {
                'init': model0.state_dict(),
                'best': best,
                'model': copy.deepcopy(model.state_dict()),
                'dynamics': dynamics,
                'epoch': epoch + 1
            }

            with open(args.outname, "wb") as handle:
                pickle.dump(args, handle)
                pickle.dump(output, handle)

            print(f"Saved file to: {args.outname}")
            break

    os.makedirs(os.path.dirname(args.outname), exist_ok=True)
        # final save at end of training if last epoch not already stored
    if len(dynamics) == 0 or dynamics[-1]['t'] != args.max_epochs:
        test_loss, test_acc = measures.test(model, test_loader)

        if test_loss < best['loss']:
            best['epoch'] = args.max_epochs
            best['loss'] = test_loss
            best['acc'] = test_acc
            best['model'] = copy.deepcopy(model.state_dict())

        dynamics.append({
            't': args.max_epochs,
            'trainloss': loss,
            'testloss': test_loss,
            'testacc': test_acc
        })

    output = {
        'init': model0.state_dict(),
        'best': best,
        'model': copy.deepcopy(model.state_dict()),
        'dynamics': dynamics,
        'epoch': args.max_epochs
    }

    with open(args.outname, "wb") as handle:
        pickle.dump(args, handle)
        pickle.dump(output, handle)

    print(f"Final save to: {args.outname}")

    return None

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Supervised Learning of the Random Hierarchy Model with deep neural networks')
parser.add_argument("--device", type=str, default='cuda')
'''
	DATASET ARGS
'''
parser.add_argument('--dataset', type=str)
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--num_features', metavar='v', type=int, help='number of features')
parser.add_argument('--num_classes', metavar='n', type=int, help='number of classes')
parser.add_argument('--num_synonyms', metavar='m', type=int, help='multiplicity of low-level representations')
parser.add_argument('--tuple_size', metavar='s', type=int, help='size of low-level representations')
parser.add_argument('--num_layers', metavar='L', type=int, help='number of layers')
parser.add_argument("--seed_rules", type=int, help='seed for the dataset')
parser.add_argument("--path", type=str, help='path of the text')
parser.add_argument("--num_tokens", type=int, help='number of input tokens (spatial size)')
parser.add_argument('--train_size', metavar='Ptr', type=int, help='training set size')
parser.add_argument('--batch_size', metavar='B', type=int, help='batch size')
parser.add_argument('--test_size', metavar='Pte', type=int, help='test set size')
parser.add_argument("--seed_sample", type=int, help='seed for the sampling of train and testset')
parser.add_argument('--input_format', type=str, default='onehot')
parser.add_argument('--whitening', type=int, default=0)
'''
	ARCHITECTURE ARGS
'''
parser.add_argument('--model', type=str, help='architecture (fcn, hcnn, hlcn, transformer implemented)')
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--width', type=int, help='width of the network')
parser.add_argument("--filter_size", type=int, default=None)
parser.add_argument('--num_heads', type=int, help='number of heads (transformer only)')
parser.add_argument('--embedding_dim', type=int, help='embedding dimension (transformer only)') #TODO use width for this too
parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument("--seed_model", type=int, help='seed for model initialization')
'''
       TRAINING ARGS
'''
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--accumulation', default=False, action='store_true')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--scheduler_time', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=100)
'''
	OUTPUT ARGS
'''
parser.add_argument('--print_freq', type=int, help='frequency of prints', default=10)
parser.add_argument('--save_freq', type=int, help='frequency of saves', default=10)
parser.add_argument('--loss_threshold', type=float, default=1e-3)
parser.add_argument('--data_root', type=str, default='data',
                    help='root folder where run folders are created')
parser.add_argument('--run_name', type=str, default=None,
                    help='name of the subfolder inside data_root where this run is saved')
parser.add_argument('--outname', type=str, default=None,
                    help='explicit output file path; if None it is built automatically from run_name')
args = parser.parse_args()

if args.outname is None and args.run_name is None:
    parser.error("you must provide either --outname or --run_name")

if args.outname is None:
    args.outname = build_auto_outname(args)

print(f"Saving run to: {args.outname}")

run( args)
