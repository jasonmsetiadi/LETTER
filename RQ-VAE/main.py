import argparse
import random
import torch
import numpy as np
from time import time
import logging
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=2000, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str, default="../data", help="Input data path.")

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=None, help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:4", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--alpha', type=float, default=0.1, help='cf loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='diversity loss weight')
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--sample_strategy', type=str, default="all", help='sample_strategy')
    parser.add_argument('--cf_emb', type=str, default="./RQ-VAE/ckpt/Instruments-32d-sasrec.pt", help='cf emb')
   
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="../checkpoint", help="output directory for model")

    # Phase 1.5 length-aware training arguments
    parser.add_argument('--phase', type=float, default=1.0, help='Tokenizer training phase: 1.0 (fixed length) or 1.5 (length-aware item-dependent training)')
    parser.add_argument('--target_lengths', '--target-lengths', type=str, default=None, dest='target_lengths', help='Path to target lengths JSON or index file for Phase 1.5')
    parser.add_argument('--target_length_strategy', '--target-length-strategy', type=str, default='popularity', dest='target_length_strategy', choices=['popularity', 'collaborative', 'shortest_unique', 'residual'], help='Strategy to determine item lengths in Phase 1.5')
    parser.add_argument('--inter_file', '--inter-file', type=str, default=None, dest='inter_file', help='Path to interaction JSON for popularity strategy')
    parser.add_argument('--collab_signal', '--collab-signal', type=str, default='frequency', dest='collab_signal', choices=['frequency', 'user_entropy', 'pagerank', 'co_occurrence', 'cf_density'], help='Collaborative signal for popularity strategy')
    parser.add_argument('--min_length', '--min-length', type=int, default=1, dest='min_length', help='Minimum length for Phase 1.5')
    parser.add_argument('--max_length', '--max-length', type=int, default=None, dest='max_length', help='Maximum length for Phase 1.5 (defaults to number of codebook layers)')
    parser.add_argument('--fixed_index_file', '--fixed-index-file', type=str, default=None, dest='fixed_index_file', help='Fixed index JSON file if using shortest_unique or residual strategy')
    parser.add_argument('--residuals_file', '--residuals-file', type=str, default=None, dest='residuals_file', help='Residuals JSON file if using residual strategy')
    parser.add_argument('--residual_threshold', '--residual-threshold', type=float, default=None, dest='residual_threshold', help='Residual threshold for residual strategy')

    args = parser.parse_args()
    if args.sk_epsilons is None:
        args.sk_epsilons = [0.0] * (len(args.num_emb_list) - 1) + [0.003]
    elif len(args.sk_epsilons) != len(args.num_emb_list):
        raise ValueError(
            f"len(sk_epsilons) ({len(args.sk_epsilons)}) must match len(num_emb_list) ({len(args.num_emb_list)})"
        )
    return args



if __name__ == '__main__':
    """fix the random seed"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    print(args)
    cf_emb = None
    if args.alpha > 0 and args.cf_emb:
        try:
            cf_emb = torch.load(args.cf_emb, weights_only=False).squeeze().detach().numpy()
        except TypeError:
            cf_emb = torch.load(args.cf_emb).squeeze().detach().numpy()

    """build dataset"""
    data = EmbDataset(args.data_path)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  beta = args.beta,
                  alpha = args.alpha,
                  n_clusters= args.n_clusters,
                  sample_strategy =args.sample_strategy,
                  cf_embedding = cf_emb
                  )
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    target_lengths = None
    if float(args.phase) == 1.5:
        if args.target_length_strategy in ("residual", "fidelity") and args.target_lengths is None and args.residuals_file is None:
            thresh = args.residual_threshold if args.residual_threshold is not None else 0.2
            args.residual_threshold = thresh
            print(f"[Phase 1.5] Online fidelity-based residual halting active inside forward pass (threshold={thresh}, min_length={args.min_length}).")
            target_lengths = None
        else:
            from truncate_indices import resolve_target_lengths
            num_items = len(data)
            max_len = args.max_length if args.max_length is not None else len(args.num_emb_list)
            target_lengths = resolve_target_lengths(
                num_items=num_items,
                target_lengths=args.target_lengths,
                strategy=args.target_length_strategy,
                inter_source=args.inter_file,
                collab_signal=args.collab_signal,
                cf_emb_file=args.cf_emb,
                residuals=args.residuals_file,
                residual_threshold=args.residual_threshold,
                min_length=args.min_length,
                max_length=max_len,
                indices_file=args.fixed_index_file,
            )
            print(f"[Phase 1.5] Length-aware training active with {len(target_lengths)} item lengths.")
            from collections import Counter
            len_dist = dict(sorted(Counter(target_lengths.values()).items()))
            print(f"[Phase 1.5] Length distribution: {len_dist}")

    trainer = Trainer(args, model, target_lengths=target_lengths)
    best_loss, best_collision_rate = trainer.fit(data_loader)


    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)



