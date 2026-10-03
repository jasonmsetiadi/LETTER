import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
import argparse
import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--dataset", type=str,default="Instruments", help='dataset')
    parser.add_argument("--root_path", type=str,default="../checkpoint/", help='root path')
    parser.add_argument('--alpha', type=str, default='1e-1', help='cf loss weight')
    parser.add_argument('--epoch', type=int, default='10000', help='epoch')
    parser.add_argument('--checkpoint', type=str, default='epoch_9999_collision_0.0012_model.pth', help='checkpoint name')
    parser.add_argument('--beta', type=str, default='1e-4', help='div loss weight')
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Direct path to an RQ-VAE checkpoint. Overrides --root_path and --checkpoint.',
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Direct path for the generated item-index JSON file.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device used to generate semantic IDs.',
    )
    parser.add_argument(
        '--phase',
        type=float,
        default=None,
        help='Phase override (1.0 for fixed, 1.5 for length-aware). If None, reads from checkpoint.',
    )
    parser.add_argument(
        '--target-lengths',
        '--target_lengths',
        type=str,
        default=None,
        dest='target_lengths',
        help='Path to target lengths JSON or index file override for Phase 1.5.',
    )
    parser.add_argument(
        '--residual-threshold',
        '--residual_threshold',
        type=float,
        default=None,
        dest='residual_threshold',
        help='Residual threshold for dynamic length halting.',
    )
    parser.add_argument(
        '--min-length',
        '--min_length',
        type=int,
        default=None,
        dest='min_length',
        help='Minimum length for dynamic length halting.',
    )

    return parser.parse_args()


args_setting = parse_args()

dataset = args_setting.dataset
ckpt_path = args_setting.checkpoint_path or (
    args_setting.root_path
    + f'alpha{args_setting.alpha}-beta{args_setting.beta}/'
    + args_setting.checkpoint
)

output_dir = f"./data/{dataset}/"
output_file = args_setting.output_file or os.path.join(
    output_dir,
    f"{dataset}.index.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json",
)
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)
device = torch.device(args_setting.device)

try:
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
except TypeError:
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]

def get_arg(name, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)

data = EmbDataset(get_arg("data_path"))

model = RQVAE(in_dim=data.dim,
                  num_emb_list=get_arg("num_emb_list"),
                  e_dim=get_arg("e_dim"),
                  layers=get_arg("layers"),
                  dropout_prob=get_arg("dropout_prob", 0.0),
                  bn=get_arg("bn", False),
                  loss_type=get_arg("loss_type", "mse"),
                  quant_loss_weight=get_arg("quant_loss_weight", 1.0),
                  kmeans_init=get_arg("kmeans_init", False),
                  kmeans_iters=get_arg("kmeans_iters", 100),
                  sk_epsilons=get_arg("sk_epsilons", None),
                  sk_iters=get_arg("sk_iters", 100),
                  )

model.load_state_dict(state_dict,strict=False)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
prefix = [f"<{chr(ord('a') + i)}_{{}}>" for i in range(26)]

def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained 
    # x = data.cpu().detach().numpy()
    # data = self.embedding.weight.cpu().detach().numpy()
    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                            n_jobs=1, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels

labels = {str(i): [] for i in range(len(model.rq.vq_layers))}
if getattr(model, "beta", 0) > 0:
    embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]
    for idx, emb in enumerate(embs):
        centers, label = constrained_km(emb)
        labels[str(idx)] = label

ckpt_phase = float(ckpt.get("phase", get_arg("phase", 1.0)))
if args_setting.phase is not None:
    phase_mode = float(args_setting.phase)
else:
    phase_mode = ckpt_phase

target_lengths = None
if phase_mode == 1.5:
    if args_setting.target_lengths:
        from truncate_indices import resolve_target_lengths
        target_lengths = resolve_target_lengths(len(data), target_lengths=args_setting.target_lengths)
    elif args_setting.residual_threshold is not None or ckpt.get("residual_threshold") is not None:
        thresh = float(
            args_setting.residual_threshold
            if args_setting.residual_threshold is not None
            else ckpt["residual_threshold"]
        )
        min_len = int(
            args_setting.min_length
            if args_setting.min_length is not None
            else ckpt.get("min_length", 1)
        )
        print(f"[Phase 1.5] Recomputing item lengths in eval() mode using saved weights (residual_threshold={thresh}, min_length={min_len})...")
        dynamic_lens = {}
        with torch.no_grad():
            for d in tqdm(data_loader, desc="[Phase 1.5] Dynamic lengths"):
                xs, e_idx = d[0].to(device), d[1]
                _, d_lens = model.get_indices(
                    xs, labels, residual_threshold=thresh, min_length=min_len, return_lengths=True
                )
                d_lens_list = d_lens.cpu().tolist() if hasattr(d_lens, "cpu") else list(d_lens)
                e_idx_list = e_idx.cpu().tolist() if hasattr(e_idx, "cpu") else list(e_idx)
                for it_id, it_len in zip(e_idx_list, d_lens_list):
                    dynamic_lens[int(it_id)] = int(it_len)
        target_lengths = dynamic_lens
    else:
        raw_lengths = ckpt.get("target_lengths", None)
        if raw_lengths is not None and isinstance(raw_lengths, (list, tuple)):
            target_lengths = {i: int(l) for i, l in enumerate(raw_lengths)}
        elif raw_lengths is not None and isinstance(raw_lengths, dict):
            target_lengths = {int(k): int(v) for k, v in raw_lengths.items()}

    if target_lengths:
        print(f"[Phase 1.5] Generating variable-length IDs with {len(target_lengths)} target lengths.")

current_item = 0
for d in tqdm(data_loader):
    d, emb_idx = d[0], d[1]
    d = d.to(device)
    
    indices = model.get_indices(d, labels, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    emb_idx_np = emb_idx.cpu().numpy() if hasattr(emb_idx, "cpu") else np.array(emb_idx)
    for j, index in enumerate(indices):
        item_id = int(emb_idx_np[j]) if j < len(emb_idx_np) else current_item
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        if target_lengths is not None:
            k_len = int(target_lengths.get(item_id, target_lengths.get(str(item_id), len(code))))
            code = code[:k_len]

        all_indices.append(code)
        all_indices_str.append(str(code))
        current_item += 1

all_indices = np.array(all_indices, dtype=object)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
while True:
    if tt >= 20 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items]
        d = d[0].to(device)
        indices = model.get_indices(d, labels, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            if target_lengths is not None:
                k_len = int(target_lengths.get(item, target_lengths.get(str(item), len(code))))
                code = code[:k_len]

            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

out_dir = os.path.dirname(output_file)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as fp:
    json.dump(all_indices_dict, fp)

summary_file = os.path.splitext(output_file)[0] + ".summary.json"
from collections import Counter
length_distribution = dict(sorted(Counter(len(v) for v in all_indices_dict.values()).items()))
summary_payload = {
    "output": str(output_file),
    "phase": phase_mode,
    "items": len(all_indices_dict),
    "min_length": min(len(v) for v in all_indices_dict.values()),
    "max_length": max(len(v) for v in all_indices_dict.values()),
    "mean_length": sum(len(v) for v in all_indices_dict.values()) / len(all_indices_dict),
    "length_distribution": length_distribution,
    "unique_ids": tot_indice,
    "collisions": tot_item - tot_indice,
}
with open(summary_file, 'w', encoding='utf-8') as sf:
    json.dump(summary_payload, sf, indent=2)
print(f"Saved {len(all_indices_dict)} IDs (phase {phase_mode}) to {output_file}")
print(f"Saved index summary to {summary_file}")

