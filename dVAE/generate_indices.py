#!/usr/bin/env python3
import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure paths
DVAE_DIR = str(Path(__file__).resolve().parent)
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
RQ_DIR = os.path.join(REPO_ROOT, "RQ-VAE")

for p in [DVAE_DIR, REPO_ROOT, RQ_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets import EmbDataset
from models.layers import Encoder, Decoder
from models.game import Game, VarlenGame
from utils import disambiguate_indices


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    return tot_item == tot_indice


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Semantic ID Indices with dVAE")

    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--checkpoint_path", "--checkpoint-path", type=str, required=True, dest="checkpoint_path", help="Path to dVAE checkpoint")
    parser.add_argument("--data_path", "--data-path", type=str, default=None, dest="data_path", help="Path to item embeddings (.npy or .pt). Overrides checkpoint args.")
    parser.add_argument("--output_file", "--output-file", type=str, default=None, dest="output_file", help="Path to write the resulting index JSON")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--length_mode", "--length-mode", type=str, default="argmax", choices=["argmax", "mean", "median"], dest="length_mode", help="Length selection mode for variable length")
    parser.add_argument("--disambiguate", action="store_true", default=True, help="Append terminal disambiguation token to guarantee zero collisions")
    parser.add_argument("--no_disambiguate", "--no-disambiguate", action="store_false", dest="disambiguate", help="Do not append disambiguation token")
    parser.add_argument("--batch_size", type=int, default=1024, help="Inference batch size")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu")

    print(f"[dVAE Indices] Loading checkpoint from {args.checkpoint_path}")
    try:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")

    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)

    data_path = args.data_path or ckpt_args.get("data_path")
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Item embeddings file not found: {data_path}")

    dataset = EmbDataset(data_path)
    embed_dim = ckpt.get("embed_dim", ckpt_args.get("embed_dim", dataset.dim))
    vocab_size = ckpt.get("vocab_size", ckpt_args.get("vocab_size", 256))
    hidden_size = ckpt.get("hidden_size", ckpt_args.get("hidden_size", 128))
    maxlen = ckpt.get("maxlen", ckpt_args.get("maxlen", 4))
    varlen = ckpt.get("varlen", ckpt_args.get("varlen", False))
    shared_codebooks = ckpt_args.get("shared_codebooks", True)

    print(f"[dVAE Indices] Config: vocab_size={vocab_size}, maxlen={maxlen}, varlen={varlen}, length_mode={args.length_mode}")

    # Reconstruct Encoder
    encoder = Encoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        maxlen=maxlen,
        varlen=varlen,
        shared_codebooks=shared_codebooks,
    )

    decoder = Decoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        maxlen=maxlen,
    )

    if varlen:
        graph = VarlenGame(encoder, decoder)
    else:
        graph = Game(encoder, decoder)

    graph.load_state_dict(ckpt["state_dict"], strict=False)
    graph = graph.to(device)
    graph.eval()

    inference_enc = encoder.inference(mode=args.length_mode).to(device)
    inference_enc.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_indices: List[List[str]] = []
    prefix = [f"<{chr(ord('a') + i)}_{{}}>" for i in range(26)]

    print(f"[dVAE Indices] Generating semantic IDs for {len(dataset)} items...")
    with torch.no_grad():
        for batch_x, batch_idx in tqdm(dataloader, desc="dVAE Encoding"):
            batch_x = batch_x.to(device)
            if varlen:
                codes, lengths = inference_enc(batch_x)
                codes_list = codes.cpu().tolist()
                lengths_list = lengths.cpu().tolist()
            else:
                codes = inference_enc(batch_x)
                codes_list = codes.cpu().tolist()
                lengths_list = None

            for i in range(len(codes_list)):
                item_codes = codes_list[i]
                if lengths_list is not None:
                    k_len = int(lengths_list[i])
                    item_codes = item_codes[:k_len]

                code_tokens = [prefix[lvl].format(int(tok)) for lvl, tok in enumerate(item_codes)]
                all_indices.append(code_tokens)

    all_indices_dict: Dict[str, List[str]] = {
        str(i): tokens for i, tokens in enumerate(all_indices)
    }

    tot_items = len(all_indices_dict)
    raw_unique = len(set(tuple(v) for v in all_indices_dict.values()))
    raw_collisions = tot_items - raw_unique
    print(f"[dVAE Indices] Raw unique IDs: {raw_unique}/{tot_items} (Collision rate: {raw_collisions / tot_items:.4f})")

    if args.disambiguate:
        print("[dVAE Indices] Disambiguating colliding prefixes to guarantee 0 collisions...")
        all_indices_dict = disambiguate_indices(all_indices_dict)
        final_unique = len(set(tuple(v) for v in all_indices_dict.values()))
        print(f"[dVAE Indices] Disambiguated unique IDs: {final_unique}/{tot_items}")

    # Output paths
    if args.output_file:
        output_file = args.output_file
    else:
        var_tag = f".varlen-{args.length_mode}" if varlen else ".fixed"
        output_file = os.path.join(REPO_ROOT, "data", args.dataset, f"{args.dataset}.index.dvae{var_tag}.json")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(all_indices_dict, fp)

    summary_file = os.path.splitext(output_file)[0] + ".summary.json"
    length_distribution = dict(sorted(collections.Counter(len(v) for v in all_indices_dict.values()).items()))
    summary_payload = {
        "output": str(output_file),
        "tokenizer": "dvae",
        "varlen": varlen,
        "length_mode": args.length_mode if varlen else "fixed",
        "items": tot_items,
        "min_length": min(len(v) for v in all_indices_dict.values()),
        "max_length": max(len(v) for v in all_indices_dict.values()),
        "mean_length": sum(len(v) for v in all_indices_dict.values()) / float(tot_items),
        "length_distribution": length_distribution,
        "unique_ids": len(set(tuple(v) for v in all_indices_dict.values())),
        "raw_collisions": raw_collisions,
    }
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(summary_payload, sf, indent=2)

    print(f"[dVAE Indices] Successfully saved index to {output_file}")
    print(f"[dVAE Indices] Saved index summary to {summary_file}")


if __name__ == "__main__":
    main()
