import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EmbDataset
from models.rqvae import RQVAE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute layer-wise RQ-VAE reconstruction residuals per item."
    )
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to item embeddings (e.g. data/<dataset>/<dataset>.emb.npy).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to trained RQ-VAE checkpoint .pth file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output path for residuals JSON file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda:0 if available else cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for residual computation (default: 128).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    try:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    ckpt_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    def get_arg(name, default=None):
        if isinstance(ckpt_args, dict):
            return ckpt_args.get(name, default)
        return getattr(ckpt_args, name, default)

    data_path = args.data_path or get_arg("data_path")
    data = EmbDataset(data_path)
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    model = RQVAE(
        in_dim=data.dim,
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
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    num_layers = len(model.rq.vq_layers)
    residuals_dict = {}

    with torch.no_grad():
        item_offset = 0
        for batch in tqdm(data_loader, desc="Computing residuals"):
            xs = batch[0].to(device)
            bsz = xs.size(0)
            x_e = model.encoder(xs)

            x_q = torch.zeros_like(x_e)
            layer_res_list = []
            norm_x = torch.norm(xs, p=2, dim=-1, keepdim=True).clamp(min=1e-8)

            for l in range(num_layers):
                quantizer = model.rq.vq_layers[l]
                residual = x_e - x_q

                cb = quantizer.embedding.weight
                dists = (
                    torch.sum(residual**2, dim=-1, keepdim=True)
                    + torch.sum(cb**2, dim=-1)
                    - 2 * torch.matmul(residual, cb.t())
                )
                indices = torch.argmin(dists, dim=-1)
                z_l = quantizer.embedding(indices)
                x_q = x_q + z_l

                x_rec = model.decoder(x_q)
                rel_err = (torch.norm(xs - x_rec, p=2, dim=-1, keepdim=True) / norm_x).squeeze(-1)
                layer_res_list.append(rel_err.cpu().numpy())

            batch_res = np.stack(layer_res_list, axis=-1)
            for i in range(bsz):
                residuals_dict[str(item_offset + i)] = [
                    round(float(v), 4) for v in batch_res[i]
                ]
            item_offset += bsz

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(residuals_dict, f)

    print(f"Saved layer-wise reconstruction residuals for {len(residuals_dict)} items to {output_path}")


if __name__ == "__main__":
    main()
