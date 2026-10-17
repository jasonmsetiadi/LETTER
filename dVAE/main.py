#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure dVAE is on path
DVAE_DIR = str(Path(__file__).resolve().parent)
if DVAE_DIR not in sys.path:
    sys.path.insert(0, DVAE_DIR)

from datasets import EmbDataset, array_to_tensor
from models.layers import Encoder, Decoder
from models.game import Game, VarlenGame


def get_cosine_schedule(step: int, start: float, end: float, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def parse_args():
    parser = argparse.ArgumentParser(description="dVAE Tokenizer Training (Fixed & Variable Length)")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to item embeddings (.npy or .pt)")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--cf_emb", type=str, default=None, help="Path to collaborative CF embeddings (.pt or .npy)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Collaborative filtering alignment weight")

    # Architecture arguments
    parser.add_argument("--vocab_size", type=int, default=256, help="Vocabulary size per codebook (default: 256)")
    parser.add_argument("--embed_dim", type=int, default=None, help="Input embedding dimension (auto-detected if None)")
    parser.add_argument("--hidden_size", type=int, default=128, help="Model hidden size (default: 128)")
    parser.add_argument("--maxlen", type=int, default=4, help="Maximum code sequence length (default: 4)")
    parser.add_argument("--shared_codebooks", action="store_true", default=True, help="Share codebooks across positions")
    parser.add_argument("--no_shared_codebooks", action="store_false", dest="shared_codebooks")

    # Variable-length mechanism arguments
    parser.add_argument("--varlen", action="store_true", default=False, help="Enable variable-length hazard stopping mechanism")
    parser.add_argument("--length_cost", type=float, default=5.0, help="Length rate penalty weight (gamma) for variable length")
    parser.add_argument("--length_selection_mode", type=str, default="argmax", choices=["argmax", "mean", "median"], help="Inference length mode")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num workers")
    parser.add_argument("--tau_start", type=float, default=2.0, help="Initial Gumbel temperature")
    parser.add_argument("--tau_end", type=float, default=0.5, help="Final Gumbel temperature")
    parser.add_argument("--beta_start", type=float, default=0.0, help="Initial vocab KL weight")
    parser.add_argument("--beta_end", type=float, default=2e-3, help="Final vocab KL weight")
    parser.add_argument("--free_bits", type=float, default=None, help="Free bits clamp for vocab KL")

    # Environment
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (e.g. cuda:0 or cpu)")
    parser.add_argument("--eval_step", type=int, default=100, help="Epoch interval to log and evaluate")
    parser.add_argument("--ckpt_dir", type=str, default="./dVAE/ckpt", help="Directory to save model checkpoints")
    parser.add_argument("--model_name", type=str, default="best_loss_model", help="Checkpoint filename stem")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu")
    print(f"[dVAE] Using device: {device}")

    # Load dataset
    dataset = EmbDataset(args.data_path)
    embed_dim = args.embed_dim or dataset.dim
    print(f"[dVAE] Loaded {len(dataset)} items with embedding dim {embed_dim}")

    # Load CF embeddings if requested
    cf_tensors = None
    if args.alpha > 0 and args.cf_emb and os.path.exists(args.cf_emb):
        print(f"[dVAE] Loading collaborative CF embeddings from {args.cf_emb} (alpha={args.alpha})")
        if args.cf_emb.endswith(".pt") or args.cf_emb.endswith(".pth"):
            cf_tensors = torch.load(args.cf_emb, map_location="cpu").float()
        elif args.cf_emb.endswith(".npy"):
            cf_tensors = array_to_tensor(np.load(args.cf_emb))
        if cf_tensors.ndim == 3:
            cf_tensors = cf_tensors.squeeze()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize Encoder & Decoder
    encoder = Encoder(
        vocab_size=args.vocab_size,
        embed_dim=embed_dim,
        hidden_size=args.hidden_size,
        maxlen=args.maxlen,
        varlen=args.varlen,
        shared_codebooks=args.shared_codebooks,
    )

    decoder = Decoder(
        vocab_size=args.vocab_size,
        embed_dim=embed_dim,
        hidden_size=args.hidden_size,
        maxlen=args.maxlen,
    )

    # Wrap in Game / VarlenGame
    if args.varlen:
        print(f"[dVAE] Initialized Variable-Length dVAE Game (length_cost={args.length_cost})")
        graph = VarlenGame(encoder, decoder, alpha=args.alpha).to(device)
    else:
        print("[dVAE] Initialized Fixed-Length dVAE Game")
        graph = Game(encoder, decoder, alpha=args.alpha).to(device)

    optimizer = torch.optim.AdamW(
        graph.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(dataloader)
    # Ramp beta to beta_end over the first 20% of training, then hold it there.
    schedule_steps = max(1, int(total_steps * 0.2))
    global_step = 0
    best_loss = float("inf")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    free_bits_t = torch.tensor(args.free_bits, device=device) if args.free_bits is not None else None
    length_cost_t = torch.tensor(args.length_cost, device=device) if args.varlen else None

    print(f"[dVAE] Training for {args.epochs} epochs ({total_steps} steps total)...")

    for epoch in range(1, args.epochs + 1):
        graph.train()
        epoch_losses = []
        epoch_recon = []
        epoch_kl = []
        epoch_exp_len = []
        epoch_length_entropy = []
        epoch_kl_length = []
        length_counts = torch.zeros(args.maxlen, dtype=torch.long)
        recon_depth_sum = torch.zeros(args.maxlen, dtype=torch.float32)
        recon_depth_batches = 0

        for batch_x, batch_idx in dataloader:
            batch_x = batch_x.to(device)

            # Cosine scheduling for tau and beta
            tau = get_cosine_schedule(global_step, args.tau_start, args.tau_end, schedule_steps)
            beta = get_cosine_schedule(global_step, args.beta_start, args.beta_end, schedule_steps)
            beta_t = torch.tensor(beta, device=device)

            batch_cf = None
            if cf_tensors is not None:
                batch_cf = cf_tensors[batch_idx].to(device)

            kwargs: Dict[str, Any] = {
                "x": batch_x,
                "tau": tau,
                "beta": beta_t,
                "free_bits": free_bits_t,
                "cf_emb": batch_cf,
            }
            if args.varlen:
                kwargs["length_cost"] = length_cost_t

            optimizer.zero_grad()
            loss, metrics = graph(**kwargs)
            loss.backward()
            optimizer.step()

            epoch_losses.append(metrics["loss"].item())
            epoch_recon.append(metrics["recon_loss"].item())
            epoch_kl.append(metrics["KL_vocab"].item())
            if "expected_len" in metrics:
                epoch_exp_len.append(metrics["expected_len"].item())
            if "length_entropy" in metrics:
                epoch_length_entropy.append(metrics["length_entropy"].item())
            if "KL_length" in metrics:
                epoch_kl_length.append(metrics["KL_length"].item())
            if "pred_length_counts" in metrics:
                length_counts += metrics["pred_length_counts"].cpu()
            if "recon_depth" in metrics:
                recon_depth_sum += metrics["recon_depth"].cpu()
                recon_depth_batches += 1

            global_step += 1

        avg_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        avg_recon = float(sum(epoch_recon) / max(len(epoch_recon), 1))
        avg_kl = float(sum(epoch_kl) / max(len(epoch_kl), 1))
        avg_beta = float(beta)
        avg_vocab_loss = avg_beta * avg_kl
        avg_len_str = f" | ExpLen: {sum(epoch_exp_len) / len(epoch_exp_len):.2f}" if epoch_exp_len else ""
        avg_entropy_str = (
            f" | LenEntropy: {sum(epoch_length_entropy) / len(epoch_length_entropy):.4f}"
            if epoch_length_entropy else ""
        )
        avg_kl_length = sum(epoch_kl_length) / len(epoch_kl_length) if epoch_kl_length else None
        avg_kl_length_str = (
            f" | LengthLoss: {avg_beta * avg_kl_length:.4f} | KL_length: {avg_kl_length:.4f}"
            if avg_kl_length is not None else ""
        )
        length_dist_str = (
            " | LenDist: " + ", ".join(
                f"L{i + 1}={100.0 * count / length_counts.sum().clamp_min(1):.2f}%"
                for i, count in enumerate(length_counts)
            )
            if epoch_exp_len else ""
        )
        recon_depth_str = (
            " | ReconDepth: " + ", ".join(
                f"L{i + 1}={value:.4f}"
                for i, value in enumerate(recon_depth_sum / max(recon_depth_batches, 1))
            )
            if recon_depth_batches else ""
        )

        if epoch % args.eval_step == 0 or epoch == args.epochs or epoch == 1:
            print(
                f"[dVAE] Epoch {epoch:4d}/{args.epochs} | Tau: {tau:.3f} | Beta: {avg_beta:.6f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Recon: {avg_recon:.4f} | VocabLoss: {avg_vocab_loss:.4f} | VocabKL: {avg_kl:.4f}"
                f"{avg_kl_length_str}{avg_len_str}{avg_entropy_str}"
                f"{length_dist_str}{recon_depth_str}"
            )

        # Checkpoint saving
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            ckpt_payload = {
                "epoch": epoch,
                "args": vars(args),
                "state_dict": graph.state_dict(),
                "best_loss": best_loss,
                "varlen": args.varlen,
                "mode": "varlen" if args.varlen else "fixed",
                "vocab_size": args.vocab_size,
                "maxlen": args.maxlen,
                "embed_dim": embed_dim,
                "hidden_size": args.hidden_size,
            }
            best_path = os.path.join(args.ckpt_dir, f"{args.model_name}.pth")
            torch.save(ckpt_payload, best_path)

    # Save final model
    final_payload = {
        "epoch": args.epochs,
        "args": vars(args),
        "state_dict": graph.state_dict(),
        "final_loss": avg_loss,
        "varlen": args.varlen,
        "mode": "varlen" if args.varlen else "fixed",
        "vocab_size": args.vocab_size,
        "maxlen": args.maxlen,
        "embed_dim": embed_dim,
        "hidden_size": args.hidden_size,
    }
    final_path = os.path.join(args.ckpt_dir, f"{args.model_name}_last.pth")
    torch.save(final_payload, final_path)
    print(f"[dVAE] Training complete! Best checkpoint saved to {os.path.join(args.ckpt_dir, args.model_name + '.pth')}")


if __name__ == "__main__":
    main()
