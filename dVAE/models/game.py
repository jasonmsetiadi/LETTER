from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import Decoder, Encoder


class Game(nn.Module):
    """Fixed-length dVAE training game."""

    def __init__(self, encoder: Encoder, decoder: Decoder, alpha: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = float(alpha)

        self.register_buffer(
            "logV",
            torch.log(torch.tensor(float(self.encoder.vocab_size), dtype=torch.float32)),
            persistent=True,
        )

    def _loss_fn(self, a: Tensor, b: Tensor) -> Tensor:
        return F.mse_loss(a, b, reduction="none").sum(dim=-1)

    def forward(
        self,
        x: Tensor,
        tau: float,
        beta: Optional[Tensor] = None,
        free_bits: Optional[Tensor] = None,
        cf_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        logits, message = self.encoder(x, tau)
        receiver_output = self.decoder(message)  # [B, T, embed_dim]

        recon_loss = self._loss_fn(x.float(), receiver_output[:, -1].float()).mean()

        q_logprobs = logits.log_softmax(dim=-1)
        q_probs = q_logprobs.exp()

        H_Z_given_X = -(q_probs * q_logprobs).sum(dim=-1)  # [B, T]
        KL_step = self.logV - H_Z_given_X  # [B, T]

        if free_bits is not None:
            KL_step_fb = F.relu(KL_step - free_bits)
        else:
            KL_step_fb = KL_step

        KL_vocab_term = KL_step_fb.sum(dim=-1).mean()

        loss = recon_loss
        if beta is not None:
            loss = loss + beta * KL_vocab_term

        # Optional collaborative alignment loss
        cf_loss = torch.tensor(0.0, device=x.device)
        if self.alpha > 0 and cf_emb is not None:
            cf_target = F.normalize(cf_emb.float(), dim=-1)
            sim = torch.matmul(receiver_output[:, -1].float(), cf_target.t()) / 0.1
            labels = torch.arange(x.size(0), device=x.device, dtype=torch.long)
            cf_loss = F.cross_entropy(sim, labels)
            loss = loss + self.alpha * cf_loss

        with torch.no_grad():
            H_Z_given_X_total = H_Z_given_X.sum(dim=-1).mean()
            q_marg = q_probs.mean(dim=0)
            H_Z = -(q_marg * (q_marg + 1e-12).log()).sum()
            I_XZ = H_Z - H_Z_given_X_total

            metrics = {
                "loss": loss.detach(),
                "recon_loss": recon_loss.detach(),
                "KL_vocab": KL_vocab_term.detach(),
                "H_Z_given_X": H_Z_given_X_total,
                "H_Z": H_Z,
                "I_XZ": I_XZ,
                "cf_loss": cf_loss.detach(),
            }

        return loss, metrics


class VarlenGame(nn.Module):
    """Variable-length dVAE training game with hazard/survival rate-distortion."""

    def __init__(self, encoder: Encoder, decoder: Decoder, alpha: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = float(alpha)

        self.register_buffer(
            "logV",
            torch.log(torch.tensor(float(self.encoder.vocab_size), dtype=torch.float32)),
            persistent=True,
        )

    def _loss_fn(self, a: Tensor, b: Tensor) -> Tensor:
        return F.mse_loss(a, b, reduction="none").sum(dim=-1)

    def forward(
        self,
        x: Tensor,
        tau: float,
        length_cost: Tensor,
        beta: Optional[Tensor] = None,
        free_bits: Optional[Tensor] = None,
        cf_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        logits, message, length_logits, survival_logits = self.encoder(x, tau)
        receiver_output = self.decoder(message)  # [B, T, embed_dim]

        T = self.encoder.maxlen
        device = x.device

        length_probs = length_logits.softmax(dim=-1)  # [B, T]
        length_logprobs = length_logits.log_softmax(dim=-1)

        # Label smoothing on length distribution
        smoothed_length_probs = 0.9 * length_probs + 0.1 * (1.0 / float(T))
        smoothed_alive = length_probs.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))

        # Multi-depth reconstruction loss
        recon_loss = 0.0
        for step in range(T):
            step_loss = self._loss_fn(x.float(), receiver_output[:, step].float())
            recon_loss = recon_loss + (smoothed_length_probs[:, step] * step_loss).mean()

        # Vocabulary KL
        q_logprobs = logits.log_softmax(dim=-1)
        q_probs = q_logprobs.exp()
        H_Z_given_X = -(q_probs * q_logprobs).sum(dim=-1)
        KL_step = self.logV - H_Z_given_X

        if free_bits is not None:
            KL_step_fb = F.relu(KL_step - free_bits)
        else:
            KL_step_fb = KL_step

        KL_weighted_step_fb = smoothed_alive * KL_step_fb
        KL_vocab_term = KL_weighted_step_fb.sum(dim=-1).mean()

        # Expected length and entropy regularization
        depths = torch.arange(1, T + 1, device=device, dtype=length_probs.dtype)[None, :]
        E_L = (length_probs * depths).sum(dim=-1).mean()
        length_entropy = -(length_probs * length_logprobs).sum(dim=-1).mean()
        KL_length_term = -length_entropy + length_cost * E_L

        loss = recon_loss
        if beta is not None:
            loss = loss + beta * (KL_vocab_term + KL_length_term)

        # Optional multi-depth collaborative alignment loss
        cf_loss = torch.tensor(0.0, device=device)
        if self.alpha > 0 and cf_emb is not None:
            cf_target = F.normalize(cf_emb.float(), dim=-1)
            cf_step_losses = []
            labels = torch.arange(x.size(0), device=device, dtype=torch.long)
            for step in range(T):
                sim = torch.matmul(receiver_output[:, step].float(), cf_target.t()) / 0.1
                cf_step_losses.append(F.cross_entropy(sim, labels))
            stacked_cf = torch.stack(cf_step_losses)  # [T]
            cf_loss = (length_probs.mean(dim=0) * stacked_cf).sum()
            loss = loss + self.alpha * cf_loss

        with torch.no_grad():
            alive = survival_logits.exp().clamp(min=0.0, max=1.0)
            H_Z_given_X_total = (alive * H_Z_given_X).sum(dim=-1).mean()
            q_marg = q_probs.mean(dim=0)
            H_Z_step = -(q_marg * (q_marg + 1e-12).log()).sum(dim=-1)
            H_Z = (alive.mean(dim=0) * H_Z_step).sum()
            I_XZ = H_Z - H_Z_given_X_total

            metrics = {
                "loss": loss.detach(),
                "recon_loss": recon_loss.detach(),
                "KL_vocab": KL_vocab_term.detach(),
                "expected_len": E_L.detach(),
                "pred_length_counts": torch.bincount(
                    length_probs.argmax(dim=-1) + 1,
                    minlength=T + 1,
                )[1:].detach(),
                "length_entropy": length_entropy.detach(),
                "KL_length": KL_length_term.detach(),
                "H_Z_given_X": H_Z_given_X_total,
                "H_Z": H_Z,
                "I_XZ": I_XZ,
                "cf_loss": cf_loss.detach(),
            }

        return loss, metrics

