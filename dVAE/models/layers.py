import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .gpt import GPT, norm


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        maxlen: int,
        dropout: float = 0.0,
        codebook_dropout: float = 0.0,
        init_logit_scale: float = 10.0,
        init_gamma: float = 0.2,
        varlen: bool = True,
        shared_codebooks: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.maxlen = int(maxlen)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.varlen = bool(varlen)
        self.shared_codebooks = bool(shared_codebooks)

        self.dropout = nn.Dropout(dropout)
        self.codebook_dropout = nn.Dropout(codebook_dropout)

        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        num_codebooks = 1 if self.shared_codebooks else self.maxlen
        self.codebook = nn.Parameter(torch.empty(num_codebooks, self.vocab_size, self.hidden_size))
        nn.init.normal_(self.codebook, mean=0.0, std=0.02)

        self.logit_scale = nn.Parameter(
            torch.full((self.maxlen,), math.log(float(init_logit_scale)))
        )
        self.gamma = nn.Parameter(
            torch.zeros(self.maxlen - 1) + math.log(float(init_gamma))
        )

        if self.varlen:
            self.stop_backbone = nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )
            num_heads = max(1, hidden_size // 64)
            self.stop_transformer = GPT(
                num_layers=1,
                num_heads=num_heads,
                model_dim=hidden_size,
                max_seq_len=self.maxlen,
                causal=True,
                dropout=dropout,
            )
            self.stop_predictor = nn.Linear(hidden_size, 1)
            self.stop_bias = nn.Parameter(torch.zeros(self.maxlen - 1))
        else:
            self.stop_backbone = None
            self.stop_transformer = None
            self.stop_predictor = None
            self.stop_bias = None

    def _get_codebook(self, step: int) -> Tensor:
        if self.shared_codebooks:
            return self.codebook[0]
        return self.codebook[step]

    def _get_logits(self, h: Tensor, step: int) -> Tensor:
        z = self.codebook_dropout(h)
        c = self._get_codebook(step)

        logit_scale = torch.exp(self.logit_scale[step])
        z_n = F.normalize(z, dim=-1)
        c_n = F.normalize(c, dim=-1)
        return (z_n @ c_n.t()) * logit_scale

    @staticmethod
    def _process_stop_logits(stop_logits: Tensor, return_survival_logits: bool = False):
        log_cont = F.logsigmoid(-stop_logits)
        S = torch.cumsum(log_cont, dim=-1)
        length_logits = torch.cat(
            [S - log_cont + F.logsigmoid(stop_logits), S[..., -1:]], dim=-1
        )
        if return_survival_logits:
            survival_logits = torch.cat([S.new_zeros(S.shape[:-1] + (1,)), S], dim=-1)
            return length_logits, survival_logits
        return length_logits

    def forward(
        self, x: Tensor, tau: float
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        B = x.size(0)
        T = self.maxlen
        V = self.vocab_size
        device = x.device
        dtype = x.dtype

        h = self.backbone(self.dropout(x))

        if self.varlen:
            stop_hidden_states = torch.empty((B, T, self.hidden_size), device=device, dtype=dtype)
            stop_hidden_states[:, 0] = self.stop_backbone(self.dropout(x))
        else:
            stop_hidden_states = None

        logits = torch.empty((B, T, V), device=device, dtype=dtype)
        gs_probs = torch.empty((B, T, V), device=device, dtype=dtype)

        for step in range(T):
            step_logits = self._get_logits(h, step)
            step_gs_probs = F.gumbel_softmax(step_logits, tau=float(tau), hard=False, dim=-1)

            if step < T - 1:
                c_b = self._get_codebook(step)
                expected_b = step_gs_probs @ c_b
                h = h - self.gamma[step].exp() * expected_b
                h = norm(h)
                if self.varlen:
                    stop_hidden_states[:, step + 1] = h

            logits[:, step] = step_logits
            gs_probs[:, step] = step_gs_probs

        if self.varlen:
            stop_hidden_states = self.stop_transformer(stop_hidden_states)
            stop_logit = (
                self.stop_predictor(self.dropout(stop_hidden_states[:, 1:])).squeeze(dim=-1)
                + self.stop_bias[None]
            )
            length_logits, survival_logits = self._process_stop_logits(
                stop_logit.to(torch.float32), return_survival_logits=True
            )
            return logits, gs_probs, length_logits, survival_logits

        return logits, gs_probs

    def inference(self, mode: str = "argmax"):
        return InferenceEncoder(self, mode=mode)


class InferenceEncoder(nn.Module):
    def __init__(self, sender: Encoder, mode: str = "argmax"):
        super().__init__()
        self.sender = sender
        self.mode = mode

    @property
    def vocab_size(self) -> int:
        return self.sender.vocab_size

    @property
    def maxlen(self) -> int:
        return self.sender.maxlen

    @property
    def varlen(self) -> bool:
        return self.sender.varlen

    def forward(self, x: Tensor):
        B = x.size(0)
        T = self.sender.maxlen
        device = x.device

        h = self.sender.backbone(x)
        if self.sender.varlen:
            stop_hidden_states = torch.empty(
                (B, T, self.sender.hidden_size), device=device, dtype=h.dtype
            )
            stop_hidden_states[:, 0] = self.sender.stop_backbone(x)
        else:
            stop_hidden_states = None

        codes = torch.empty((B, T), device=device, dtype=torch.long)

        for step in range(T):
            step_logits = self.sender._get_logits(h, step)
            step_code = step_logits.argmax(dim=-1)
            codes[:, step] = step_code

            if step < T - 1:
                c_b = self.sender._get_codebook(step)
                codeword_b = c_b.index_select(0, step_code)
                h = h - self.sender.gamma[step].exp() * codeword_b
                h = norm(h)
                if self.sender.varlen:
                    stop_hidden_states[:, step + 1] = h

        if self.sender.varlen:
            stop_hidden_states = self.sender.stop_transformer(stop_hidden_states)
            stop_logit = (
                self.sender.stop_predictor(stop_hidden_states[:, 1:]).squeeze(dim=-1)
                + self.sender.stop_bias[None]
            )
            length_logits = self.sender._process_stop_logits(stop_logit.to(torch.float32))

            if self.mode == "argmax":
                lengths = length_logits.argmax(dim=-1) + 1  # [B] in 1..T
            elif self.mode == "mean":
                probs = length_logits.softmax(dim=-1)
                t = torch.arange(1, probs.size(-1) + 1, device=probs.device, dtype=probs.dtype)
                expected_l = (probs * t[None, :]).sum(dim=-1)
                lengths = expected_l.round().clamp(1, probs.size(-1)).to(torch.long)
            elif self.mode == "median":
                probs = length_logits.softmax(dim=-1)
                cdf = probs.cumsum(dim=-1)
                lengths = (cdf >= 0.5).to(torch.long).argmax(dim=-1) + 1
            else:
                raise ValueError(f"Unknown length selection mode: {self.mode}")

            return codes, lengths

        return codes


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        maxlen: int,
        dropout: float = 0.0,
        num_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.maxlen = int(maxlen)

        self.embedding = nn.Linear(self.vocab_size, self.hidden_size, bias=False)
        self.decoder = nn.Linear(self.hidden_size, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        num_heads = max(1, self.hidden_size // 64)
        self.transformer = GPT(
            num_layers=num_layers,
            num_heads=num_heads,
            model_dim=self.hidden_size,
            max_seq_len=self.maxlen,
            causal=True,
            dropout=dropout,
        )

    def forward(self, message: Tensor) -> Tensor:
        # message: [B, T, V] continuous relaxation
        emb = self.dropout(self.embedding(message))
        emb = self.transformer(emb)
        x = self.decoder(emb)
        return F.normalize(x, dim=-1)

    def inference(self):
        return InferenceDecoder(self)


class InferenceDecoder(nn.Module):
    def __init__(self, receiver: Decoder):
        super().__init__()
        self.receiver = receiver
        # Linear weight shape: [hidden_size, vocab_size] -> transpose for embedding lookup
        self.transposed_embedding = self.receiver.embedding.weight.T

    def forward(self, message: Tensor) -> Tensor:
        # message: [B, T] integer tokens
        emb = F.embedding(message, self.transposed_embedding)
        emb = self.receiver.transformer(emb)
        x = self.receiver.decoder(emb)
        return F.normalize(x, dim=-1)

