import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
import collections
from .layers import MLPLayers, StopPredictor
from .rq import ResidualVectorQuantizer



class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 alpha = 1.0,
                 beta = 0.001,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 cf_embedding = 0,
                 phase = 1.0,
                 length_cost = 0.05,
                 length_entropy_weight = 0.01,
                 stop_hidden_dim = 128,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.phase = float(phase)
        self.length_cost = float(length_cost)
        self.length_entropy_weight = float(length_entropy_weight)
        self.stop_hidden_dim = int(stop_hidden_dim)
        if sk_epsilons is None and num_emb_list is not None:
            sk_epsilons = [0.0] * (len(num_emb_list) - 1) + [0.003] if len(num_emb_list) > 0 else []
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cf_embedding = cf_embedding
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy


        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

        if self.phase == 2.0 and num_emb_list is not None:
            self.stop_predictor = StopPredictor(
                in_dim=self.e_dim,
                max_depth=len(self.num_emb_list),
                hidden_dim=self.stop_hidden_dim,
                dropout=self.dropout_prob,
            )
        else:
            self.stop_predictor = None

    def forward(
        self,
        x,
        labels,
        lengths=None,
        residual_threshold=None,
        min_length=1,
        use_sk=True,
        return_lengths=False,
    ):
        if self.phase == 2.0 and self.stop_predictor is not None:
            x_e = self.encoder(x)
            stop_logits, stop_probs = self.stop_predictor(x_e)
            stacked_xq, all_losses, indices = self.rq(
                x_e,
                labels,
                use_sk=use_sk,
                return_prefixes=True,
            )
            # stacked_xq: [B, L, e_dim]
            B, L, D = stacked_xq.shape
            flat_xq = stacked_xq.reshape(B * L, D)
            decoded_flat = self.decoder(flat_xq)
            all_recons = decoded_flat.reshape(B, L, -1)  # [B, L, in_dim]

            # survival_probs: P(L >= l) = sum_{j=l}^L q(L=j)
            survival_probs = torch.flip(torch.cumsum(torch.flip(stop_probs, dims=[1]), dim=1), dims=[1])
            pred_lengths = torch.argmax(stop_probs, dim=-1) + 1  # 1-indexed

            if return_lengths:
                return all_recons, all_losses, indices, stacked_xq, stop_probs, survival_probs, pred_lengths
            return all_recons, all_losses, indices, stacked_xq, stop_probs, survival_probs

        x = self.encoder(x)
        rq_out = self.rq(
            x,
            labels,
            lengths=lengths,
            residual_threshold=residual_threshold,
            min_length=min_length,
            use_sk=use_sk,
            return_lengths=return_lengths,
        )
        if return_lengths:
            x_q, rq_loss, indices, dyn_lengths = rq_out
            out = self.decoder(x_q)
            return out, rq_loss, indices, x_q, dyn_lengths
        else:
            x_q, rq_loss, indices = rq_out
            out = self.decoder(x_q)
            return out, rq_loss, indices, x_q
    
    def CF_loss(self, quantized_rep, encoded_rep):
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
        similarities = torch.matmul(quantized_rep, encoded_rep.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss
    
    def vq_initialization(self,x, use_sk=True):
        self.rq.vq_ini(self.encoder(x))

    @torch.no_grad()
    def get_indices(
        self,
        xs,
        labels,
        lengths=None,
        residual_threshold=None,
        min_length=1,
        use_sk=False,
        return_lengths=False,
    ):
        x_e = self.encoder(xs)
        if self.phase == 2.0 and self.stop_predictor is not None:
            stop_logits, stop_probs = self.stop_predictor(x_e)
            pred_lengths = torch.argmax(stop_probs, dim=-1) + 1
            rq_out = self.rq(
                x_e,
                labels,
                use_sk=use_sk,
                return_prefixes=False,
            )
            indices = rq_out[2]
            if return_lengths:
                return indices, pred_lengths
            return indices

        rq_out = self.rq(
            x_e,
            labels,
            lengths=lengths,
            residual_threshold=residual_threshold,
            min_length=min_length,
            use_sk=use_sk,
            return_lengths=return_lengths,
        )
        if return_lengths:
            return rq_out[2], rq_out[3]
        return rq_out[2]

    def compute_loss(
        self,
        out,
        quant_loss,
        emb_idx,
        dense_out,
        xs=None,
        stop_probs=None,
        survival_probs=None,
    ):
        if self.phase == 2.0 and stop_probs is not None:
            return self.compute_phase2_loss(
                out,
                quant_loss,
                emb_idx,
                dense_out,
                xs=xs,
                stop_probs=stop_probs,
                survival_probs=survival_probs,
            )

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        rqvae_n_diversity_loss = loss_recon + self.quant_loss_weight * quant_loss

        # CF_Loss
        if self.alpha > 0 and self.cf_embedding is not None and not (isinstance(self.cf_embedding, int) and self.cf_embedding == 0):
            cf_embedding_in_batch = self.cf_embedding[emb_idx]
            cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(dense_out.device)
            cf_loss = self.CF_loss(dense_out, cf_embedding_in_batch)
        else:
            cf_loss = torch.tensor(0.0, device=dense_out.device)

        total_loss = rqvae_n_diversity_loss + self.alpha * cf_loss

        return total_loss, cf_loss, loss_recon, quant_loss

    def compute_phase2_loss(
        self,
        all_recons,
        quant_losses,
        emb_idx,
        stacked_xq,
        xs=None,
        stop_probs=None,
        survival_probs=None,
    ):
        # 1. Multi-depth reconstruction loss
        # all_recons: [B, L, in_dim], xs: [B, in_dim]
        if self.loss_type == 'mse':
            recon_diff = (all_recons - xs.unsqueeze(1)) ** 2
            recon_per_layer = recon_diff.mean(dim=-1)  # [B, L]
        elif self.loss_type == 'l1':
            recon_diff = torch.abs(all_recons - xs.unsqueeze(1))
            recon_per_layer = recon_diff.mean(dim=-1)  # [B, L]
        else:
            raise ValueError('incompatible loss type')

        loss_recon = (stop_probs * recon_per_layer).sum(dim=-1).mean()

        # 2. Survival-weighted quantization loss
        if isinstance(quant_losses, (list, tuple)):
            layer_losses = torch.stack(quant_losses)  # [L]
        else:
            layer_losses = quant_losses
        if survival_probs is None:
            survival_probs = torch.flip(torch.cumsum(torch.flip(stop_probs, dims=[1]), dim=1), dims=[1])
        mean_survival = survival_probs.mean(dim=0)  # [L]
        loss_quant = (mean_survival * layer_losses).sum()

        # 3. Multi-depth Collaborative Alignment Loss (LETTER)
        if self.alpha > 0 and self.cf_embedding is not None and not (isinstance(self.cf_embedding, int) and self.cf_embedding == 0):
            cf_embedding_in_batch = self.cf_embedding[emb_idx]
            cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(stacked_xq.device)
            cf_losses = []
            for l in range(stacked_xq.shape[1]):
                cf_losses.append(self.CF_loss(stacked_xq[:, l], cf_embedding_in_batch))
            stacked_cf = torch.stack(cf_losses)  # [L]
            mean_stop = stop_probs.mean(dim=0)   # [L]
            cf_loss = (mean_stop * stacked_cf).sum()
        else:
            cf_loss = torch.tensor(0.0, device=stacked_xq.device)

        # 4. Rate Penalty (Expected Length) & Entropy Regularization
        L = stop_probs.shape[1]
        depths = torch.arange(1, L + 1, device=stacked_xq.device, dtype=torch.float32)
        expected_lengths = (stop_probs * depths).sum(dim=-1)  # [B]
        mean_expected_len = expected_lengths.mean()
        entropy = -(stop_probs * torch.log(stop_probs + 1e-9)).sum(dim=-1).mean()
        rate_loss = self.length_cost * mean_expected_len - self.length_entropy_weight * entropy

        total_loss = loss_recon + self.quant_loss_weight * loss_quant + self.alpha * cf_loss + rate_loss

        return total_loss, cf_loss, loss_recon, loss_quant, mean_expected_len, entropy