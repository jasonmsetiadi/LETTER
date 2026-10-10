import logging
import json
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os
from datasets import EmbDataset
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, args, model, target_lengths=None):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.labels = {str(i): [] for i in range(len(self.model.rq.vq_layers))}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"

        self.phase = float(getattr(args, "phase", 1.0))
        self.target_length_strategy = getattr(args, "target_length_strategy", "popularity")
        self.min_length = getattr(args, "min_length", 1)
        self.online_residual = (
            self.phase == 1.5
            and self.target_length_strategy in ("residual", "fidelity")
            and target_lengths is None
        )
        if self.online_residual:
            self.residual_threshold = getattr(args, "residual_threshold", None)
            if self.residual_threshold is None:
                self.residual_threshold = 0.2
        else:
            self.residual_threshold = None

        self.catalog_lengths = None
        self.target_lengths = None
        if target_lengths is not None:
            if isinstance(target_lengths, dict):
                max_idx = max(int(k) for k in target_lengths.keys())
                lengths_list = [
                    int(target_lengths.get(i, target_lengths.get(str(i), len(self.model.rq.vq_layers))))
                    for i in range(max_idx + 1)
                ]
                self.target_lengths = torch.tensor(lengths_list, dtype=torch.long, device=self.device)
            elif isinstance(target_lengths, (list, tuple)) or type(target_lengths).__name__ == "ndarray":
                self.target_lengths = torch.tensor(list(target_lengths), dtype=torch.long, device=self.device)
            elif hasattr(target_lengths, "to"):
                self.target_lengths = target_lengths.to(self.device).long()
            else:
                self.target_lengths = torch.tensor(list(target_lengths), dtype=torch.long, device=self.device)


        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}
        self.valid_collision_rate = {"val":[]}



    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            # optimizer = optim.AdamW([
            # {'params': self.model.parameters(), 'lr': learning_rate, 'weight_decay':weight_decay}, 
            # {'params': self.awl.parameters(), 'weight_decay':0}
            # ])
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data, n_clusters=10):
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
    
    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)
        print(len(init_loader))
        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)

            self.model.vq_initialization(data)    

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_cf_loss = 0
        total_quant_loss = 0
        print(len(train_data))
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )
        if getattr(self.model, "beta", 0) > 0:
            embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels[str(idx)] = labels

        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            lengths = None
            if self.target_lengths is not None:
                lengths = self.target_lengths[emb_idx.to(self.device)]
            out, rq_loss, indices, dense_out, dyn_lengths = self.model(
                data,
                self.labels,
                lengths=lengths,
                residual_threshold=self.residual_threshold,
                min_length=self.min_length,
                return_lengths=True,
            )

            if self.online_residual and self.catalog_lengths is not None:
                self.catalog_lengths[emb_idx.to(self.device)] = dyn_lengths

            loss, cf_loss, loss_recon, quant_loss = self.model.compute_loss(out, rq_loss, emb_idx, dense_out, xs=data)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            # iter_data.set_postfix_str("Loss: {:.4f}, RQ Loss: {:.4f}".format(loss.item(),rq_loss.item()))
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cf_loss += (cf_loss.item() if hasattr(cf_loss, 'item') else (cf_loss if cf_loss != 0 else 0))
            total_quant_loss += quant_loss.item()

        if self.online_residual and self.catalog_lengths is not None and hasattr(self.catalog_lengths, "cpu"):
            from collections import Counter
            len_dist = dict(sorted(Counter(self.catalog_lengths.cpu().tolist()).items()))
            self.logger.info(f"[Phase 1.5 Online Residual] Epoch {epoch_idx} length distribution: {len_dist}")

        return total_loss, total_recon_loss, total_cf_loss, quant_loss.item()

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        indices_set = set()

        num_sample = 0
        if getattr(self.model, "beta", 0) > 0:
            embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels[str(idx)] = labels
        for batch_idx, data in enumerate(iter_data):

            data, emb_idx = data[0], data[1]
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data, self.labels)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            emb_idx_np = emb_idx.cpu().numpy() if hasattr(emb_idx, "cpu") else np.array(emb_idx)
            for j, index in enumerate(indices):
                if self.target_lengths is not None:
                    item_id = int(emb_idx_np[j])
                    k_len = int(
                        self.target_lengths[item_id].item()
                        if hasattr(self.target_lengths[item_id], "item")
                        else self.target_lengths[item_id]
                    )
                    code = "-".join([str(int(_)) for _ in index[:k_len]])
                elif self.online_residual and self.catalog_lengths is not None:
                    item_id = int(emb_idx_np[j])
                    k_len = int(
                        self.catalog_lengths[item_id].item()
                        if hasattr(self.catalog_lengths[item_id], "item")
                        else self.catalog_lengths[item_id]
                    )
                    code = "-".join([str(int(_)) for _ in index[:k_len]])
                else:
                    code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set))/num_sample
        # balance_score = self.balance_overall(tokens_appearance)
        # wandb.log({"collision_rate": collision_rate, "balance_score": 0})


        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        target_lengths_data = None
        if self.target_lengths is not None and hasattr(self.target_lengths, "cpu"):
            target_lengths_data = self.target_lengths.cpu().tolist()
        elif self.online_residual and self.catalog_lengths is not None and hasattr(self.catalog_lengths, "cpu"):
            target_lengths_data = self.catalog_lengths.cpu().tolist()
        elif self.target_lengths is not None:
            target_lengths_data = list(self.target_lengths)

        state = {
            "args": self.args,
            "epoch": epoch,
            "phase": self.phase,
            "target_lengths": target_lengths_data,
            "residual_threshold": self.residual_threshold,
            "min_length": self.min_length,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )


    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, cf_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("cf loss", "blue") + ": %.4f" % cf_loss
        return train_loss_output + "]"

    def fit(self, data):

        cur_eval_step = 0
        if self.online_residual and self.catalog_lengths is None:
            num_items = len(data.dataset) if hasattr(data, "dataset") else len(data)
            num_layers = len(self.model.rq.vq_layers)
            if hasattr(torch, "full"):
                self.catalog_lengths = torch.full((num_items,), num_layers, dtype=torch.long, device=self.device)
        self.vq_init()
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, cf_loss, quant_loss = self._train_epoch(data, epoch_idx)

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss, cf_loss
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                # self._save_checkpoint(epoch=epoch_idx,ckpt_file=self.best_loss_ckpt)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                # if cur_eval_step >= 10:
                #     print("Finish!")
                #     break

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)

                if epoch_idx>2500:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)


        return self.best_loss, self.best_collision_rate

