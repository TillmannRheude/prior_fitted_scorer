import torch
import schedulefree
import time
import math

import torch.nn as nn 
import pytorch_lightning as pl
import torch.distributed as dist
import torch.distributed.nn.functional as distnn
import torch.nn.functional as F

from torchmetrics.aggregation import MinMetric

from losses.clip import clip
from losses.gram import gram
from losses.symile import symile
from losses.triangle import triangle

LEARNED_SCORERS = {"transformer", "mlp", "barybind"}


class LightningModuleParent(pl.LightningModule):
    def __init__(
        self,
        modelname: str = "transformer",
        params_optimizer: dict = {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "eps": 1e-8,
            "betas": (0.9, 0.999),
            "warmup_steps": 1000,
        },
        params_method: dict = {
            "modelname": "transformer",
            "negative_sampling": "pair",
            "logit_scale_init": 2.65926,
            "batch_size": 128,
            "embedding_norm": True,
            "bias_init_mult": 1.0,
            "pair_num_negatives": 128,
        },
        **kwargs,
    ) -> None: 
        super().__init__()

        self.params_optimizer = params_optimizer
        self.params_method = params_method
        self.modelname = modelname
        self._scorer_lr_group_name = "transformer_scorer"
        self._last_logged_scorer_lr_multiplier = None
        if self.modelname == "barybind":
            self.automatic_optimization = False

        # Walltime logging (disabled only if explicitly turned off).
        # Logged as epoch-mean `train/step_walltime_s` every `step_walltime_every` batches.
        self.log_step_walltime = bool(self.params_method.get("log_step_walltime", True))
        self.step_walltime_every = int(self.params_method.get("step_walltime_every", 50))

        # Temperature Tau
        if params_method["logit_scale_init"] is None or params_method["logit_scale_init"] == "None":
            self.logit_scale = None
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * float(params_method["logit_scale_init"]))


        #freeze_logit_scale = True  # DEBUG toggle
        #if freeze_logit_scale:
        #    self.logit_scale.requires_grad_(False)

        self.bias = None
        self._bias_needs_global_init = False

        # Logging variables
        self.best_losses = {
            "train": float('inf'),
            "val": float('inf'),
            "test": float('inf'),
        }
        self.best_acc_top1s = {
            "train": 0.0,
            "val": 0.0,
            "test": 0.0,
        }
        self.val_loss_best = MinMetric()

        self.val_step_accuracies = []

        if modelname in LEARNED_SCORERS:
            self.loss = None
        elif modelname == "symile":
            self.loss = symile
        elif modelname == "clip":
            self.loss = clip
        elif modelname == "triangle":
            self.loss = triangle
        elif modelname == "gram":
            self.loss = gram
        else:
            raise ValueError(
                f"Unsupported modelname={modelname!r}. Expected one of "
                "['transformer', 'mlp', 'barybind', 'symile', 'clip', "
                "'triangle', 'gram']."
            )

        self.save_hyperparameters()

    def _get_encoder_stack(self):
        encoders = getattr(self.model, "encoders", None)
        if encoders is None and hasattr(self.model, "contrastive_model"):
            encoders = self.model.contrastive_model.encoders
        return encoders

    def forward(
        self, 
        x: list = [torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self.model(x)
    
    def shared_step(
        self, 
        batch: dict,
        set: str = "train",
        return_embeddings: bool = False,
    ) -> torch.tensor:
        logit_scale_exp = self.get_logit_scale_exp()
        attention_candidate_dependent = bool(self.params_method.get("candidate_dependent", True))
        model_output = self.forward(batch)
        embeddings = model_output["embeddings"]
        modality_missing_mask = None
        if self.modelname in LEARNED_SCORERS and hasattr(self, "_get_modality_missing_mask"):
            modality_missing_mask = self._get_modality_missing_mask(batch).to(
                device=embeddings[0].device,
                dtype=torch.bool,
            )

        if self.params_method["embedding_norm"]:
            # embeddings = [nn.functional.normalize(emb, dim=1) for emb in embeddings]
            embeddings = [nn.functional.normalize(emb, dim=-1) for emb in embeddings]
        
        if self.modelname == "barybind":
            objective_embeddings = embeddings
            objective_missing = modality_missing_mask
            if (
                set == "train"
                and dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            ):
                objective_embeddings = [
                    self._all_gather_with_grad(embedding)
                    for embedding in embeddings
                ]
                if modality_missing_mask is not None:
                    objective_missing = self._all_gather_without_grad(
                        modality_missing_mask
                    )

            barybind_losses = self.model.scorer.compute_losses(
                objective_embeddings,
                missing=objective_missing,
                logit_scale=logit_scale_exp,
            )
            loss = barybind_losses.loss
            self._barybind_potential_loss = barybind_losses.potential_loss
            for name, value in (
                ("mwb", barybind_losses.mwb),
                ("bvc", barybind_losses.bvc),
                ("dam", barybind_losses.dam),
                ("potential", barybind_losses.potential_loss),
            ):
                self.log(
                    f"{set}/loss_{name}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )

        elif self.modelname not in LEARNED_SCORERS:
            loss = self._analytic_objective_loss(embeddings, set=set, logit_scale_exp=logit_scale_exp)

        # DDP for global-batch learned scorer (candidate-dependent).
        elif (
            set == "train"
            and attention_candidate_dependent
            and self.params_method["negative_sampling"] == "pair"
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            r_a_local, r_b_local, r_c_local = embeddings

            r_a_all = self._all_gather_with_grad(r_a_local)
            r_b_all = self._all_gather_with_grad(r_b_local)
            r_c_all = self._all_gather_with_grad(r_c_local)
            candidate_missing_masks = (
                self._all_gather_without_grad(modality_missing_mask)
                if modality_missing_mask is not None
                else None
            )

            b_local = r_a_local.shape[0]
            rank = dist.get_rank()
            labels = torch.arange(rank * b_local, rank * b_local + b_local, device=self.device)

            loss = self._transformer_pair_loss(
                embeddings=(r_a_local, r_b_local, r_c_local),
                labels=labels,
                candidates=(r_a_all, r_b_all, r_c_all),
                modality_missing_mask=modality_missing_mask,
                candidate_missing_masks=candidate_missing_masks,
            )

        # DDP for full candidate scalar-scoring objective.
        elif (
            set == "train"
            and not attention_candidate_dependent
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            r_a_local, r_b_local, r_c_local = embeddings

            r_a_all = self._all_gather_with_grad(r_a_local)
            r_b_all = self._all_gather_with_grad(r_b_local)
            r_c_all = self._all_gather_with_grad(r_c_local)
            candidate_missing_masks = (
                self._all_gather_without_grad(modality_missing_mask)
                if modality_missing_mask is not None
                else None
            )

            b_local = r_a_local.shape[0]
            rank = dist.get_rank()
            labels = torch.arange(rank * b_local, rank * b_local + b_local, device=self.device)

            loss = self._transformer_full_loss(
                embeddings=(r_a_local, r_b_local, r_c_local),
                labels=labels,
                candidates=(r_a_all, r_b_all, r_c_all),
                modality_missing_mask=modality_missing_mask,
                candidate_missing_masks=candidate_missing_masks,
            )
        else:
            if attention_candidate_dependent:
                loss = self._transformer_pair_loss(
                    embeddings,
                    modality_missing_mask=modality_missing_mask,
                )
            else:
                loss = self._transformer_full_loss(
                    embeddings,
                    modality_missing_mask=modality_missing_mask,
                )

        self.log(f"{set}/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        if self.bias is not None:
            # log the scalar bias and the effective temperature scale
            self.log(f"{set}/bias", self.bias.detach(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        if self.logit_scale is not None:
            self.log(f"{set}/logit_scale_exp", logit_scale_exp.detach(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        if return_embeddings:
            return loss, embeddings
        return loss

    def _analytic_objective_loss(self, embeddings, set: str, logit_scale_exp):
        if (
            set == "train"
            and self.modelname == "symile"
            and self.params_method["negative_sampling"] in ["n_squared", "pair"]
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            r_a_local, r_b_local, r_c_local = embeddings

            r_a_all = self._all_gather_with_grad(r_a_local)
            r_b_all = self._all_gather_with_grad(r_b_local)
            r_c_all = self._all_gather_with_grad(r_c_local)

            b_local = r_a_local.shape[0]
            rank = dist.get_rank()
            labels = torch.arange(rank * b_local, rank * b_local + b_local, device=self.device)

            kwargs = {
                "logit_scale": logit_scale_exp,
                "negative_sampling": self.params_method["negative_sampling"],
                "bias": self.bias,
                "labels": labels,
                "candidates": (r_a_all, r_b_all, r_c_all),
            }
            if self.params_method["negative_sampling"] == "pair":
                kwargs["pair_num_negatives"] = self.params_method["pair_num_negatives"]
            return self.loss(r_a_local, r_b_local, r_c_local, **kwargs)

        if self.params_method["negative_sampling"] == "pair":
            return self.loss(
                *embeddings,
                logit_scale=logit_scale_exp,
                negative_sampling=self.params_method["negative_sampling"],
                bias=self.bias,
                pair_num_negatives=self.params_method["pair_num_negatives"],
            )
        return self.loss(
            *embeddings,
            logit_scale=logit_scale_exp,
            negative_sampling=self.params_method["negative_sampling"],
            bias=self.bias,
        )

    def get_logit_scale_exp(self):
        if self.logit_scale is None:
            return None
        return self.logit_scale.exp()

    def _ensure_seq(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, None, :] if x.ndim == 2 else x  # (B, 1, D) or (B, T, D)

    def _transformer_pair_loss(
        self,
        embeddings,
        labels=None,
        candidates=None,
        modality_missing_mask=None,
        candidate_missing_masks=None,
    ):
        """
        Pair-sampled scalar-scoring loss for TransformerScorer.
        """
        logits_list = self._transformer_pair_logits(
            embeddings=embeddings,
            labels=labels,
            candidates=candidates,
            modality_missing_mask=modality_missing_mask,
            candidate_missing_masks=candidate_missing_masks,
        )

        losses = []
        for logits in logits_list:
            scale = self.get_logit_scale_exp()
            if scale is not None:
                logits = scale * logits
            if self.bias is not None:
                logits = logits + self.bias

            y = torch.zeros((logits.shape[0],), device=logits.device, dtype=torch.long)
            losses.append(F.cross_entropy(logits, y))

        if len(losses) == 0:
            raise ValueError("No valid candidate pools for transformer pair loss.")
        return sum(losses) / len(losses)


    def _transformer_pair_logits(
        self,
        embeddings,
        labels=None,
        candidates=None,
        modality_missing_mask=None,
        candidate_missing_masks=None,
    ):
        """
        Candidate-dependent pair-sampled logits for TransformerScorer.

        Returns:
            z: list of (B, K+1) tensors where column 0 is the positive score.
        """
        r_a, r_b, r_c = embeddings
        emb_local = [self._ensure_seq(r_a), self._ensure_seq(r_b), self._ensure_seq(r_c)]

        if candidates is None:
            cand_pools = emb_local
        else:
            cand_pools = [self._ensure_seq(c) for c in candidates]

        B = r_a.shape[0]
        K = int(self.params_method["pair_num_negatives"])
        if modality_missing_mask is not None:
            modality_missing_mask = modality_missing_mask.to(
                device=r_a.device,
                dtype=torch.bool,
            )
            if modality_missing_mask.shape != (B, 3):
                raise ValueError(
                    f"Expected modality_missing_mask shape {(B, 3)}, "
                    f"got {tuple(modality_missing_mask.shape)}"
                )
        if candidate_missing_masks is None:
            candidate_missing_masks = modality_missing_mask
        elif candidate_missing_masks is not None:
            candidate_missing_masks = candidate_missing_masks.to(
                device=r_a.device,
                dtype=torch.bool,
            )

        if labels is None:
            labels = torch.arange(B, device=r_a.device)

        def _logits_for_target(t: int):
            c_t = cand_pools[t]
            N = c_t.shape[0]
            if N < 2:
                return None

            K_eff = max(1, min(K, N - 1))
            pos = labels.to(device=r_a.device, dtype=torch.long)

            if pos.min().item() < 0 or pos.max().item() >= N:
                raise ValueError(
                    f"Labels out of range for target {t}: got [{int(pos.min())},{int(pos.max())}] but N={N}"
                )

            neg = torch.randint(0, N - 1, (B, K_eff), device=r_a.device, dtype=torch.long)
            neg = neg + (neg >= pos[:, None]).long()
            idx = torch.cat([pos[:, None], neg], dim=1)
            KK = K_eff + 1

            triplets = []
            triplet_missing = []
            for m in range(3):
                x_all = cand_pools[m] if m == t else emb_local[m]
                Tm = x_all.shape[1]
                Dm = x_all.shape[2]

                if m == t:
                    x = x_all[idx.reshape(-1)]
                    if candidate_missing_masks is not None:
                        missing = candidate_missing_masks[:, m][idx.reshape(-1)]
                else:
                    x0 = x_all
                    x = x0[:, None, :, :].expand(B, KK, Tm, Dm).reshape(B * KK, Tm, Dm)
                    if modality_missing_mask is not None:
                        missing = (
                            modality_missing_mask[:, m : m + 1]
                            .expand(B, KK)
                            .reshape(B * KK)
                        )
                triplets.append(x)
                if modality_missing_mask is not None:
                    triplet_missing.append(missing)

            scorer_mask = (
                torch.stack(triplet_missing, dim=1)
                if modality_missing_mask is not None
                else None
            )
            z = self.model.transformer(
                triplets,
                src_key_padding_mask=scorer_mask,
            )
            if z.dim() == 2 and z.shape[1] == 1:
                z = z.squeeze(1)
            elif z.dim() != 1:
                raise ValueError(
                    f"TransformerScorer expected shape (B*KK,) or (B*KK,1); got {tuple(z.shape)}"
                )

            return z.view(B, KK)

        zs = [_logits_for_target(t) for t in range(3)]
        zs = [z for z in zs if z is not None]
        if len(zs) == 0:
            raise ValueError("No valid candidate pools for transformer pair logits.")

        return zs

    def _transformer_full_loss(
        self,
        embeddings,
        labels=None,
        candidates=None,
        modality_missing_mask=None,
        candidate_missing_masks=None,
    ):
        """
        Full candidate scalar-scoring loss for TransformerScorer.
        """
        r_a, r_b, r_c = embeddings
        emb_local = [self._ensure_seq(r_a), self._ensure_seq(r_b), self._ensure_seq(r_c)]

        if candidates is None:
            cand_pools = emb_local
        else:
            cand_pools = [self._ensure_seq(c) for c in candidates]

        B = r_a.shape[0]
        if modality_missing_mask is not None:
            modality_missing_mask = modality_missing_mask.to(
                device=r_a.device,
                dtype=torch.bool,
            )
            if modality_missing_mask.shape != (B, 3):
                raise ValueError(
                    f"Expected modality_missing_mask shape {(B, 3)}, "
                    f"got {tuple(modality_missing_mask.shape)}"
                )
        if candidate_missing_masks is None:
            candidate_missing_masks = modality_missing_mask
        elif candidate_missing_masks is not None:
            candidate_missing_masks = candidate_missing_masks.to(
                device=r_a.device,
                dtype=torch.bool,
            )

        if labels is None:
            labels = torch.arange(B, device=r_a.device)

        losses = []
        for t in range(3):
            c_t = cand_pools[t]
            N = c_t.shape[0]
            if N < 2:
                continue

            pos = labels.to(device=r_a.device, dtype=torch.long)
            if pos.min().item() < 0 or pos.max().item() >= N:
                raise ValueError(
                    f"Labels out of range for target {t}: got [{int(pos.min())},{int(pos.max())}] but N={N}"
                )

            triplets = []
            triplet_missing = []
            for m in range(3):
                x_all = cand_pools[m] if m == t else emb_local[m]
                Tm = x_all.shape[1]
                Dm = x_all.shape[2]

                if m == t:
                    x = x_all.unsqueeze(0).expand(B, N, Tm, Dm).reshape(B * N, Tm, Dm)
                    if candidate_missing_masks is not None:
                        missing = (
                            candidate_missing_masks[:, m]
                            .unsqueeze(0)
                            .expand(B, N)
                            .reshape(B * N)
                        )
                else:
                    x = x_all[:, None, :, :].expand(B, N, Tm, Dm).reshape(B * N, Tm, Dm)
                    if modality_missing_mask is not None:
                        missing = (
                            modality_missing_mask[:, m : m + 1]
                            .expand(B, N)
                            .reshape(B * N)
                        )
                triplets.append(x)
                if modality_missing_mask is not None:
                    triplet_missing.append(missing)

            scorer_mask = (
                torch.stack(triplet_missing, dim=1)
                if modality_missing_mask is not None
                else None
            )
            z = self.model.transformer(
                triplets,
                src_key_padding_mask=scorer_mask,
            )
            if z.dim() == 2 and z.shape[1] == 1:
                z = z.squeeze(1)
            elif z.dim() != 1:
                raise ValueError(
                    f"TransformerScorer expected shape (B*N,) or (B*N,1); got {tuple(z.shape)}"
                )

            logits = z.view(B, N)
            scale = self.get_logit_scale_exp()
            if scale is not None:
                logits = scale * logits
            if self.bias is not None:
                logits = logits + self.bias

            losses.append(F.cross_entropy(logits, pos))

        if len(losses) == 0:
            raise ValueError("No valid candidate pools for transformer full loss.")

        return sum(losses) / len(losses)

    def training_step(self, batch, batch_idx):
        if self.modelname == "barybind":
            main_optimizer, potential_optimizer = self.optimizers()
            main_optimizer.zero_grad()
            potential_optimizer.zero_grad()

            loss = self.shared_step(batch, "train")
            potential_loss = self._barybind_potential_loss

            self.manual_backward(loss)
            self.clip_gradients(
                main_optimizer,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            main_optimizer.step()

            potential_optimizer.zero_grad()
            self.manual_backward(potential_loss)
            self.clip_gradients(
                potential_optimizer,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            potential_optimizer.step()
            return loss.detach()
        if self.automatic_optimization:
            return self.shared_step(batch, "train")

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        self._update_scorer_lr_multiplier(batch_idx=batch_idx)

        # Optional walltime logging (intended for UKB gating runtime comparisons).
        if getattr(self.trainer, "sanity_checking", False):
            return

        if not bool(getattr(self, "log_step_walltime", False)):
            return

        # Default: log occasionally to limit overhead; can be overridden per module.
        every = int(getattr(self, "step_walltime_every", 50))
        if every > 1 and (int(batch_idx) % every) != 0:
            self._walltime_t0 = None
            return

        self._walltime_t0 = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        t0 = getattr(self, "_walltime_t0", None)
        if t0 is None:
            return

        # Ensure we include GPU work (forward+backward+optimizer) in the walltime.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = float(time.perf_counter() - float(t0))
        self._walltime_t0 = None

        self.log(
            "train/step_walltime_s",
            dt,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _has_scorer_lr_schedule(self) -> bool:
        scorer_lr = self.params_optimizer.get("scorer_lr", None)
        return scorer_lr not in [None, "None", "null", ""]

    def _scorer_lr_multiplier(self, batch_idx: int) -> float:
        if not self._has_scorer_lr_schedule():
            return 1.0

        start_epoch = float(self.params_optimizer.get("scorer_warmup_start_epoch", 0) or 0)
        warmup_epochs = float(self.params_optimizer.get("scorer_warmup_epochs", 0) or 0)
        curve = str(self.params_optimizer.get("scorer_warmup_curve", "cosine")).lower()

        num_batches = getattr(self.trainer, "num_training_batches", 0)
        try:
            num_batches = float(num_batches)
        except (TypeError, ValueError):
            num_batches = 0.0

        if num_batches > 0:
            epoch_position = float(self.current_epoch) + float(batch_idx) / num_batches
        else:
            epoch_position = float(self.current_epoch)

        if epoch_position < start_epoch:
            return 0.0
        if warmup_epochs <= 0:
            return 1.0

        progress = (epoch_position - start_epoch) / warmup_epochs
        progress = max(0.0, min(1.0, progress))

        if curve == "linear":
            return progress
        if curve == "constant":
            return 1.0 if progress >= 1.0 else 0.0
        if curve != "cosine":
            raise ValueError(
                f"Unknown scorer_warmup_curve={curve!r}. Expected 'cosine', 'linear', or 'constant'."
            )
        return 0.5 - 0.5 * math.cos(math.pi * progress)

    def _update_scorer_lr_multiplier(self, batch_idx: int) -> None:
        if not self._has_scorer_lr_schedule():
            return

        try:
            optimizer = self.optimizers()
        except RuntimeError:
            return

        multiplier = float(self._scorer_lr_multiplier(batch_idx=batch_idx))
        for group in optimizer.param_groups:
            if group.get("name") != self._scorer_lr_group_name:
                continue
            base_lr = group.get("base_lr", group.get("initial_lr", group["lr"]))
            group["lr"] = float(base_lr) * multiplier

        self.log(
            "train/scorer_lr_multiplier",
            torch.tensor(multiplier, device=self.device),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=False,
        )
        
    def validation_step(self, batch, batch_idx):
        loss, embeddings = self.shared_step(batch, "val", return_embeddings=True)

        if bool(self.params_method.get("validate_retrieval", True)) and hasattr(self, "retrieval_step"):
            retrieval_out = self.retrieval_step(batch, embeddings, split="val")

            ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

            if isinstance(retrieval_out, dict):
                stats = torch.tensor(
                    [
                        float(retrieval_out.get("correct_top1", 0.0)),
                        float(retrieval_out.get("correct_top5", 0.0)),
                        float(retrieval_out.get("correct_top10", 0.0)),
                        float(retrieval_out.get("correct_top100", 0.0)),
                        float(retrieval_out.get("count", 0.0)),
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                if ddp:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

                if self.trainer.is_global_zero and stats[4].item() > 0:
                    self.val_step_accuracies.append(tuple(float(x.item()) for x in stats))
            else:
                accs = retrieval_out or []

                # local stats (can be zero-length)
                correct = float(sum(accs))
                count = float(len(accs))

                stats = torch.tensor([correct, count], device=self.device, dtype=torch.float32)
                if ddp:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

                # store per-step global stats on rank 0 only
                if self.trainer.is_global_zero and count > 0:
                    self.val_step_accuracies.append((float(stats[0].item()), float(stats[1].item())))

        return loss

    def test_step(self, batch, batch_idx):
        loss, embeddings = self.shared_step(batch, "test", return_embeddings=True)

        if hasattr(self, "retrieval_step"):
            retrieval_out = self.retrieval_step(batch, embeddings, split="test")

            ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

            if isinstance(retrieval_out, dict):
                stats = torch.tensor(
                    [
                        float(retrieval_out.get("correct_top1", 0.0)),
                        float(retrieval_out.get("correct_top5", 0.0)),
                        float(retrieval_out.get("correct_top10", 0.0)),
                        float(retrieval_out.get("correct_top100", 0.0)),
                        float(retrieval_out.get("count", 0.0)),
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                if ddp:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

                if self.trainer.is_global_zero and stats[4].item() > 0:
                    self.test_step_accuracies.append(tuple(float(x.item()) for x in stats))
            else:
                accs = retrieval_out or []
                correct = float(sum(accs))
                count = float(len(accs))

                stats = torch.tensor([correct, count], device=self.device, dtype=torch.float32)
                if ddp:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

                if self.trainer.is_global_zero and count > 0:
                    self.test_step_accuracies.append((float(stats[0].item()), float(stats[1].item())))

        return loss

    def run_zeroshot_retrieval(self, set: str = "val"):
        if self.dataset_name == "symile_mimic":
            acc_dict = self.zeroshot_retrieval(f"{set}", split_nr=self.params_retrival_ds["split_nr"])
            acc_top1 = acc_dict["acc@top1"]
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported, yet.")
        
        # get best metrics over epochs 
        current_set_loss = self.trainer.callback_metrics[f"{set}/loss_epoch"].item()
        if current_set_loss < self.best_losses[set]:
            self.best_losses[set] = current_set_loss
        current_set_acc_top1 = acc_top1
        if current_set_acc_top1 > self.best_acc_top1s[set]:
            self.best_acc_top1s[set] = current_set_acc_top1

        self.log(f"{set}/min_loss", self.best_losses[set], sync_dist=True, prog_bar=True)

        return acc_dict

    def on_train_epoch_start(self):
        self.set_optimizer_mode(mode="train")

    def on_train_start(self):
        self.set_optimizer_mode(mode="train")

    def on_validation_start(self):
        self.set_optimizer_mode(mode="eval")
    
    def on_validation_epoch_start(self):
        if (
            bool(self.params_method.get("validate_retrieval", True))
            and hasattr(self, "build_candidate_bank")
        ):
            self.candidate_bank = self.build_candidate_bank(split="val")
        else:
            self.candidate_bank = None

    def on_validation_epoch_end(self):
        # During sanity check, it's normal to have no candidate bank / no valid queries
        if getattr(self.trainer, "sanity_checking", False):
            self.val_step_accuracies.clear()
            return

        current = self.trainer.callback_metrics.get("val/loss")
        self.val_loss_best.update(current)
        self.log("val/min_loss", self.val_loss_best.compute(), prog_bar=True, sync_dist=True)

        if not bool(self.params_method.get("validate_retrieval", True)):
            self.val_step_accuracies.clear()
            return

        ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        # -------------------------
        # Case 1: UKB-style retrieval (per-batch accuracies collected into a Python list)
        # -------------------------
        if hasattr(self, "retrieval_step"):
            # Only rank 0 has the list populated (see validation_step).
            if self.trainer.is_global_zero and self.val_step_accuracies:
                if len(self.val_step_accuracies[0]) == 5:
                    total_correct_top1 = sum(c1 for c1, _, _, _, _ in self.val_step_accuracies)
                    total_correct_top5 = sum(c5 for _, c5, _, _, _ in self.val_step_accuracies)
                    total_correct_top10 = sum(c10 for _, _, c10, _, _ in self.val_step_accuracies)
                    total_correct_top100 = sum(c100 for _, _, _, c100, _ in self.val_step_accuracies)
                    total_count = sum(n for _, _, _, _, n in self.val_step_accuracies)
                    acc_top1 = (total_correct_top1 / total_count) if total_count > 0 else float("nan")
                    acc_top5 = (total_correct_top5 / total_count) if total_count > 0 else float("nan")
                    acc_top10 = (total_correct_top10 / total_count) if total_count > 0 else float("nan")
                    acc_top100 = (total_correct_top100 / total_count) if total_count > 0 else float("nan")
                else:
                    total_correct = sum(c for c, n in self.val_step_accuracies)
                    total_count = sum(n for c, n in self.val_step_accuracies)
                    acc_top1 = (total_correct / total_count) if total_count > 0 else float("nan")
                    acc_top5 = float("nan")
                    acc_top10 = float("nan")
                    acc_top100 = float("nan")
            else:
                acc_top1 = float("nan")
                acc_top5 = float("nan")
                acc_top10 = float("nan")
                acc_top100 = float("nan")
            self.val_step_accuracies.clear()

            acc_t = torch.tensor([acc_top1, acc_top5, acc_top10, acc_top100], device=self.device)
            if ddp:
                dist.broadcast(acc_t, src=0)

            acc_val = float(acc_t[0].item())
            if acc_val == acc_val:  # not NaN
                if acc_val > self.best_acc_top1s["val"]:
                    self.best_acc_top1s["val"] = acc_val

            # Log on ALL ranks so EarlyStopping sees the metric everywhere
            self.log("val/acc_top1", acc_t[0], prog_bar=True, sync_dist=False, rank_zero_only=False)
            if float(acc_t[1].item()) == float(acc_t[1].item()):
                self.log("val/acc_top5", acc_t[1], prog_bar=False, sync_dist=False, rank_zero_only=False)
            if float(acc_t[2].item()) == float(acc_t[2].item()):
                self.log("val/acc_top10", acc_t[2], prog_bar=False, sync_dist=False, rank_zero_only=False)
            if float(acc_t[3].item()) == float(acc_t[3].item()):
                self.log("val/acc_top100", acc_t[3], prog_bar=False, sync_dist=False, rank_zero_only=False)
            self.log(
                "val/max_acc_top1",
                torch.tensor(self.best_acc_top1s["val"], device=self.device),
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=False,
            )
            return

        # -------------------------
        # Case 2: MIMIC-style retrieval (run_zeroshot_retrieval computes acc dict)
        # -------------------------
        if hasattr(self, "run_zeroshot_retrieval"):
            if getattr(self, "dataset_name", None) == "symile_mimic":
                acc_dict = self.run_zeroshot_retrieval("val")
                acc_top1 = float(acc_dict["acc@top1"])
            elif self.trainer.is_global_zero:
                acc_dict = self.run_zeroshot_retrieval("val")
                acc_top1 = float(acc_dict["acc@top1"])
            else:
                acc_top1 = float("nan")

            acc_t = torch.tensor(acc_top1, device=self.device)
            if ddp:
                dist.broadcast(acc_t, src=0)

            acc_val = float(acc_t.item())
            if acc_val == acc_val:  # not NaN
                if acc_val > self.best_acc_top1s["val"]:
                    self.best_acc_top1s["val"] = acc_val

            # Log on ALL ranks so EarlyStopping sees the metric everywhere
            self.log("val/acc_top1", acc_t, prog_bar=True, sync_dist=False, rank_zero_only=False)
            self.log(
                "val/max_acc_top1",
                torch.tensor(self.best_acc_top1s["val"], device=self.device),
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=False,
            )

            return

        return

    def on_test_start(self):
        self.configure_optimizers()
        self.set_optimizer_mode(mode="eval")

    def on_test_epoch_start(self):
        if hasattr(self, "build_candidate_bank"):
            self.candidate_bank = self.build_candidate_bank(split="test")
        else:
            self.candidate_bank = None

    def on_test_epoch_end(self):
        # Case 1: models with per-batch retrieval_step accumulation (e.g., UKB)
        if hasattr(self, "test_step_accuracies") and self.test_step_accuracies:
            if not self.trainer.is_global_zero:
                self.test_step_accuracies.clear()
                return

            if len(self.test_step_accuracies[0]) == 5:
                total_correct_top1 = sum(c1 for c1, _, _, _, _ in self.test_step_accuracies)
                total_correct_top5 = sum(c5 for _, c5, _, _, _ in self.test_step_accuracies)
                total_correct_top10 = sum(c10 for _, _, c10, _, _ in self.test_step_accuracies)
                total_correct_top100 = sum(c100 for _, _, _, c100, _ in self.test_step_accuracies)
                total_count = sum(n for _, _, _, _, n in self.test_step_accuracies)
                acc_top1 = (total_correct_top1 / total_count) if total_count > 0 else float("nan")
                acc_top5 = (total_correct_top5 / total_count) if total_count > 0 else float("nan")
                acc_top10 = (total_correct_top10 / total_count) if total_count > 0 else float("nan")
                acc_top100 = (total_correct_top100 / total_count) if total_count > 0 else float("nan")
            else:
                total_correct = sum(c for c, n in self.test_step_accuracies)
                total_count = sum(n for c, n in self.test_step_accuracies)
                acc_top1 = (total_correct / total_count) if total_count > 0 else float("nan")
                acc_top5 = float("nan")
                acc_top10 = float("nan")
                acc_top100 = float("nan")

            self.test_step_accuracies.clear()
            self.log("test/acc_top1", acc_top1, sync_dist=False, rank_zero_only=False, prog_bar=False)
            if acc_top5 == acc_top5:
                self.log("test/acc_top5", acc_top5, sync_dist=False, rank_zero_only=False, prog_bar=False)
            if acc_top10 == acc_top10:
                self.log("test/acc_top10", acc_top10, sync_dist=False, rank_zero_only=False, prog_bar=False)
            if acc_top100 == acc_top100:
                self.log("test/acc_top100", acc_top100, sync_dist=False, rank_zero_only=False, prog_bar=False)
            return

        # Case 2: models with run_zeroshot_retrieval (e.g., MIMIC)
        if hasattr(self, "run_zeroshot_retrieval") and not hasattr(self, "retrieval_step"):
            acc_dict = self.run_zeroshot_retrieval("test")
            acc_top1 = acc_dict["acc@top1"]
            self.log("test/acc_top1", acc_top1, sync_dist=True, prog_bar=False)
            return

    def _all_gather_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return x
        xs = distnn.all_gather(x) 
        return torch.cat(xs, dim=0)   # (B_global, d)

    def _all_gather_without_grad(self, x: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return x
        gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        return torch.cat(gathered, dim=0)

    def _apply_loss_mask(self, embeddings: list[torch.Tensor], mask: torch.Tensor) -> list[torch.Tensor]:
        # mask: (B,) bool
        mask = mask.to(device=embeddings[0].device, dtype=torch.bool)
        return [e[mask] for e in embeddings]

    def set_optimizer_mode(
            self, 
            mode: str ="train",
            just_optim: bool = False,
            return_optim: bool = False
    ) -> None: 
        optim = self.optimizers()
        optimizers = optim if isinstance(optim, (list, tuple)) else [optim]
        
        if "schedulefree" in self.params_optimizer["name"]:
            for optimizer in optimizers:
                method = getattr(optimizer, mode, None)
                if method is not None:
                    method()
        
        if return_optim:
            return optim

    def configure_optimizers(self):
        if self.modelname == "barybind":
            potential_parameters = list(self.model.scorer.potentials.parameters())
            potential_parameter_ids = {
                id(parameter) for parameter in potential_parameters
            }
            main_parameters = [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
                and id(parameter) not in potential_parameter_ids
            ]
            main_group = {
                "params": main_parameters,
                "lr": self.params_optimizer["lr"],
                "weight_decay": self.params_optimizer["weight_decay"],
            }
            optimizer_name = self.params_optimizer["name"]
            if optimizer_name == "schedulefree_adamw":
                main_optimizer = schedulefree.AdamWScheduleFree(
                    [main_group],
                    eps=self.params_optimizer["eps"],
                    warmup_steps=self.params_optimizer["warmup_steps"],
                    betas=self.params_optimizer["betas"],
                )
            elif optimizer_name == "schedulefree_sgd":
                main_optimizer = schedulefree.SGDScheduleFree(
                    [main_group],
                    lr=self.params_optimizer["lr"],
                    weight_decay=self.params_optimizer["weight_decay"],
                    momentum=self.params_optimizer["momentum"],
                    warmup_steps=self.params_optimizer["warmup_steps"],
                )
            elif optimizer_name == "sgd":
                main_optimizer = torch.optim.SGD(
                    [main_group],
                    lr=self.params_optimizer["lr"],
                    weight_decay=self.params_optimizer["weight_decay"],
                )
            elif optimizer_name == "adamw":
                main_optimizer = torch.optim.AdamW(
                    [main_group],
                    lr=self.params_optimizer["lr"],
                    weight_decay=self.params_optimizer["weight_decay"],
                    eps=self.params_optimizer["eps"],
                    betas=self.params_optimizer["betas"],
                )
            else:
                raise ValueError(f"Optimizer {optimizer_name} not supported.")

            potential_optimizer = torch.optim.AdamW(
                potential_parameters,
                lr=(
                    self.params_optimizer["lr"]
                    * float(self.params_optimizer["potential_lr_multiplier"])
                ),
                weight_decay=self.params_optimizer["weight_decay"],
                eps=self.params_optimizer["eps"],
                betas=self.params_optimizer["betas"],
            )
            return [main_optimizer, potential_optimizer]

        scorer_lr = self.params_optimizer.get("scorer_lr", None)
        if scorer_lr in [None, "None", "null", ""]:
            param_groups = [
                {
                    "params": self.parameters(),
                    "lr": self.params_optimizer["lr"],
                    "weight_decay": self.params_optimizer["weight_decay"],
                },
            ]
        else:
            model = getattr(self, "model", None)
            scorer = getattr(model, "scorer", None)
            if scorer is None:
                scorer = getattr(model, "transformer", None)
            if scorer is None:
                raise ValueError(
                    "optimizer.scorer_lr/modelname.scorer_lr was set, "
                    "but no learned scorer exists."
                )

            scorer_param_ids = {id(param) for param in scorer.parameters()}
            scorer_params = [param for param in scorer.parameters() if param.requires_grad]
            base_params = [
                param
                for param in self.parameters()
                if param.requires_grad and id(param) not in scorer_param_ids
            ]
            param_groups = [
                {
                    "params": base_params,
                    "name": "base",
                    "lr": self.params_optimizer["lr"],
                    "base_lr": self.params_optimizer["lr"],
                    "weight_decay": self.params_optimizer["weight_decay"],
                },
                {
                    "params": scorer_params,
                    "name": self._scorer_lr_group_name,
                    "lr": float(scorer_lr),
                    "base_lr": float(scorer_lr),
                    "weight_decay": self.params_optimizer["weight_decay"],
                },
            ]

        if self.params_optimizer["name"] == "schedulefree_adamw":
            optimizer = schedulefree.AdamWScheduleFree(
                param_groups, 
                eps=self.params_optimizer["eps"],
                warmup_steps=self.params_optimizer["warmup_steps"],
                betas=self.params_optimizer["betas"]
            )
        elif self.params_optimizer["name"] == "schedulefree_sgd":
            optimizer = schedulefree.SGDScheduleFree(
                param_groups, 
                lr=self.params_optimizer["lr"],
                weight_decay=self.params_optimizer["weight_decay"], 
                momentum=self.params_optimizer["momentum"],
                warmup_steps=self.params_optimizer["warmup_steps"],
            )
        elif self.params_optimizer["name"] == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.params_optimizer["lr"],
                weight_decay=self.params_optimizer["weight_decay"]
            )
        elif self.params_optimizer["name"] == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups, 
                #lr=self.params_optimizer["lr"],
                #weight_decay=self.params_optimizer["weight_decay"], 
                #eps=self.params_optimizer["eps"],
                #betas=self.params_optimizer["betas"]
            )
        else:
            raise ValueError(f"Optimizer {self.params_optimizer['name']} not supported.")
        
        return optimizer
