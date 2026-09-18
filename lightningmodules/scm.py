import torch
import torch.nn.functional as F

from lightningmodules.utils import LEARNED_SCORERS, LightningModuleParent
from losses.retrieval import zeroshot_retrieval_logits


class SCMPretrainModel(LightningModuleParent):
    def __init__(
        self,
        model,
        params_retrival_ds: dict = None,
        **args,
    ):
        super().__init__(**args)

        self.dataset_name = "scm_pretrain"
        self.model = model

        if params_retrival_ds is None:
            params_retrival_ds = {"batch_size": 128, "split_nr": 0}
        self.params_retrival_ds = params_retrival_ds

        self.modalities = ["A", "B", "C"]
        self.candidate_idx = 0
        self.test_step_accuracies = []

        self.save_hyperparameters()

    def forward(self, x):
        if isinstance(x, dict):
            a = x["A"]
            b = x["B"]
            c = x["C"]
        else:
            a, b, c = x

        return self.model([a, b, c])

    def _encode_embeddings(self, batch):
        a = batch["A"]
        b = batch["B"]
        c = batch["C"]

        if a.ndim == 3:
            n_scms, samples_per_scm = a.shape[:2]
            flat_inputs = [
                x.reshape(n_scms * samples_per_scm, *x.shape[2:])
                for x in (a, b, c)
            ]
            contrastive_model = getattr(self.model, "contrastive_model", self.model)
            embeddings = contrastive_model(flat_inputs)["embeddings"]
            return [
                emb.reshape(n_scms, samples_per_scm, *emb.shape[1:])
                for emb in embeddings
            ]

        contrastive_model = getattr(self.model, "contrastive_model", self.model)
        return contrastive_model([a, b, c])["embeddings"]

    def shared_step(
        self,
        batch: dict,
        set: str = "train",
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        embeddings = self._encode_embeddings(batch)

        if self.params_method["embedding_norm"]:
            embeddings = [F.normalize(emb, dim=-1) for emb in embeddings]

        if embeddings[0].ndim != 3:
            loss = (
                self._transformer_pair_loss(embeddings)
                if self.modelname in LEARNED_SCORERS and self.params_method["candidate_dependent"]
                else self._transformer_full_loss(embeddings)
                if self.modelname in LEARNED_SCORERS
                else self._analytic_objective_loss(
                    embeddings,
                    set=set,
                    logit_scale_exp=self.get_logit_scale_exp(),
                )
            )
        elif self.modelname in LEARNED_SCORERS and self.params_method["candidate_dependent"]:
            loss = self._grouped_transformer_pair_loss(embeddings)
        elif self.modelname in LEARNED_SCORERS:
            loss = self._grouped_transformer_full_loss(embeddings)
        else:
            losses = [
                self._analytic_objective_loss(
                    [emb[g] for emb in embeddings],
                    set=set,
                    logit_scale_exp=self.get_logit_scale_exp(),
                )
                for g in range(embeddings[0].shape[0])
            ]
            loss = sum(losses) / len(losses)

        self.log(f"{set}/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        logit_scale_exp = self.get_logit_scale_exp()
        if self.bias is not None:
            self.log(f"{set}/bias", self.bias.detach(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        if logit_scale_exp is not None:
            self.log(
                f"{set}/logit_scale_exp",
                logit_scale_exp.detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=False,
            )

        if return_embeddings:
            return loss, embeddings
        return loss

    def _grouped_transformer_pair_loss(self, embeddings):
        r_a, r_b, r_c = embeddings
        emb = [self._ensure_group_seq(x) for x in (r_a, r_b, r_c)]
        n_scms, samples_per_scm = emb[0].shape[:2]
        n_negatives = int(self.params_method["pair_num_negatives"])

        losses = []
        for target in range(3):
            k_eff = max(1, min(n_negatives, samples_per_scm - 1))
            positives = torch.arange(samples_per_scm, device=r_a.device)
            negatives = torch.randint(
                0,
                samples_per_scm - 1,
                (n_scms, samples_per_scm, k_eff),
                device=r_a.device,
                dtype=torch.long,
            )
            negatives = negatives + (negatives >= positives[None, :, None]).long()
            candidate_idx = torch.cat(
                [
                    positives[None, :, None].expand(n_scms, samples_per_scm, 1),
                    negatives,
                ],
                dim=2,
            )
            n_candidates = k_eff + 1

            triplets = []
            for modality in range(3):
                x_all = emb[modality]
                token_count = x_all.shape[2]
                dim = x_all.shape[3]

                if modality == target:
                    scm_offsets = (
                        torch.arange(n_scms, device=r_a.device)[:, None, None]
                        * samples_per_scm
                    )
                    flat_idx = (candidate_idx + scm_offsets).reshape(-1)
                    x = x_all.reshape(n_scms * samples_per_scm, token_count, dim)[flat_idx]
                else:
                    x = (
                        x_all[:, :, None, :, :]
                        .expand(n_scms, samples_per_scm, n_candidates, token_count, dim)
                        .reshape(n_scms * samples_per_scm * n_candidates, token_count, dim)
                    )

                triplets.append(self._validate_scorer_input(x, "pair", modality))

            z = self.model.transformer(triplets)
            if z.dim() == 2 and z.shape[1] == 1:
                z = z.squeeze(1)
            elif z.dim() != 1:
                raise ValueError(
                    f"Expected transformer score shape (G*S*K,) or (G*S*K,1), got {tuple(z.shape)}"
                )

            logits = z.view(n_scms * samples_per_scm, n_candidates)
            scale = self.get_logit_scale_exp()
            if scale is not None:
                logits = scale * logits
            if self.bias is not None:
                logits = logits + self.bias

            labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
            losses.append(F.cross_entropy(logits, labels))

        return sum(losses) / len(losses)

    def _grouped_transformer_full_loss(self, embeddings):
        r_a, r_b, r_c = embeddings
        emb = [self._ensure_group_seq(x) for x in (r_a, r_b, r_c)]
        n_scms, samples_per_scm = emb[0].shape[:2]
        labels = torch.arange(samples_per_scm, device=r_a.device).repeat(n_scms)

        losses = []
        for target in range(3):
            triplets = []
            for modality in range(3):
                x_all = emb[modality]
                token_count = x_all.shape[2]
                dim = x_all.shape[3]

                if modality == target:
                    x = (
                        x_all[:, None, :, :, :]
                        .expand(n_scms, samples_per_scm, samples_per_scm, token_count, dim)
                        .reshape(n_scms * samples_per_scm * samples_per_scm, token_count, dim)
                    )
                else:
                    x = (
                        x_all[:, :, None, :, :]
                        .expand(n_scms, samples_per_scm, samples_per_scm, token_count, dim)
                        .reshape(n_scms * samples_per_scm * samples_per_scm, token_count, dim)
                    )

                triplets.append(self._validate_scorer_input(x, "full", modality))

            z = self.model.transformer(triplets)
            if z.dim() == 2 and z.shape[1] == 1:
                z = z.squeeze(1)
            elif z.dim() != 1:
                raise ValueError(
                    f"Expected transformer score shape (G*S*S,) or (G*S*S,1), got {tuple(z.shape)}"
                )

            logits = z.view(n_scms * samples_per_scm, samples_per_scm)
            scale = self.get_logit_scale_exp()
            if scale is not None:
                logits = scale * logits
            if self.bias is not None:
                logits = logits + self.bias

            losses.append(F.cross_entropy(logits, labels))

        return sum(losses) / len(losses)

    def _ensure_group_seq(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x[:, :, None, :]
        if x.ndim == 4:
            return x
        raise ValueError(f"Expected grouped embedding shape (G,S,D) or (G,S,T,D), got {tuple(x.shape)}")

    def _validate_scorer_input(self, x: torch.Tensor, context: str, modality: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected scorer input for {context}/modality {modality} to have shape "
                f"(batch, tokens, dim), got {tuple(x.shape)}"
            )
        expected_dim = getattr(self.model.transformer, "input_dim", None)
        if expected_dim is not None and int(x.shape[-1]) != int(expected_dim):
            raise ValueError(
                f"Expected scorer input dim {expected_dim} for {context}/modality {modality}, "
                f"got {x.shape[-1]}. Full shape: {tuple(x.shape)}"
            )
        if x.shape[1] < 1:
            raise ValueError(f"Scorer input has no tokens for {context}/modality {modality}: {tuple(x.shape)}")
        return x.contiguous()

    def retrieval_step(self, batch, embeddings, split: str):
        """
        In-SCM retrieval: retrieve A_i from candidates {A_j} given query
        modalities (B_i, C_i). Validation/test dataloaders provide held-out SCMs.
        """
        r_a, r_b, r_c = embeddings
        if r_a.numel() == 0:
            return []

        if r_a.ndim == 3 and self.modelname in LEARNED_SCORERS:
            n_scms, samples_per_scm, dim = r_a.shape

            a_pair = (
                r_a[:, None, :, :]
                .expand(n_scms, samples_per_scm, samples_per_scm, dim)
                .reshape(n_scms * samples_per_scm * samples_per_scm, 1, dim)
            )
            b_pair = (
                r_b[:, :, None, :]
                .expand(n_scms, samples_per_scm, samples_per_scm, dim)
                .reshape(n_scms * samples_per_scm * samples_per_scm, 1, dim)
            )
            c_pair = (
                r_c[:, :, None, :]
                .expand(n_scms, samples_per_scm, samples_per_scm, dim)
                .reshape(n_scms * samples_per_scm * samples_per_scm, 1, dim)
            )

            triplets = [
                self._validate_scorer_input(x, "retrieval", modality)
                for modality, x in enumerate([a_pair, b_pair, c_pair])
            ]

            scores = self.model.transformer(triplets)
            if scores.dim() == 2 and scores.shape[1] == 1:
                scores = scores.squeeze(1)
            elif scores.dim() != 1:
                raise ValueError(
                    f"Expected transformer score shape (G*S*S,) or (G*S*S,1), got {tuple(scores.shape)}"
                )

            logits = scores.view(n_scms, samples_per_scm, samples_per_scm)
            scale = self.get_logit_scale_exp()
            if scale is not None:
                logits = scale * logits
            if self.bias is not None:
                logits = logits + self.bias

            pred = torch.argmax(logits, dim=2)
            target = torch.arange(samples_per_scm, device=pred.device)[None, :]
            return (pred == target).float().reshape(-1).tolist()

        if r_a.ndim == 3:
            accs = []
            for g in range(r_a.shape[0]):
                logits = zeroshot_retrieval_logits(
                    r_a[g],
                    [r_b[g], r_c[g]],
                    self.get_logit_scale_exp(),
                    bias=self.bias,
                    modelname=self.modelname,
                )
                pred = torch.argmax(logits, dim=1)
                target = torch.arange(r_a.shape[1], device=pred.device, dtype=pred.dtype)
                accs.extend((pred == target).float().tolist())
            return accs

        if self.modelname in LEARNED_SCORERS:
            batch_size = int(r_a.shape[0])
            dim = int(r_a.shape[1])

            a_pair = r_a[None, :, :].expand(batch_size, batch_size, dim).reshape(batch_size * batch_size, dim)
            b_pair = r_b[:, None, :].expand(batch_size, batch_size, dim).reshape(batch_size * batch_size, dim)
            c_pair = r_c[:, None, :].expand(batch_size, batch_size, dim).reshape(batch_size * batch_size, dim)

            scores = self.model.transformer([a_pair, b_pair, c_pair])
            if scores.dim() == 2 and scores.shape[1] == 1:
                scores = scores.squeeze(1)
            elif scores.dim() != 1:
                raise ValueError(
                    f"Expected transformer score shape (B*B,) or (B*B,1), got {tuple(scores.shape)}"
                )

            logits = scores.view(batch_size, batch_size)
            scale = self.get_logit_scale_exp()
            if scale is not None:
                logits = scale * logits
            if self.bias is not None:
                logits = logits + self.bias
        else:
            logits = zeroshot_retrieval_logits(
                r_a,
                [r_b, r_c],
                self.get_logit_scale_exp(),
                bias=self.bias,
                modelname=self.modelname,
            )

        pred = torch.argmax(logits, dim=1)
        target = torch.arange(r_a.shape[0], device=pred.device, dtype=pred.dtype)
        return (pred == target).float().tolist()
