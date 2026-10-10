from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaryBindLosses:
    loss: torch.Tensor
    potential_loss: torch.Tensor
    mwb: torch.Tensor
    bvc: torch.Tensor
    dam: torch.Tensor


class _WBMap(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.net(x), dim=-1)


class _Potential(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _DAMHead(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        anchor: torch.Tensor,
        context: torch.Tensor,
        context_missing: torch.Tensor,
    ) -> torch.Tensor:
        context = context.masked_fill(context_missing.unsqueeze(-1), 0.0)
        attended, _ = self.attention(
            anchor[:, None],
            context,
            context,
            key_padding_mask=context_missing,
            need_weights=False,
        )
        representation = self.norm(anchor + attended[:, 0])
        return self.classifier(representation).squeeze(-1)


class BaryBind(nn.Module):
    """Symmetric trimodal adaptation of BaryBind.

    Every modality acts as the Wasserstein-barycenter initializer in turn.
    Missing modalities are placeholders only: masks remove them from the WB,
    simplex, contrastive, and matching computations.
    """

    def __init__(self, input_dim: int, params: dict, num_modalities: int = 3):
        super().__init__()
        if num_modalities != 3:
            raise ValueError("BaryBind currently supports exactly three modalities.")

        self.input_dim = int(input_dim)
        self.num_modalities = int(num_modalities)
        map_hidden_dim = int(params.get("map_hidden_dim", 2 * self.input_dim))
        potential_hidden_dim = int(
            params.get("potential_hidden_dim", self.input_dim)
        )
        dam_hidden_dim = int(params.get("dam_hidden_dim", self.input_dim))
        dam_num_heads = int(params.get("dam_num_heads", 2))
        if self.input_dim % dam_num_heads != 0:
            raise ValueError(
                "BaryBind embedding dimension must be divisible by dam_num_heads."
            )

        self.wb_maps = nn.ModuleList(
            [_WBMap(self.input_dim, map_hidden_dim) for _ in range(3)]
        )
        self.potentials = nn.ModuleList(
            [_Potential(self.input_dim, potential_hidden_dim) for _ in range(3)]
        )
        self.dam_heads = nn.ModuleList(
            [
                _DAMHead(self.input_dim, dam_num_heads, dam_hidden_dim)
                for _ in range(3)
            ]
        )

        self.alpha_mwb = float(params.get("alpha_mwb", 1.0))
        self.alpha_bvc = float(params.get("alpha_bvc", 1.0))
        self.alpha_dam = float(params.get("alpha_dam", 0.1))
        self.bvc_label_smoothing = float(
            params.get("bvc_label_smoothing", 0.1)
        )
        self.retrieval_anchor_idx = int(params.get("retrieval_anchor_idx", 0))
        if self.retrieval_anchor_idx not in range(3):
            raise ValueError("retrieval_anchor_idx must be 0, 1, or 2.")
        self.potential_constraint_weight = float(
            params.get("potential_constraint_weight", 10.0)
        )
        self.volume_eps = float(params.get("volume_eps", 0.0))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _validate_embeddings(
        self,
        embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        if len(embeddings) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(embeddings)}."
            )
        result = []
        batch_size = embeddings[0].shape[0]
        for modality, embedding in enumerate(embeddings):
            if embedding.ndim == 3:
                embedding = embedding.mean(dim=1)
            if embedding.ndim != 2:
                raise ValueError(
                    f"Modality {modality} must have shape (B,D) or (B,T,D)."
                )
            if embedding.shape != (batch_size, self.input_dim):
                raise ValueError(
                    f"Expected modality {modality} shape "
                    f"{(batch_size, self.input_dim)}, got {tuple(embedding.shape)}."
                )
            result.append(embedding)
        return result

    def _missing_mask(
        self,
        embeddings: list[torch.Tensor],
        missing: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = embeddings[0].shape[0]
        if missing is None:
            return torch.zeros(
                batch_size,
                self.num_modalities,
                device=embeddings[0].device,
                dtype=torch.bool,
            )
        if missing.shape != (batch_size, self.num_modalities):
            raise ValueError(
                f"Expected missing mask shape {(batch_size, self.num_modalities)}, "
                f"got {tuple(missing.shape)}."
            )
        return missing.to(device=embeddings[0].device, dtype=torch.bool)

    def barycenters(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            self.wb_maps[m](torch.nan_to_num(embeddings[m]))
            for m in range(3)
        ]

    def _volume(
        self,
        barycenter: torch.Tensor,
        modalities: list[torch.Tensor],
    ) -> torch.Tensor:
        vectors = [barycenter]
        vectors.extend(barycenter - modality for modality in modalities)
        spanning = torch.stack(vectors, dim=-2)
        with torch.autocast(device_type=spanning.device.type, enabled=False):
            spanning_float = spanning.float()
            gram = spanning_float @ spanning_float.transpose(-1, -2)
            determinant = torch.linalg.det(gram).abs()
        if self.volume_eps > 0.0:
            determinant = determinant.clamp_min(self.volume_eps)
        return determinant.sqrt().to(dtype=barycenter.dtype)

    def simplex_volume(
        self,
        barycenter: torch.Tensor,
        embeddings: list[torch.Tensor],
        available: torch.Tensor,
        anchor_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample volumes and a mask of valid simplexes."""
        other_indices = [m for m in range(3) if m != anchor_idx]
        other_available = available[:, other_indices]
        valid = available[:, anchor_idx] & other_available.any(dim=1)
        volumes = barycenter.new_zeros(barycenter.shape[0])

        for pattern in ((True, False), (False, True), (True, True)):
            pattern_tensor = torch.tensor(
                pattern,
                device=available.device,
                dtype=torch.bool,
            )
            selected = valid & (other_available == pattern_tensor).all(dim=1)
            if not selected.any():
                continue
            observed = [
                embeddings[other_indices[position]][selected]
                for position, is_present in enumerate(pattern)
                if is_present
            ]
            volumes[selected] = self._volume(barycenter[selected], observed)
        return volumes, valid

    def score_tuples(
        self,
        embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...],
        missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self._validate_embeddings(embeddings)
        missing = self._missing_mask(embeddings, missing)
        return self._score_validated_tuples(
            embeddings,
            missing,
            self.barycenters(embeddings),
        )

    def _score_validated_tuples(
        self,
        embeddings: list[torch.Tensor],
        missing: torch.Tensor,
        barycenters: list[torch.Tensor],
        anchor_indices: tuple[int, ...] = (0, 1, 2),
    ) -> torch.Tensor:
        available = ~missing

        score_sum = embeddings[0].new_zeros(embeddings[0].shape[0])
        score_count = embeddings[0].new_zeros(embeddings[0].shape[0])
        for anchor_idx in anchor_indices:
            barycenter = barycenters[anchor_idx]
            volume, valid = self.simplex_volume(
                barycenter,
                embeddings,
                available,
                anchor_idx,
            )
            score_sum[valid] -= volume[valid]
            score_count[valid] += 1

        return score_sum / score_count.clamp_min(1)

    def forward(
        self,
        embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...],
        src_key_padding_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        embeddings = self._validate_embeddings(embeddings)
        missing = self._missing_mask(embeddings, src_key_padding_mask)
        return self._score_validated_tuples(
            embeddings,
            missing,
            self.barycenters(embeddings),
            anchor_indices=(self.retrieval_anchor_idx,),
        )

    def _target_score_matrix(
        self,
        embeddings: list[torch.Tensor],
        missing: torch.Tensor,
        target_idx: int,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        selected_embeddings = [embedding[indices] for embedding in embeddings]
        selected_missing = missing[indices]
        selected_barycenters = self.barycenters(selected_embeddings)
        size = indices.numel()
        pair_embeddings = []
        pair_barycenters = []
        pair_missing = []
        for modality in range(3):
            values = selected_embeddings[modality]
            mapped = selected_barycenters[modality]
            masks = selected_missing[:, modality]
            if modality == target_idx:
                values = values[None].expand(size, size, -1)
                mapped = mapped[None].expand(size, size, -1)
                masks = masks[None].expand(size, size)
            else:
                values = values[:, None].expand(size, size, -1)
                mapped = mapped[:, None].expand(size, size, -1)
                masks = masks[:, None].expand(size, size)
            pair_embeddings.append(values.reshape(size * size, -1))
            pair_barycenters.append(mapped.reshape(size * size, -1))
            pair_missing.append(masks.reshape(size * size))

        pair_mask = torch.stack(pair_missing, dim=1)
        return self._score_validated_tuples(
            pair_embeddings,
            pair_mask,
            pair_barycenters,
            anchor_indices=(target_idx,),
        ).reshape(size, size)

    def bvc_loss(
        self,
        embeddings: list[torch.Tensor],
        missing: torch.Tensor,
        logit_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        available = ~missing
        losses = []
        for target_idx in range(3):
            other = [m for m in range(3) if m != target_idx]
            for pattern in ((True, False), (False, True), (True, True)):
                pattern_tensor = torch.tensor(
                    pattern,
                    device=missing.device,
                    dtype=torch.bool,
                )
                selected = (
                    available[:, target_idx]
                    & (available[:, other] == pattern_tensor).all(dim=1)
                )
                indices = selected.nonzero(as_tuple=False).flatten()
                if indices.numel() < 2:
                    continue
                logits = self._target_score_matrix(
                    embeddings,
                    missing,
                    target_idx,
                    indices,
                )
                if logit_scale is not None:
                    logits = logits * logit_scale
                labels = torch.arange(indices.numel(), device=logits.device)
                losses.append(
                    0.5
                    * (
                        F.cross_entropy(
                            logits,
                            labels,
                            label_smoothing=self.bvc_label_smoothing,
                        )
                        + F.cross_entropy(
                            logits.t(),
                            labels,
                            label_smoothing=self.bvc_label_smoothing,
                        )
                    )
                )
        if not losses:
            return sum(embedding.sum() for embedding in embeddings) * 0.0
        return torch.stack(losses).mean()

    def mwb_losses(
        self,
        embeddings: list[torch.Tensor],
        missing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = ~missing
        barycenters = self.barycenters(embeddings)
        main_terms = []
        potential_terms = []

        for anchor_idx, barycenter in enumerate(barycenters):
            anchor_present = available[:, anchor_idx]
            if not anchor_present.any():
                continue
            b = barycenter[anchor_present]
            sample_available = available[anchor_present]
            weights = sample_available.to(dtype=b.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)

            distances = []
            for modality in range(3):
                distance = (
                    0.5
                    * (
                        b
                        - torch.nan_to_num(
                            embeddings[modality][anchor_present]
                        )
                    )
                    .square()
                    .mean(dim=-1)
                )
                distances.append(
                    distance.masked_fill(~sample_available[:, modality], 0.0)
                )
            distances = torch.stack(distances, dim=1)
            potential_values = torch.stack(
                [self.potentials[m](b) for m in range(3)],
                dim=1,
            )
            weighted_potential = (weights * potential_values).sum(dim=1)
            main_terms.append(
                ((weights * distances).sum(dim=1) - weighted_potential).mean()
            )

            detached_b = b.detach()
            detached_values = torch.stack(
                [self.potentials[m](detached_b) for m in range(3)],
                dim=1,
            )
            weighted_detached = (weights * detached_values).sum(dim=1)
            potential_terms.append(
                weighted_detached.mean()
                + self.potential_constraint_weight
                * weighted_detached.square().mean()
            )

        zero = sum(embedding.sum() for embedding in embeddings) * 0.0
        main = torch.stack(main_terms).mean() if main_terms else zero
        potential = (
            torch.stack(potential_terms).mean() if potential_terms else zero
        )
        return main, potential

    def _dam_logits(
        self,
        embeddings: list[torch.Tensor],
        missing: torch.Tensor,
        anchor_indices: tuple[int, ...] = (0, 1, 2),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = ~missing
        barycenters = self.barycenters(embeddings)
        logits = embeddings[0].new_zeros(embeddings[0].shape[0])
        counts = embeddings[0].new_zeros(embeddings[0].shape[0])

        for anchor_idx in anchor_indices:
            others = [m for m in range(3) if m != anchor_idx]
            valid = available[:, anchor_idx] & available[:, others].any(dim=1)
            if not valid.any():
                continue
            context = torch.stack(
                [embeddings[m][valid] for m in others],
                dim=1,
            )
            context_missing = missing[valid][:, others]
            anchor_logits = self.dam_heads[anchor_idx](
                barycenters[anchor_idx][valid],
                context,
                context_missing,
            )
            logits[valid] += anchor_logits
            counts[valid] += 1
        return logits / counts.clamp_min(1), counts > 0

    def dam_loss(
        self,
        embeddings: list[torch.Tensor],
        missing: torch.Tensor,
        logit_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        available = ~missing
        losses = []
        for target_idx in range(3):
            other = [m for m in range(3) if m != target_idx]
            for pattern in ((True, False), (False, True), (True, True)):
                pattern_tensor = torch.tensor(
                    pattern,
                    device=missing.device,
                    dtype=torch.bool,
                )
                selected = (
                    available[:, target_idx]
                    & (available[:, other] == pattern_tensor).all(dim=1)
                )
                indices = selected.nonzero(as_tuple=False).flatten()
                if indices.numel() < 2:
                    continue

                positive_embeddings = [embedding[indices] for embedding in embeddings]
                positive_missing = missing[indices]

                with torch.no_grad():
                    sampling_scores = self._target_score_matrix(
                        embeddings,
                        missing,
                        target_idx,
                        indices,
                    )
                    if logit_scale is not None:
                        sampling_scores = sampling_scores * logit_scale
                    target_weights = F.softmax(sampling_scores, dim=1) + 1e-4
                    context_weights = F.softmax(sampling_scores.t(), dim=1) + 1e-4
                    target_weights.fill_diagonal_(0.0)
                    context_weights.fill_diagonal_(0.0)
                    negative_target_indices = torch.multinomial(
                        target_weights,
                        1,
                    ).squeeze(1)
                    negative_context_indices = torch.multinomial(
                        context_weights,
                        1,
                    ).squeeze(1)

                negative_target_embeddings = list(positive_embeddings)
                negative_target_embeddings[target_idx] = positive_embeddings[
                    target_idx
                ][negative_target_indices]
                negative_context_embeddings = [
                    positive_embeddings[modality]
                    if modality == target_idx
                    else positive_embeddings[modality][negative_context_indices]
                    for modality in range(3)
                ]

                positive_logits, positive_valid = self._dam_logits(
                    positive_embeddings,
                    positive_missing,
                    anchor_indices=(target_idx,),
                )
                negative_target_logits, negative_target_valid = self._dam_logits(
                    negative_target_embeddings,
                    positive_missing,
                    anchor_indices=(target_idx,),
                )
                negative_context_logits, negative_context_valid = self._dam_logits(
                    negative_context_embeddings,
                    positive_missing,
                    anchor_indices=(target_idx,),
                )
                valid = (
                    positive_valid
                    & negative_target_valid
                    & negative_context_valid
                )
                if not valid.any():
                    continue
                logits = torch.cat(
                    [
                        positive_logits[valid],
                        negative_target_logits[valid],
                        negative_context_logits[valid],
                    ],
                    dim=0,
                )
                labels = torch.cat(
                    [
                        torch.ones_like(positive_logits[valid]),
                        torch.zeros_like(negative_target_logits[valid]),
                        torch.zeros_like(negative_context_logits[valid]),
                    ],
                    dim=0,
                )
                losses.append(F.binary_cross_entropy_with_logits(logits, labels))

        if not losses:
            return sum(embedding.sum() for embedding in embeddings) * 0.0
        return torch.stack(losses).mean()

    def compute_losses(
        self,
        embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...],
        missing: torch.Tensor | None = None,
        logit_scale: torch.Tensor | None = None,
    ) -> BaryBindLosses:
        embeddings = self._validate_embeddings(embeddings)
        missing = self._missing_mask(embeddings, missing)
        mwb, potential_loss = self.mwb_losses(embeddings, missing)
        bvc = self.bvc_loss(embeddings, missing, logit_scale)
        dam = self.dam_loss(embeddings, missing, logit_scale)
        loss = self.alpha_mwb * mwb + self.alpha_bvc * bvc + self.alpha_dam * dam
        return BaryBindLosses(loss, potential_loss, mwb, bvc, dam)


class BaryBindModel(nn.Module):
    def __init__(self, contrastive_model: nn.Module, params: dict):
        super().__init__()
        encoders = contrastive_model.encoders
        if len(encoders) != 3:
            raise ValueError("BaryBindModel requires three modality encoders.")
        input_dim = int(getattr(encoders[0], "emb_dim"))
        if any(int(getattr(encoder, "emb_dim")) != input_dim for encoder in encoders):
            raise ValueError("All BaryBind encoder output dimensions must match.")
        self.contrastive_model = contrastive_model
        self.scorer = BaryBind(input_dim=input_dim, params=params)

    @property
    def transformer(self) -> BaryBind:
        return self.scorer

    def forward(
        self,
        x: list[torch.Tensor],
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, object]:
        return self.contrastive_model(x)
