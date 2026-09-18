import torch
import torch.distributed as dist
import torch.nn as nn

from lightningmodules.utils import LEARNED_SCORERS, LightningModuleParent
from losses.retrieval import zeroshot_retrieval_logits


class MIMICModel(LightningModuleParent):
    def __init__(
        self,
        model,
        candidate_idx: int = 0,
        params_retrival_ds: dict = None,
        **args,
    ):
        super().__init__(**args)
        self.dataset_name = "mimic"
        self.model = model
        self.modalities = ["radiology", "timeseries", "demographics"]
        self.candidate_idx = int(candidate_idx)
        self.test_step_accuracies = []
        if self.candidate_idx < 0 or self.candidate_idx >= len(self.modalities):
            raise ValueError(f"Invalid candidate_idx={self.candidate_idx}.")

        if params_retrival_ds is None:
            params_retrival_ds = {"batch_size": 64, "split_nr": 0}
        self.params_retrival_ds = params_retrival_ds
        self.retrieval_mode = str(self.params_retrival_ds.get("retrieval_mode", "preselected")).lower()
        self.preselected_num_candidates = int(self.params_retrival_ds.get("preselected_num_candidates", 10))
        self.retrieval_seed = int(self.params_retrival_ds.get("retrieval_seed", 420))
        if self.preselected_num_candidates < 1:
            raise ValueError(f"preselected_num_candidates must be >= 1, got {self.preselected_num_candidates}.")

        self.save_hyperparameters()

    def _get_encoder_stack(self):
        encoders = getattr(self.model, "encoders", None)
        if encoders is None and hasattr(self.model, "contrastive_model"):
            encoders = self.model.contrastive_model.encoders
        if encoders is None:
            raise ValueError("Could not locate encoder stack for MIMICModel.")
        return encoders

    def _get_modalities(self, batch):
        return [
            batch["radiology"],
            batch["timeseries"],
            batch["demographics"],
        ]

    def _tensor_modality_present(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return ~torch.isnan(x)
        batch_size = x.shape[0]
        return ~torch.isnan(x).reshape(batch_size, -1).all(dim=1)

    def _modality_present(self, batch, modality_idx: int) -> torch.Tensor:
        mod_name = self.modalities[int(modality_idx)]
        if mod_name == "radiology":
            return batch["radiology"]["embedding_present"].to(device=self.device, dtype=torch.bool)
        if mod_name == "timeseries":
            if "timeseries_present" in batch:
                return batch["timeseries_present"].to(device=self.device, dtype=torch.bool)
            return self._tensor_modality_present(batch["timeseries"]).to(device=self.device)
        if mod_name == "demographics":
            if "demographics_present" in batch:
                return batch["demographics_present"].to(device=self.device, dtype=torch.bool)
            return self._tensor_modality_present(batch["demographics"]).to(device=self.device)
        raise ValueError(f"Unknown modality name {mod_name}.")

    @staticmethod
    def _index_modality(modality, mask: torch.Tensor):
        if torch.is_tensor(modality):
            return modality[mask]
        if isinstance(modality, dict):
            return {
                k: v[mask] if torch.is_tensor(v) and v.shape[0] == mask.shape[0] else v
                for k, v in modality.items()
            }
        raise TypeError(f"Unsupported modality type: {type(modality)}")

    def forward(self, batch):
        x = self._get_modalities(batch)
        return self.model(x)

    def build_candidate_bank(self, split):
        r_list, hadm_list = [], []
        dl = self.trainer.datamodule.val_dataloader() if split == "val" else self.trainer.datamodule.test_dataloader()
        encoder_stack = self._get_encoder_stack()

        for batch in dl:
            batch = self.trainer.strategy.batch_to_device(batch, self.device)
            target_present = self._modality_present(batch, self.candidate_idx)
            if target_present.sum().item() == 0:
                continue

            target_name = self.modalities[self.candidate_idx]
            target_batch = self._index_modality(batch[target_name], target_present)
            reps = encoder_stack[self.candidate_idx](target_batch)
            if self.params_method.get("embedding_norm", False):
                reps = nn.functional.normalize(reps, dim=1)

            r_list.append(reps)
            hadm_list.append(batch["hadm_id"][target_present].to(self.device))

        if len(r_list) == 0:
            return None

        r_local = torch.cat(r_list, dim=0)
        hadm_local = torch.cat(hadm_list, dim=0)
        ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if not ddp:
            return {"r": r_local, "hadm_id": hadm_local}

        emb_dim = r_local.shape[1]
        len_t = torch.tensor([r_local.shape[0]], device=self.device, dtype=torch.long)
        lens = [torch.empty_like(len_t) for _ in range(dist.get_world_size())]
        dist.all_gather(lens, len_t)
        lens = [int(x.item()) for x in lens]
        max_len = max(lens)

        if r_local.shape[0] < max_len:
            pad = max_len - r_local.shape[0]
            r_local = torch.cat([r_local, torch.zeros((pad, emb_dim), device=self.device, dtype=r_local.dtype)], dim=0)
            hadm_local = torch.cat([hadm_local, torch.full((pad,), -1, device=self.device, dtype=hadm_local.dtype)], dim=0)

        r_gather = [torch.empty((max_len, emb_dim), device=self.device, dtype=r_local.dtype) for _ in range(dist.get_world_size())]
        id_gather = [torch.empty((max_len,), device=self.device, dtype=hadm_local.dtype) for _ in range(dist.get_world_size())]
        dist.all_gather(r_gather, r_local)
        dist.all_gather(id_gather, hadm_local)

        r_full = torch.cat([r_gather[i][: lens[i]] for i in range(dist.get_world_size())], dim=0)
        id_full = torch.cat([id_gather[i][: lens[i]] for i in range(dist.get_world_size())], dim=0)
        return {"r": r_full, "hadm_id": id_full}

    def retrieval_step(self, batch, embeddings, split):
        bank = getattr(self, "candidate_bank", None)
        if bank is None:
            return []

        query_indices = [i for i in range(len(self.modalities)) if i != self.candidate_idx]
        keep = self._modality_present(batch, self.candidate_idx)
        for i in query_indices:
            keep = keep & self._modality_present(batch, i)
        if keep.sum().item() == 0:
            return []

        emb_keep = [embeddings[i][keep] for i in range(len(embeddings))]
        rep_list = [emb_keep[i] for i in query_indices]

        r_candidates = bank["r"]
        candidate_ids = bank["hadm_id"]
        y = batch["hadm_id"].to(self.device)[keep]
        n_candidates = int(r_candidates.shape[0])
        if n_candidates == 0:
            return []

        # By default we evaluate with a preselected candidate set of size 10
        # (1 positive + sampled negatives) rather than the full candidate bank.
        use_preselected = self.retrieval_mode != "global"
        k = min(self.preselected_num_candidates, n_candidates) if use_preselected else n_candidates

        results = []
        for q_idx in range(int(y.shape[0])):
            true_id = int(y[q_idx].item())
            if use_preselected:
                positive_idx = torch.nonzero(candidate_ids == true_id, as_tuple=False).flatten()
                if positive_idx.numel() == 0:
                    continue
                pos_idx = int(positive_idx[0].item())
                g = torch.Generator(device="cpu")
                g.manual_seed(self.retrieval_seed + (104729 * q_idx) + true_id)
                if k <= 1:
                    selected = torch.tensor([pos_idx], device=self.device, dtype=torch.long)
                else:
                    neg_all = torch.arange(n_candidates, device=self.device, dtype=torch.long)
                    neg_all = neg_all[neg_all != pos_idx]
                    k_neg = min(k - 1, int(neg_all.numel()))
                    perm = torch.randperm(int(neg_all.numel()), generator=g, device="cpu")[:k_neg].to(device=self.device)
                    neg_selected = neg_all.index_select(0, perm)
                    selected = torch.cat(
                        [torch.tensor([pos_idx], device=self.device, dtype=torch.long), neg_selected],
                        dim=0,
                    )
                if selected.numel() > 1:
                    # Position-unbiased eval: shuffle candidate order so the positive
                    # sample is not always at index 0 when ties occur.
                    selected_perm = torch.randperm(int(selected.numel()), generator=g, device="cpu").to(device=self.device)
                    selected = selected.index_select(0, selected_perm)
            else:
                selected = torch.arange(n_candidates, device=self.device, dtype=torch.long)

            cand_sel = r_candidates.index_select(0, selected)
            cand_id_sel = candidate_ids.index_select(0, selected)

            if self.modelname in LEARNED_SCORERS:
                emb_dim = int(emb_keep[0].shape[1])
                n_sel = int(cand_sel.shape[0])
                pair_embs = []
                for i in range(len(emb_keep)):
                    if i == self.candidate_idx:
                        x = cand_sel
                    else:
                        x = emb_keep[i][q_idx].unsqueeze(0).expand(n_sel, emb_dim)
                    pair_embs.append(x)
                z = self.model.transformer(pair_embs)
                if z.dim() == 2 and z.shape[1] == 1:
                    z = z.squeeze(1)
                elif z.dim() != 1:
                    raise ValueError(f"Unexpected transformer score shape: {tuple(z.shape)}")
                logits = z.unsqueeze(0)
                scale = self.get_logit_scale_exp()
                if scale is not None:
                    logits = logits * scale
                if self.bias is not None:
                    logits = logits + self.bias
            else:
                single_rep_list = [rep[q_idx].unsqueeze(0) for rep in rep_list]
                logits = zeroshot_retrieval_logits(
                    cand_sel,
                    single_rep_list,
                    self.get_logit_scale_exp(),
                    bias=self.bias,
                    modelname=self.modelname,
                )

            pred = cand_id_sel[torch.argmax(logits, dim=1)[0]]
            results.append(float(pred.item() == true_id))

        return results
