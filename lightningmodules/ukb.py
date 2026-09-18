import torch
import torch.distributed as dist

from lightningmodules.utils import LEARNED_SCORERS, LightningModuleParent
from losses.retrieval import zeroshot_retrieval_logits


class UKBModel(LightningModuleParent):
    def __init__(
        self,
        model,
        candidate_idx: int = 0, 
        params_retrival_ds: dict = {
            "batch_size": 128, 
            "split_nr": 1,
        },
        modalities: list = ["nmr", "ehr", "olink"],
        **args,
    ):
        super().__init__(**args)
        self.dataset_name = "ukb"
        self.model = model
        self.params_retrival_ds = params_retrival_ds
        self.modalities = modalities
        self.candidate_idx = candidate_idx

        self.val_step_accuracies = []
        self.test_step_accuracies = []

        self.save_hyperparameters()

    def _get_encoder_stack(self):
        encoders = getattr(self.model, "encoders", None)
        if encoders is None and hasattr(self.model, "contrastive_model"):
            encoders = self.model.contrastive_model.encoders
        if encoders is None:
            raise ValueError("Could not locate encoder stack for UKBModel.")
        return encoders

    @staticmethod
    def _modality_present(x: torch.Tensor) -> torch.Tensor:
        """
        Treat a modality as "missing" for a sample if ALL features are NaN.
        Returns bool mask of shape (B,).
        """
        if x.dim() == 1:
            return ~torch.isnan(x)
        # works for tabular (B,D) and MRI (B,C,...) alike
        B = x.shape[0]
        return ~torch.isnan(x).reshape(B, -1).all(dim=1)

    def _get_modalities(self, batch):
        modality_type_0 = "tabular_data"
        modality_type_1 = "tabular_data"
        modality_type_2 = "tabular_data"
        x0 = batch[self.modalities[0]][modality_type_0]
        x1 = batch[self.modalities[1]][modality_type_1]
        x2 = batch[self.modalities[2]][modality_type_2]
        return [
            x0,
            x1,
            x2,
            #batch["prs"]["tabular_data"],
            #batch["bloodbio"]["tabular_data"],
            #batch["baselinechars"]["tabular_data"],
            #batch["localenvironment"]["tabular_data"],
            #batch["arterialstiffness"]["tabular_data"],
            #batch["anthropometry"]["tabular_data"],
            #batch["bloodpressure"]["tabular_data"],
            #batch["ecgduringexercise"]["tabular_data"],
            #batch["eyemeasures"]["tabular_data"],
            #batch["bonedensitometry"]["tabular_data"],
            #batch["handgripstrength"]["tabular_data"],
            #batch["spirometry"]["tabular_data"],
            #batch["touchscreen"]["tabular_data"],
            #batch["cognitivefunction"]["tabular_data"],
            #batch["hearingtest"]["tabular_data"],
            #batch["verbalinterview"]["tabular_data"],
            #batch["bloodcount"]["tabular_data"],
            #batch["urineassays"]["tabular_data"],
            #batch["telomeres"]["tabular_data"],
            #batch["infectiousdiseases"]["tabular_data"],
        ]

    def _get_modality_missing_mask(self, batch) -> torch.Tensor:
        """
        Return a modality-level mask where True means the entire modality is
        absent for that sample. Feature-level NaNs remain handled by the UKB
        union encoders.
        """
        modalities = self._get_modalities(batch)
        return torch.stack(
            [~self._modality_present(modality) for modality in modalities],
            dim=1,
        )

    def forward(self, batch):
        x = self._get_modalities(batch)
        if self.modelname in LEARNED_SCORERS:
            return self.model(
                x,
                src_key_padding_mask=self._get_modality_missing_mask(batch),
            )
        return self.model(x)
    
    def build_candidate_bank(self, split):
        r_list, cls_list = [], []
        dl = self.trainer.datamodule.val_dataloader() if split == "val" else self.trainer.datamodule.test_dataloader()
        encoder_stack = self._get_encoder_stack()

        for batch in dl:
            mods = self._get_modalities(batch)
            cls_id = torch.tensor(batch["eids"], device=self.device)

            candidates_raw = mods[self.candidate_idx].to(self.device)
            present_cand = self._modality_present(candidates_raw)
            if present_cand.sum().item() == 0:
                continue

            candidates = candidates_raw[present_cand].float() 
            reps = encoder_stack[self.candidate_idx](candidates)
            if self.params_method.get("embedding_norm", False):
                reps = torch.nn.functional.normalize(reps, dim=1)

            r_list.append(reps)
            cls_list.append(cls_id[present_cand])

        ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        # Local tensors (can be empty!)
        if r_list:
            r_local = torch.cat(r_list, dim=0)
            cls_local = torch.cat(cls_list, dim=0)
            emb_dim_local = r_local.shape[1]
        else:
            r_local = None
            cls_local = torch.empty((0,), device=self.device, dtype=torch.long)
            emb_dim_local = -1

        if not ddp:
            if r_local is None:
                return None
            return {"r": r_local, "cls_id": cls_local}

        # --- DDP path: make shapes consistent across ranks ---
        # 1) agree on embedding dim
        emb_dim_t = torch.tensor([emb_dim_local], device=self.device, dtype=torch.long)
        emb_dims = [torch.empty_like(emb_dim_t) for _ in range(dist.get_world_size())]
        dist.all_gather(emb_dims, emb_dim_t)
        emb_dim = int(torch.stack(emb_dims).max().item())
        if emb_dim <= 0:
            return None  # nobody had candidates

        if r_local is None:
            r_local = torch.empty((0, emb_dim), device=self.device, dtype=torch.float32)

        # 2) gather lengths + pad to max_len
        len_t = torch.tensor([r_local.shape[0]], device=self.device, dtype=torch.long)
        lens = [torch.empty_like(len_t) for _ in range(dist.get_world_size())]
        dist.all_gather(lens, len_t)
        lens = [int(x.item()) for x in lens]
        max_len = max(lens)

        if r_local.shape[0] < max_len:
            pad_rows = max_len - r_local.shape[0]
            r_local = torch.cat([r_local, torch.zeros((pad_rows, emb_dim), device=self.device, dtype=r_local.dtype)], dim=0)
            cls_local = torch.cat([cls_local, torch.full((pad_rows,), -1, device=self.device, dtype=cls_local.dtype)], dim=0)

        # 3) all_gather padded tensors
        r_gather = [torch.empty((max_len, emb_dim), device=self.device, dtype=r_local.dtype) for _ in range(dist.get_world_size())]
        cls_gather = [torch.empty((max_len,), device=self.device, dtype=cls_local.dtype) for _ in range(dist.get_world_size())]
        dist.all_gather(r_gather, r_local)
        dist.all_gather(cls_gather, cls_local)

        # 4) unpad + concat
        r_full = torch.cat([r_gather[i][:lens[i]] for i in range(dist.get_world_size())], dim=0)
        cls_full = torch.cat([cls_gather[i][:lens[i]] for i in range(dist.get_world_size())], dim=0)

        return {"r": r_full, "cls_id": cls_full}
    
    def retrieval_step(self, batch, embeddings, split):
        bank = getattr(self, "candidate_bank", None)
        if bank is None:
            return []

        mods = self._get_modalities(batch)
        # query modalities are all except candidate modality
        query_indices = [i for i in range(len(mods)) if i != self.candidate_idx]

        # Build per-sample presence masks from raw inputs (NaNs)
        present_masks = []
        for i in query_indices:
            x_raw = mods[i].to(self.device)
            present_masks.append(self._modality_present(x_raw))  # (B,)

        # only evaluate queries that have at least one query modality AND have candidate idx modality present
        present_count = torch.stack([m.float() for m in present_masks], dim=0).sum(dim=0)  # (B,)
        cand_raw = mods[self.candidate_idx].to(self.device)
        present_target = self._modality_present(cand_raw)  # (B,)
        keep = (present_count > 0) & present_target
        if keep.sum().item() == 0:
            return []

        emb_keep = [embeddings[i][keep] for i in range(len(embeddings))]
        rep_list = [embeddings[i][keep] for i in query_indices]  # list of (B_keep, D)

        r_candidates = bank["r"]
        if self.modelname in LEARNED_SCORERS:
            Bk = emb_keep[0].shape[0]
            if Bk == 0:
                return []
            D = emb_keep[0].shape[1]
            M_total = len(emb_keep)
            Nc = r_candidates.shape[0]

            pair_embs = []
            for m in range(M_total):
                if m == self.candidate_idx:
                    x = r_candidates.unsqueeze(0).expand(Bk, Nc, D).reshape(Bk * Nc, D)
                else:
                    x0 = emb_keep[m]  # (Bk, D)
                    x = x0.unsqueeze(1).expand(Bk, Nc, D).reshape(Bk * Nc, D)
                pair_embs.append(x)

            query_missing = self._get_modality_missing_mask(batch)[keep]
            scorer_mask = (
                query_missing[:, None, :]
                .expand(Bk, Nc, len(emb_keep))
                .clone()
            )
            # Candidate-bank entries are guaranteed present by build_candidate_bank.
            scorer_mask[:, :, self.candidate_idx] = False

            z = self.model.transformer(
                pair_embs,
                src_key_padding_mask=scorer_mask.reshape(
                    Bk * Nc,
                    len(emb_keep),
                ),
            )
            if z.dim() == 2 and z.shape[1] == 1:
                z = z.squeeze(1)
            elif z.dim() != 1:
                raise ValueError(
                    f"Expected transformer score shape (Bk*Nc,) or (Bk*Nc,1), got {tuple(z.shape)}"
                )

            logits = z.view(Bk, Nc)
            scale = self.get_logit_scale_exp()
            if scale is not None:
                logits = logits * scale
            if self.bias is not None:
                logits = logits + self.bias
        else:
            logits = zeroshot_retrieval_logits(
                r_candidates,
                rep_list,
                self.get_logit_scale_exp(),
                bias=self.bias,
                modelname=self.modelname,
            )

        y = torch.tensor(batch["eids"]).to(self.device)[keep]
        r_cls_id = bank["cls_id"]
        pred = r_cls_id[torch.argmax(logits, dim=1)]
        return (y == pred).float().tolist()
    
    def training_step(self, batch, batch_idx):
        # get loss + embeddings from the shared step (this also logs train/loss)
        loss, embeddings = self.shared_step(batch, "train", return_embeddings=True)

        return loss
