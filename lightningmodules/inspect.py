from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.inspect import INSPECTDataset, inspect_collate_fn
from lightningmodules.utils import LEARNED_SCORERS, LightningModuleParent
from losses.retrieval import zeroshot_retrieval_logits


class INSPECTModel(LightningModuleParent):
    """Trimodal contrastive model with preselected INSPECT retrieval."""

    def __init__(
        self,
        model,
        candidate_idx: int = 0,
        params_retrival_ds: dict | None = None,
        **args,
    ):
        super().__init__(**args)
        self.dataset_name = "inspect"
        self.model = model
        self.modalities = ["vision", "ehr", "report"]
        self.candidate_idx = int(candidate_idx)
        if not 0 <= self.candidate_idx < len(self.modalities):
            raise ValueError(f"Invalid INSPECT candidate_idx={candidate_idx}.")

        defaults = {
            "batch_size": 128,
            "split_nr": 0,
            "preselected_num_candidates": 10,
            "retrieval_seed": 420,
            "data_dir": str(INSPECTDataset.DEFAULT_DATA_DIR),
            "num_workers": 4,
            "text_model_id": "StanfordAIMI/RadBERT",
            "max_text_length": 256,
            "same_label_negatives": False,
        }
        defaults.update(params_retrival_ds or {})
        self.params_retrival_ds = defaults

        if int(self.params_retrival_ds["preselected_num_candidates"]) != 10:
            raise ValueError(
                "INSPECT retrieval currently requires exactly 10 candidates."
            )

        self.save_hyperparameters(ignore=["model"])

    def _get_modalities(self, batch):
        return [batch[name] for name in self.modalities]

    def forward(self, batch):
        return self.model(self._get_modalities(batch))

    @staticmethod
    def _move_to_device(value, device):
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {
                key: INSPECTModel._move_to_device(item, device)
                for key, item in value.items()
            }
        return value

    def _retrieval_dataset(self, split: str) -> INSPECTDataset:
        dataset_split = "valid" if split == "val" else "test"
        return INSPECTDataset(
            data_dir=self.params_retrival_ds["data_dir"],
            split=dataset_split,
            split_nr=int(self.params_retrival_ds["split_nr"]),
            text_model_id=self.params_retrival_ds["text_model_id"],
            max_text_length=int(
                self.params_retrival_ds["max_text_length"]
            ),
        )

    @torch.no_grad()
    def _encode_retrieval_split(self, split: str):
        dataset = self._retrieval_dataset(split)
        loader = DataLoader(
            dataset,
            batch_size=int(self.params_retrival_ds["batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=int(self.params_retrival_ds["num_workers"]),
            persistent_workers=int(self.params_retrival_ds["num_workers"]) > 0,
            collate_fn=inspect_collate_fn,
        )

        encoded = [[] for _ in self.modalities]
        person_ids = []
        pe_labels = []

        for batch in loader:
            batch = self._move_to_device(batch, self.device)
            embeddings = self.forward(batch)["embeddings"]
            if self.params_method["embedding_norm"]:
                embeddings = [
                    F.normalize(embedding, dim=-1)
                    for embedding in embeddings
                ]
            for modality, embedding in enumerate(embeddings):
                encoded[modality].append(embedding)
            person_ids.append(batch["person_id"])
            pe_labels.append(batch["pe_label"])

        return {
            "embeddings": [
                torch.cat(chunks, dim=0) for chunks in encoded
            ],
            "person_id": torch.cat(person_ids, dim=0),
            "pe_label": torch.cat(pe_labels, dim=0),
        }

    def _candidate_indices(self, encoded, split: str):
        person_ids = encoded["person_id"].detach().cpu().numpy()
        pe_labels = encoded["pe_label"].detach().cpu().numpy()
        studies_by_patient = defaultdict(list)
        for index, person_id in enumerate(person_ids.tolist()):
            studies_by_patient[int(person_id)].append(index)

        patients_by_label = defaultdict(list)
        for person_id, indices in studies_by_patient.items():
            patient_labels = {
                bool(pe_labels[index]) for index in indices
            }
            for label in patient_labels:
                patients_by_label[label].append(person_id)

        split_offset = 0 if split == "val" else 1_000_000
        rng = np.random.default_rng(
            int(self.params_retrival_ds["retrieval_seed"]) + split_offset
        )
        same_label = bool(
            self.params_retrival_ds["same_label_negatives"]
        )

        candidate_sets = []
        for query_index, query_patient in enumerate(person_ids.tolist()):
            if same_label:
                patient_pool = patients_by_label[bool(pe_labels[query_index])]
            else:
                patient_pool = list(studies_by_patient)
            eligible_patients = [
                patient_id
                for patient_id in patient_pool
                if patient_id != int(query_patient)
            ]
            if len(eligible_patients) < 9:
                raise ValueError(
                    "INSPECT split has fewer than nine eligible negative patients."
                )

            negative_patients = rng.choice(
                eligible_patients,
                size=9,
                replace=False,
            )
            negatives = []
            for patient_id in negative_patients:
                studies = studies_by_patient[int(patient_id)]
                if same_label:
                    query_label = bool(pe_labels[query_index])
                    studies = [
                        index
                        for index in studies
                        if bool(pe_labels[index]) == query_label
                    ]
                negatives.append(int(rng.choice(studies)))
            candidate_sets.append([query_index, *negatives])

        return candidate_sets

    def _learned_scorer_logits(self, embeddings, candidate_indices):
        query_index = candidate_indices[0]
        n_candidates = len(candidate_indices)
        scorer_inputs = []

        for modality, modality_embeddings in enumerate(embeddings):
            if modality == self.candidate_idx:
                indices = torch.tensor(
                    candidate_indices,
                    device=modality_embeddings.device,
                    dtype=torch.long,
                )
                scorer_input = modality_embeddings.index_select(0, indices)
            else:
                scorer_input = (
                    modality_embeddings[query_index : query_index + 1]
                    .expand(n_candidates, -1)
                )
            scorer_inputs.append(scorer_input)

        logits = self.model.transformer(scorer_inputs)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        elif logits.ndim != 1:
            raise ValueError(
                f"Expected INSPECT scorer logits shaped ({n_candidates},) "
                f"or ({n_candidates},1), got {tuple(logits.shape)}."
            )
        return logits

    @torch.no_grad()
    def zeroshot_retrieval(self, split: str):
        encoded = self._encode_retrieval_split(split)
        embeddings = encoded["embeddings"]
        candidate_sets = self._candidate_indices(encoded, split)
        correct = 0

        for candidate_indices in candidate_sets:
            query_index = candidate_indices[0]
            if self.modelname in LEARNED_SCORERS:
                logits = self._learned_scorer_logits(
                    embeddings,
                    candidate_indices,
                )
                scale = self.get_logit_scale_exp()
                if scale is not None:
                    logits = logits * scale
                if self.bias is not None:
                    logits = logits + self.bias
            else:
                candidate_index_tensor = torch.tensor(
                    candidate_indices,
                    device=embeddings[self.candidate_idx].device,
                    dtype=torch.long,
                )
                candidates = embeddings[self.candidate_idx].index_select(
                    0,
                    candidate_index_tensor,
                )
                queries = [
                    embedding[query_index : query_index + 1]
                    for modality, embedding in enumerate(embeddings)
                    if modality != self.candidate_idx
                ]
                logits = zeroshot_retrieval_logits(
                    candidates,
                    queries,
                    self.get_logit_scale_exp(),
                    bias=self.bias,
                    modelname=self.modelname,
                ).squeeze(0)

            correct += int(torch.argmax(logits).item() == 0)

        return {
            "acc@top1": correct / len(candidate_sets),
            "n_queries": len(candidate_sets),
            "n_candidates": 10,
        }

    def run_zeroshot_retrieval(self, set: str = "val"):
        return self.zeroshot_retrieval(set)
