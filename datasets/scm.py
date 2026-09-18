from __future__ import annotations

from collections.abc import Iterable
from math import ceil

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from scm.framework import SCMConfig, SCMPrior


class SCMDataset(IterableDataset):
    """
    Streams one generated SCM dataset at a time.

    Each yielded item is already a training batch with matched samples from one
    SCM: A_i, B_i, C_i are positives, and other rows in the same batch are
    negatives.
    """

    def __init__(
        self,
        scm_indices: Iterable[int],
        samples_per_scm: int,
        scm_config: SCMConfig,
        shuffle: bool = False,
        seed: int = 0,
        control_mode: str = "none",
        scms_per_batch: int = 1,
    ):
        super().__init__()

        if int(scm_config.n_modalities) != 3:
            raise ValueError("SCMDataset currently expects exactly three modalities.")
        if int(samples_per_scm) < 2:
            raise ValueError("samples_per_scm must be at least 2 for contrastive training.")
        if int(scms_per_batch) < 1:
            raise ValueError("scms_per_batch must be at least 1.")

        self.scm_indices = [int(index) for index in scm_indices]
        self.samples_per_scm = int(samples_per_scm)
        self.scm_config = scm_config
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.control_mode = str(control_mode)
        self.scms_per_batch = int(scms_per_batch)
        if self.control_mode not in {"none", "permute_c", "permute_bc"}:
            raise ValueError(
                f"Unsupported SCM control_mode={self.control_mode!r}. "
                "Expected 'none', 'permute_c', or 'permute_bc'."
            )

    def __len__(self) -> int:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return ceil(ceil(len(self.scm_indices) / world_size) / self.scms_per_batch)

    def _worker_indices(self) -> list[int]:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        info = get_worker_info()
        if info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(info.id)
            num_workers = int(info.num_workers)

        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        return list(self.scm_indices)[shard_id::num_shards]

    def __iter__(self):
        indices = self._worker_indices()

        if self.shuffle and len(indices) > 1:
            info = get_worker_info()
            worker_id = 0 if info is None else int(info.id)
            generator = torch.Generator().manual_seed(self.seed + worker_id)
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[index] for index in order]

        prior = SCMPrior(self.scm_config)

        batch_items = []
        for scm_index in indices:
            dataset, _ = prior.sample_dataset(
                scm_index=scm_index,
                n_samples=self.samples_per_scm,
            )
            b = dataset["modality_1"]
            c = dataset["modality_2"]
            if self.control_mode in {"permute_c", "permute_bc"}:
                generator_c = torch.Generator().manual_seed(
                    self.seed + 1_000_000 + int(scm_index)
                )
                c = c[torch.randperm(c.shape[0], generator=generator_c)]
            if self.control_mode == "permute_bc":
                generator_b = torch.Generator().manual_seed(
                    self.seed + 2_000_000 + int(scm_index)
                )
                b = b[torch.randperm(b.shape[0], generator=generator_b)]

            item = {
                "A": dataset["modality_0"].float(),
                "B": b.float(),
                "C": c.float(),
                "scm_id": torch.full(
                    (self.samples_per_scm,),
                    int(scm_index),
                    dtype=torch.long,
                ),
                "sample_id": torch.arange(self.samples_per_scm, dtype=torch.long),
            }
            batch_items.append(item)

            if len(batch_items) == self.scms_per_batch:
                yield self._collate_scm_batch(batch_items)
                batch_items = []

        if batch_items:
            yield self._collate_scm_batch(batch_items)

    def _collate_scm_batch(self, batch_items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if len(batch_items) == 1 and self.scms_per_batch == 1:
            return batch_items[0]

        return {
            key: torch.stack([item[key] for item in batch_items], dim=0)
            for key in batch_items[0]
        }
