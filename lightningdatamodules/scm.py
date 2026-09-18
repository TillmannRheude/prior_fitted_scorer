from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.scm import SCMDataset
from scm.framework import SCMConfig


class DataModule_SCMPretrain(pl.LightningDataModule):
    def __init__(
        self,
        n_train_scms: int = 10_000,
        n_val_scms: int = 1_000,
        n_test_scms: int = 1_000,
        samples_per_scm: int = 128,
        scms_per_batch: int = 1,
        num_workers: int = 0,
        seed: int = 420,
        scm_config: SCMConfig | None = None,
        control_mode: str = "none",
    ):
        super().__init__()

        self.n_train_scms = int(n_train_scms)
        self.n_val_scms = int(n_val_scms)
        self.n_test_scms = int(n_test_scms)
        self.samples_per_scm = int(samples_per_scm)
        self.scms_per_batch = int(scms_per_batch)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.scm_config = scm_config if scm_config is not None else SCMConfig(seed=seed)
        self.control_mode = str(control_mode)

    def setup(self, stage: str | None = None) -> None:
        train_start = 0
        val_start = train_start + self.n_train_scms
        test_start = val_start + self.n_val_scms

        self.train_indices = range(train_start, val_start)
        self.val_indices = range(val_start, test_start)
        self.test_indices = range(test_start, test_start + self.n_test_scms)

    def _make_dataset(self, indices, shuffle: bool, seed_offset: int = 0) -> SCMDataset:
        return SCMDataset(
            scm_indices=indices,
            samples_per_scm=self.samples_per_scm,
            scm_config=self.scm_config,
            shuffle=shuffle,
            seed=self.seed + int(seed_offset),
            control_mode=self.control_mode,
            scms_per_batch=self.scms_per_batch,
        )

    def train_dataloader(self):
        dataset = self._make_dataset(self.train_indices, shuffle=True, seed_offset=0)
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = self._make_dataset(self.val_indices, shuffle=False, seed_offset=10_000)
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = self._make_dataset(self.test_indices, shuffle=False, seed_offset=20_000)
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
