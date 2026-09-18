from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.inspect import INSPECTDataset, inspect_collate_fn


class INSPECTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path = INSPECTDataset.DEFAULT_DATA_DIR,
        batch_size: int = 128,
        split_nr: int = 0,
        num_workers: int = 4,
        text_model_id: str = "StanfordAIMI/RadBERT",
        max_text_length: int = 256,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = str(data_dir)
        self.batch_size = int(batch_size)
        self.split_nr = int(split_nr)
        self.num_workers = int(num_workers)
        self.text_model_id = str(text_model_id)
        self.max_text_length = int(max_text_length)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)

    def _dataset(self, split: str) -> INSPECTDataset:
        return INSPECTDataset(
            data_dir=self.data_dir,
            split=split,
            split_nr=self.split_nr,
            text_model_id=self.text_model_id,
            max_text_length=self.max_text_length,
        )

    def setup(self, stage=None):
        self.ds_train = self._dataset("train")
        self.ds_val = self._dataset("valid")
        self.ds_test = self._dataset("test")

    def _loader(
        self,
        dataset: INSPECTDataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
            collate_fn=inspect_collate_fn,
        )

    def train_dataloader(self):
        return self._loader(
            self.ds_train,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return self._loader(
            self.ds_val,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return self._loader(
            self.ds_test,
            shuffle=False,
            drop_last=False,
        )
