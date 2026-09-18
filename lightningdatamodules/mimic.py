import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.mimic import MimicDataset, mimic_collate_fn


class DataModule_MIMIC(pl.LightningDataModule):
    def __init__(
        self,
        train_split_csv: str,
        val_split_csv: str,
        test_split_csv: str,
        mimic_root: str = ".",
        text_model_id: str = "StanfordAIMI/RadBERT",
        max_text_length: int = 256,
        load_radiology_text: bool = True,
        require_all_modalities: bool = False,
        radiology_chunk_size: int = 250_000,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_split_csv = train_split_csv
        self.val_split_csv = val_split_csv
        self.test_split_csv = test_split_csv
        self.mimic_root = mimic_root
        self.text_model_id = text_model_id
        self.max_text_length = int(max_text_length)
        self.load_radiology_text = bool(load_radiology_text)
        self.require_all_modalities = bool(require_all_modalities)
        self.radiology_chunk_size = int(radiology_chunk_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def _build_dataset(self, split_csv: str) -> MimicDataset:
        return MimicDataset(
            split_csv=split_csv,
            mimic_root=self.mimic_root,
            text_model_id=self.text_model_id,
            max_text_length=self.max_text_length,
            load_radiology_text=self.load_radiology_text,
            require_all_modalities=self.require_all_modalities,
            radiology_chunk_size=self.radiology_chunk_size,
        )

    def setup(self, stage=None):
        self.ds_train = self._build_dataset(self.train_split_csv)
        self.ds_val = self._build_dataset(self.val_split_csv)
        self.ds_test = self._build_dataset(self.test_split_csv)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=mimic_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=mimic_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=mimic_collate_fn,
        )
