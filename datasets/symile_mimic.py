import torch
import os

from torch.utils.data import Dataset


class Dataset_SymileMimic(Dataset):
    DEFAULT_DATA_DIR = "data/mimic-symile"

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        split: str = "train",
        split_nr: int = 0,
    ):
        self.data_dir = data_dir
        self.split = split
        self.split_nr = split_nr

        self.cxr = torch.load(os.path.join(self.data_dir, f"split{self.split_nr}/{self.split}/cxr_{self.split}.pt"))
        self.ecg = torch.load(os.path.join(self.data_dir, f"split{self.split_nr}/{self.split}/ecg_{self.split}.pt"))
        self.labs_percentiles = torch.load(os.path.join(self.data_dir, f"split{self.split_nr}/{self.split}/labs_percentiles_{self.split}.pt"))
        self.labs_missingness = torch.load(os.path.join(self.data_dir, f"split{self.split_nr}/{self.split}/labs_missingness_{self.split}.pt"))
        self.hadm_id = torch.load(os.path.join(self.data_dir, f"split{self.split_nr}/{self.split}/hadm_id_{self.split}.pt"))

        if "retrieval" in self.split or self.split == "test":
            self.label_hadm_id = torch.load(os.path.join(self.data_dir, f"split{self.split_nr}/{self.split}/label_hadm_id_{self.split}.pt"))
            self.label = torch.load(os.path.join(self.data_dir, f"split{self.split_nr}/{self.split}/label_{self.split}.pt"))

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        labs = torch.cat([self.labs_percentiles, self.labs_missingness], dim=1)[idx]

        return_dict = {
            "cxr": self.cxr[idx],
            "ecg": self.ecg[idx],
            "labs": labs,
            "hadm_id": self.hadm_id[idx],
        }
        if "retrieval" in self.split or self.split == "test":
            return_dict["label_hadm_id"] = self.label_hadm_id[idx]
            return_dict["label"] = self.label[idx]
            
        return return_dict
