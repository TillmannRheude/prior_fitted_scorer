from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class INSPECTDataset(Dataset):
    """Aligned INSPECT vision, EHR, and radiology-report triples."""

    DEFAULT_DATA_DIR = Path(
        "/sc-projects/sc-proj-ukb-cvd/projects/data/inspect"
    )

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        split: str = "train",
        split_nr: int = 0,
        text_model_id: str = "StanfordAIMI/RadBERT",
        max_text_length: int = 256,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = "valid" if split == "val" else str(split)
        self.split_nr = int(split_nr)
        self.text_model_id = str(text_model_id)
        self.max_text_length = int(max_text_length)

        split_column = f"split_{self.split_nr}"
        splits = pd.read_pickle(self.data_dir / "cv_splits.pkl")
        if split_column not in splits:
            raise ValueError(
                f"INSPECT split {self.split_nr} is unavailable; "
                f"expected column {split_column!r}."
            )
        splits = splits.loc[
            splits[split_column] == self.split,
            ["impression_id", "person_id"],
        ].copy()

        ehr = pd.read_pickle(self.data_dir / "ehr_motor_embeddings.pkl")
        ehr = ehr.rename(columns={"patient_ids": "person_id"})
        reports = pd.read_csv(
            self.data_dir / "impressions_20250611.tsv",
            sep="\t",
            usecols=["impression_id", "impressions"],
        )
        targets = pd.read_pickle(self.data_dir / "PE_targets.pkl")
        targets = targets.loc[:, ["image_id", "label"]].rename(
            columns={"label": "pe_label"}
        )

        frame = ehr.merge(
            splits,
            on=["impression_id", "person_id"],
            how="inner",
            validate="one_to_one",
        )
        frame = frame.merge(
            reports,
            on="impression_id",
            how="inner",
            validate="one_to_one",
        )
        frame = frame.merge(
            targets,
            on="image_id",
            how="inner",
            validate="one_to_one",
        )

        vision_dir = self.data_dir / "vision_med3dvlm_embeddings"
        suffix = "_vision_embeddings.npz"
        vision_paths = {
            path.name[: -len(suffix)]: path
            for path in vision_dir.glob(f"*{suffix}")
        }
        frame["vision_path"] = frame["image_id"].map(vision_paths)
        frame = frame.loc[frame["vision_path"].notna()].reset_index(drop=True)

        if frame.empty:
            raise ValueError(
                f"No complete INSPECT triples found for split={self.split!r}, "
                f"split_nr={self.split_nr}."
            )
        if frame["image_id"].duplicated().any():
            raise ValueError("INSPECT image_id must uniquely identify a study.")
        if frame["impression_id"].duplicated().any():
            raise ValueError(
                "INSPECT impression_id must uniquely identify a report."
            )

        self.frame = frame
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_id)

    def __len__(self) -> int:
        return len(self.frame)

    def _tokenize_report(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    @staticmethod
    def _parse_pe_label(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]

        with np.load(row["vision_path"]) as archive:
            vision = torch.from_numpy(
                np.asarray(archive["arr_0"], dtype=np.float32)
            )
        if vision.ndim != 4 or vision.shape[0] != 768:
            raise ValueError(
                f"Expected INSPECT vision features shaped (768,D,H,W), "
                f"got {tuple(vision.shape)} for image_id={row['image_id']}."
            )
        ehr = torch.as_tensor(
            np.asarray(row["embeddings"], dtype=np.float32).copy()
        )
        if ehr.shape != (769,):
            raise ValueError(
                f"Expected INSPECT EHR embedding shape (769,), got "
                f"{tuple(ehr.shape)} for image_id={row['image_id']}."
            )

        report_text = "" if pd.isna(row["impressions"]) else str(row["impressions"])
        report = self._tokenize_report(report_text)

        return {
            "vision": vision.contiguous(),
            "ehr": ehr.contiguous(),
            "report": report,
            "image_id": str(row["image_id"]),
            "impression_id": int(row["impression_id"]),
            "person_id": int(row["person_id"]),
            "pe_label": self._parse_pe_label(row["pe_label"]),
        }


def inspect_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "vision": torch.stack([sample["vision"] for sample in batch]),
        "ehr": torch.stack([sample["ehr"] for sample in batch]),
        "report": {
            "input_ids": torch.stack(
                [sample["report"]["input_ids"] for sample in batch]
            ),
            "attention_mask": torch.stack(
                [sample["report"]["attention_mask"] for sample in batch]
            ),
        },
        "image_id": [sample["image_id"] for sample in batch],
        "impression_id": torch.tensor(
            [sample["impression_id"] for sample in batch],
            dtype=torch.long,
        ),
        "person_id": torch.tensor(
            [sample["person_id"] for sample in batch],
            dtype=torch.long,
        ),
        "pe_label": torch.tensor(
            [sample["pe_label"] for sample in batch],
            dtype=torch.bool,
        ),
    }
