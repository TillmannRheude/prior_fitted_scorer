from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MimicDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        mimic_root: str | Path = ".",
        text_model_id: str = "StanfordAIMI/RadBERT",
        max_text_length: int = 256,
        load_radiology_text: bool = True,
        require_all_modalities: bool = False,
        radiology_chunk_size: int = 250_000,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.mimic_root = Path(mimic_root)
        self.text_model_id = str(text_model_id)
        self.max_text_length = int(max_text_length)
        self.load_radiology_text = bool(load_radiology_text)
        self.require_all_modalities = bool(require_all_modalities)
        self.radiology_chunk_size = int(radiology_chunk_size)

        self.df = pd.read_csv(self.split_csv)
        self.timeseries_columns = sorted([c for c in self.df.columns if c.startswith("ts_item")])
        self.ts_count_columns = [c for c in self.timeseries_columns if c.endswith("_count")]
        self.timeseries_dim = int(len(self.timeseries_columns))
        self.demographics_dim = 5

        self.radiology_text_lookup: dict[str, str] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_id)

        if self.load_radiology_text:
            note_ids = set(self.df["note_id"].astype(str).tolist())
            note_dir = self.mimic_root / "note"
            radiology_path = note_dir / "radiology.csv"
            compression = "gzip" if radiology_path.suffix == ".gz" else None
            for chunk in pd.read_csv(
                radiology_path,
                usecols=["note_id", "text"],
                compression=compression,
                chunksize=self.radiology_chunk_size,
            ):
                chunk["note_id"] = chunk["note_id"].astype(str)
                sub = chunk.loc[chunk["note_id"].isin(note_ids), ["note_id", "text"]]
                for note_id, text in zip(sub["note_id"], sub["text"]):
                    self.radiology_text_lookup[str(note_id)] = "" if pd.isna(text) else str(text)
                if len(self.radiology_text_lookup) == len(note_ids):
                    break

        if self.require_all_modalities:
            keep: list[int] = []
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                if self._has_radiology(row) and self._has_timeseries(row) and self._has_demographics(row):
                    keep.append(i)
            self.indices = np.asarray(keep, dtype=np.int64)
        else:
            self.indices = np.arange(len(self.df), dtype=np.int64)

    def _to_float_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr.astype(np.float32, copy=False), dtype=torch.float32)

    def _nan_tensor(self, length: int) -> torch.Tensor:
        return torch.full((length,), float("nan"), dtype=torch.float32)

    def _has_radiology(self, row: pd.Series) -> bool:
        if not self.load_radiology_text:
            return False
        text = self.radiology_text_lookup.get(str(row.get("note_id")), "")
        return bool(text.strip())

    def _has_timeseries(self, row: pd.Series) -> bool:
        if len(self.ts_count_columns) > 0:
            counts = pd.to_numeric(row[self.ts_count_columns], errors="coerce").fillna(0.0)
            return float(counts.sum()) > 0.0
        ts_values = pd.to_numeric(row[self.timeseries_columns], errors="coerce")
        return not bool(ts_values.isna().all())

    def _has_demographics(self, row: pd.Series) -> bool:
        return (not pd.isna(row.get("anchor_age"))) and (not pd.isna(row.get("gender")))

    def _tokenize_report(self, text: str) -> dict[str, torch.Tensor]:
        tok = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
        }

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[int(self.indices[idx])]
        note_id = str(row["note_id"])

        if self._has_radiology(row):
            report_text = self.radiology_text_lookup[note_id]
            rad_tokens = self._tokenize_report(report_text)
            report_present = True
        else:
            report_text = ""
            rad_tokens = self._tokenize_report("")
            report_present = False

        if self._has_timeseries(row):
            ts_values = pd.to_numeric(row[self.timeseries_columns], errors="coerce").fillna(0.0).to_numpy()
            ts_tensor = self._to_float_tensor(ts_values)
            ts_present = True
        else:
            ts_tensor = self._nan_tensor(self.timeseries_dim)
            ts_present = False

        if self._has_demographics(row):
            gender = str(row["gender"]).strip().upper()
            gender_oh = np.array(
                [1.0 if gender == "F" else 0.0, 1.0 if gender == "M" else 0.0, 1.0 if gender not in {"F", "M"} else 0.0],
                dtype=np.float32,
            )
            year_text = str(row.get("anchor_year_group", "")).strip()
            year_head = year_text.split("-")[0].strip() if year_text else ""
            try:
                anchor_year_start = float(int(year_head))
            except ValueError:
                anchor_year_start = np.nan
            demo_values = np.concatenate(
                (np.array([float(row["anchor_age"]), anchor_year_start], dtype=np.float32), gender_oh),
                axis=0,
            )
            if np.isnan(demo_values).any():
                demo_tensor = self._nan_tensor(self.demographics_dim)
                demo_present = False
            else:
                demo_tensor = self._to_float_tensor(demo_values)
                demo_present = True
        else:
            demo_tensor = self._nan_tensor(self.demographics_dim)
            demo_present = False

        return {
            "sample_id": row.get("sample_id"),
            "subject_id": int(float(row["subject_id"])),
            "hadm_id": int(float(row["hadm_id"])),
            "stay_id": int(float(row["stay_id"])),
            "note_id": note_id,
            "radiology": {
                "input_ids": rad_tokens["input_ids"],
                "attention_mask": rad_tokens["attention_mask"],
                "embedding_present": report_present,
                "text": report_text,
            },
            "timeseries": ts_tensor,
            "timeseries_present": ts_present,
            "demographics": demo_tensor,
            "demographics_present": demo_present,
            "mortality": int(row["mortality_label"]),
        }


def mimic_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [s["sample_id"] for s in batch],
        "subject_id": torch.tensor([s["subject_id"] for s in batch], dtype=torch.long),
        "hadm_id": torch.tensor([s["hadm_id"] for s in batch], dtype=torch.long),
        "stay_id": torch.tensor([s["stay_id"] for s in batch], dtype=torch.long),
        "note_id": [s["note_id"] for s in batch],
        "radiology": {
            "input_ids": torch.stack([s["radiology"]["input_ids"] for s in batch], dim=0),
            "attention_mask": torch.stack([s["radiology"]["attention_mask"] for s in batch], dim=0),
            "embedding_present": torch.tensor([s["radiology"]["embedding_present"] for s in batch], dtype=torch.bool),
            "text": [s["radiology"]["text"] for s in batch],
        },
        "timeseries": torch.stack([s["timeseries"] for s in batch], dim=0),
        "timeseries_present": torch.tensor([s["timeseries_present"] for s in batch], dtype=torch.bool),
        "demographics": torch.stack([s["demographics"] for s in batch], dim=0),
        "demographics_present": torch.tensor([s["demographics_present"] for s in batch], dtype=torch.bool),
        "mortality": torch.tensor([s["mortality"] for s in batch], dtype=torch.long),
    }
