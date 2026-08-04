
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Store:
    hash_ids: list
    texts: list
    embeddings: np.ndarray

    def __len__(self):
        return len(self.hash_ids)

    def id_of_text(self) -> dict:
        return {t: h for h, t in zip(self.hash_ids, self.texts)}

    def index_of_id(self) -> dict:
        return {h: i for i, h in enumerate(self.hash_ids)}


def load_store(index_dir: Path, name: str) -> Store:
    df = pd.read_parquet(index_dir / f"{name}_embedding.parquet")
    emb = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    return Store(df["hash_id"].tolist(), df["text"].tolist(), emb)


def load_ner(index_dir: Path) -> dict:
    with open(index_dir / "ner_results.json") as f:
        return json.load(f)


def load_sentence_map(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run:  python scripts/build_index.py"
        )
    with open(path) as f:
        return json.load(f)


def load_questions(dataset_dir: Path) -> list:
    with open(dataset_dir / "questions.json") as f:
        return json.load(f)


def load_chunks(dataset_dir: Path) -> list:
    with open(dataset_dir / "chunks.json") as f:
        return json.load(f)
