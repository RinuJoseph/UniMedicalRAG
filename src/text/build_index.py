"""Offline index build for the passage-entity-sentence task."""
import json
import os

from src.text import indexing, stores


def build(cfg, rebuild=False):
    out_dir = cfg.path("index_dir")
    marker = out_dir / "ner_results.json"
    if marker.exists() and not rebuild:
        print(f"[index] cache present in {out_dir} ; pass --rebuild to force a rebuild")
        return
    import spacy
    from sentence_transformers import SentenceTransformer

    os.makedirs(out_dir, exist_ok=True)
    chunks = stores.load_chunks(cfg.path("dataset_dir"))
    print(f"[data] {len(chunks)} passages")
    nlp = spacy.load(cfg.spacy_model)
    encoder = SentenceTransformer(str(cfg.path("embedding_model")), device=cfg.get("device", "cpu"))
    indexing.build(nlp, encoder, chunks, out_dir)
    print(f"[save] {out_dir}")
