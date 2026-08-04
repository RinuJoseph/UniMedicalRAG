"""Builds the passage, entity and sentence embedding tables."""
import json
from collections import defaultdict
from hashlib import md5

import numpy as np
import pandas as pd

SKIP_LABELS = {"ORDINAL", "CARDINAL"}


def hash_id(text: str, prefix: str) -> str:
    return prefix + md5(text.encode()).hexdigest()


def extract(nlp, passages, batch_size=8):
    """Run NER over every passage"""
    passage_entities = {}          # passage hash id -> entity texts
    sentence_entities = {}         # sentence text   -> entity texts
    sentence_passage = {}          # sentence text   -> passage index
    failures = []

    from tqdm import tqdm
    docs = nlp.pipe(passages, batch_size=batch_size)
    for index, doc in enumerate(tqdm(docs, total=len(passages), desc="ner")):
        try:
            found = set()
            for ent in doc.ents:
                if ent.label_ in SKIP_LABELS:
                    continue
                sentence = ent.sent.text
                found.add(ent.text)
                bucket = sentence_entities.setdefault(sentence, [])
                if ent.text not in bucket:
                    bucket.append(ent.text)
                # A sentence can occur in several passages; keep the first, so the map is
                # one-to-one and does not depend on iteration order.
                sentence_passage.setdefault(sentence, index)
            passage_entities[hash_id(passages[index], "passage-")] = sorted(found)
        except Exception as exc:                      # noqa: BLE001 - report, do not hide
            failures.append((index, str(exc)[:120]))

    if failures:
        print(f"  WARNING: {len(failures)} passages failed NER and contribute no entities:")
        for index, message in failures[:5]:
            print(f"    passage {index}: {message}")

    return passage_entities, sentence_entities, sentence_passage


def embed_store(encoder, texts, prefix, batch_size=64) -> pd.DataFrame:
    """Embed a list of texts into the parquet layout the retriever reads."""
    embeddings = encoder.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False, batch_size=batch_size)
    return pd.DataFrame({
        "hash_id": [hash_id(t, prefix) for t in texts],
        "text": texts,
        "embedding": [np.asarray(e, dtype=np.float32) for e in embeddings],
    })


def build(nlp, encoder, chunks, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # The chunk index is carried in the passage text; downstream code reads it back out.
    passages = [f"{i}:{chunk}" for i, chunk in enumerate(chunks)]

    passage_entities, sentence_entities, sentence_passage = extract(nlp, passages)

    # Sorted, so two builds of the same corpus produce byte-identical files. The original
    # implementation iterated a set here, which left row order up to the interpreter.
    entities = sorted({e for ents in sentence_entities.values() for e in ents}
                      | {e for ents in passage_entities.values() for e in ents})
    sentences = sorted(sentence_entities)

    print(f"  {len(passages)} passages | {len(entities)} entities | {len(sentences)} sentences")

    embed_store(encoder, passages, "passage-").to_parquet(out_dir / "passage_embedding.parquet")
    embed_store(encoder, entities, "entity-").to_parquet(out_dir / "entity_embedding.parquet")
    embed_store(encoder, sentences, "sentence-").to_parquet(out_dir / "sentence_embedding.parquet")

    with open(out_dir / "ner_results.json", "w") as f:
        json.dump({"passage_hash_id_to_entities": passage_entities,
                   "sentence_to_entities": sentence_entities}, f)
    with open(out_dir / "sentence_to_passage.json", "w") as f:
        json.dump(sentence_passage, f)

    return {"passages": len(passages), "entities": len(entities), "sentences": len(sentences)}
