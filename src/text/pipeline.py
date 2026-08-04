"""Retrieval for the passage-entity-sentence task."""
import json
import os
import time

from src.text.retriever import Retriever
from src.text import stores


def main(cfg, args):
    print(f"[data] {cfg.name}: solver={cfg.get('solver')} damping={cfg.get('damping')} "
          f"top_sentence={cfg.get('top_sentence')} "
          f"weights=({cfg.w_entity}, {cfg.w_sentence}, {cfg.w_passage})")
    retriever = Retriever(cfg)
    print("[graph] " + retriever.describe())

    questions = stores.load_questions(cfg.path("dataset_dir"))
    if args.limit:
        questions = questions[:args.limit]

    start = time.time()
    results = retriever.retrieve(questions)
    elapsed = time.time() - start

    out_dir = cfg.path("results_dir")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "retrieval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[done] {len(results)} questions in {elapsed:.0f}s "
          f"({elapsed / max(len(results), 1):.3f}s each)")
    print(f"[save] {out_dir}/retrieval.json")
