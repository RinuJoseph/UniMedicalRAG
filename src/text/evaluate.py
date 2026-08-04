"""QA accuracy over the retrieved passages"""
import json
import os

import numpy as np

from src.common import config as config_mod
from src.common import llm, prompts


def extract_answer(raw):
    try:
        return raw.split("Answer:")[1].strip()
    except Exception:
        return raw


def main(cfg, args):
    src = cfg.path("results_dir") / "retrieval.json"
    if not src.exists():
        raise SystemExit(f"{src} not found -- run scripts/run.py first")
    items = json.load(open(src, encoding="utf-8"))
    if args.limit:
        items = items[:args.limit]
    for i, it in enumerate(items):
        it.setdefault("id", i)
    n_ctx = int(cfg.get("context_passages", 5))
    qa_model = cfg.get("qa_model", "gpt-4o-mini")
    grade_model = cfg.get("grade_model", "gpt-4o-mini")
    print(f"[data] {cfg.name}: {len(items)} questions; answering from the top-{n_ctx} passages "
          f"with {qa_model}, grading with {grade_model}")

    build_qa = lambda it: prompts.qa(it["retrieved"][:n_ctx], it["question"])
    if args.dry_run:
        s, p = build_qa(items[0])
        mean = float(np.mean([len(build_qa(it)[1]) for it in items]))
        cost, tin = llm.estimate(len(items), mean, 300, qa_model)
        print("\n" + "=" * 78 + "\nSAMPLE ANSWER PROMPT (truncated)\n" + "=" * 78)
        print(p[:1200] + f"\n   ... [{len(p)} chars]")
        print("=" * 78)
        print(f"mean prompt ~{tin:.0f} tokens -> ${cost:.2f} for answering; "
              f"grading adds roughly ${cost * 0.1:.2f}")
        return
    if not args.run:
        return

    out_dir = cfg.path("results_dir")
    os.makedirs(out_dir, exist_ok=True)
    key = config_mod.api_key()
    answers = llm.run_batch([(it["id"], it) for it in items], build_qa,
                            str(out_dir / "qa_responses.jsonl"), qa_model, key,
                            max_tokens=600, workers=args.workers)
    preds = {i: extract_answer(t) for i, t in answers.items()}

    build_grade = lambda it: prompts.grade(it.get("gold_answer", ""), preds.get(str(it["id"]), ""))
    grades = llm.run_batch([(it["id"], it) for it in items], build_grade,
                           str(out_dir / "grade_responses.jsonl"), grade_model, key,
                           max_tokens=10, workers=args.workers)

    ok = [1.0 if (grades.get(str(it["id"]), "").strip().lower() == "correct") else 0.0
          for it in items]
    acc = 100 * float(np.mean(ok))
    print(f"\nQA accuracy   n={len(items)}, answerer={qa_model}, grader={grade_model}")
    print(f"  Accuracy {acc:.2f}")

    import pandas as pd
    pd.DataFrame([{"id": it["id"], "question": it["question"],
                   "gold_answer": it.get("gold_answer", ""),
                   "prediction": preds.get(str(it["id"]), ""),
                   "grade": grades.get(str(it["id"]), "")} for it in items]).to_csv(
        out_dir / "qa.csv", index=False)
    json.dump({"qa_model": qa_model, "grade_model": grade_model, "n": len(items),
               "accuracy": round(acc, 2)}, open(out_dir / "qa.json", "w"), indent=2)
    print(f"[save] {out_dir}/qa.csv")
