"""Evaluation with the RareBench LLM judge"""
import csv
import json
import os

import numpy as np

from src.common import config as config_mod
from src.common import llm, prompts

csv.field_size_limit(10 ** 9)


def rank_from_judgement(text, topn):
    miss = topn + 1
    if "否" in text or "No" in text:
        return miss
    pat = r"\b(?:10|[1-9])\b" if topn == 10 else r"\b(?:20|1[0-9]|[1-9])\b"
    import re
    found = re.findall(pat, text)
    if not found:
        return miss
    v = int(found[0])
    return v if v <= topn else miss


def load_predictions(cfg, topn):
    path = cfg.path("results_dir") / "ranking.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/run.py --rerank first")
    name_file = cfg.path("disease_names")
    names = json.load(open(name_file, encoding="utf-8-sig")) if name_file and name_file.exists() else {}
    rows = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        pairs = [x.strip() for x in (r["ranking"] or "").split(" | ") if x.strip()][:topn]
        pred = [p.split(" ", 1)[1] if " " in p else p for p in pairs]
        codes = [g.strip() for g in (r["gold"] or "").split(";") if g.strip()]
        gold = [names[c] for c in codes if c in names]
        if not gold:
            gold = [g.strip() for g in (r["gold_names"] or "").split(";") if g.strip()] or codes
        rows.append({"id": int(r["id"]),
                     "pred": "\n".join(f"{i + 1}. {n}" for i, n in enumerate(pred)),
                     "gold": ",".join(gold)})
    return rows


def main(cfg, args):
    topn = int(cfg.get("judge_topn", 10))
    model = cfg.get("judge_model", "gpt-4o")
    rows = load_predictions(cfg, topn)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[data] {cfg.name}: {len(rows)} queries; judging top-{topn} with {model}")
    if topn != 10:
        print("       note: topn other than 10 deviates from the reference protocol")

    build = lambda r: prompts.judge(r["pred"], r["gold"], topn)
    if args.dry_run:
        s, p = build(rows[0])
        mean = float(np.mean([len(build(r)[1]) for r in rows]))
        cost, tin = llm.estimate(len(rows), mean, 5, model)
        print("\n" + "=" * 78 + "\nSAMPLE JUDGE PROMPT\n" + "=" * 78)
        print("[system] " + s + "\n[user]   " + p)
        print("=" * 78)
        print(f"mean prompt ~{tin:.0f} tokens -> ${cost:.2f} for {len(rows)} calls")
        return
    if not args.run:
        return

    out_dir = cfg.path("results_dir")
    os.makedirs(out_dir, exist_ok=True)
    cache = llm.run_batch([(r["id"], r) for r in rows], build,
                          str(out_dir / f"judge_top{topn}_responses.jsonl"),
                          model, config_mod.api_key(), max_tokens=10, workers=args.workers)
    ranks = [rank_from_judgement(cache.get(str(r["id"]), "No"), topn) for r in rows]
    ks = (1, 3, 10) + ((20,) if topn >= 20 else ())
    hit = {k: round(100 * len([x for x in ranks if x <= k]) / len(ranks), 2) for k in ks}
    med = float(np.median(ranks))

    print(f"\nLLM-judge evaluation   n={len(rows)}, judge={model}, top-{topn}")
    print("  " + "".join(f"{'Hit@' + str(k):>10}" for k in ks) + f"{'Median Rank':>14}")
    print("  " + "".join(f"{hit[k]:>10.2f}" for k in ks) + f"{med:>14.0f}")

    import pandas as pd
    pd.DataFrame([{"id": r["id"], "judgement": cache.get(str(r["id"]), "No"),
                   "judge_rank": rk, "predictions": r["pred"], "standard_diagnosis": r["gold"]}
                  for r, rk in zip(rows, ranks)]).to_csv(
        out_dir / f"judge_top{topn}.csv", index=False)
    json.dump({"judge": model, "topn": topn, "n": len(rows),
               **{f"Hit@{k}": hit[k] for k in ks}, "median_rank": med},
              open(out_dir / f"judge_top{topn}.json", "w"), indent=2)
    print(f"[save] {out_dir}/judge_top{topn}.csv")
