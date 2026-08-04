"""Retrieval and LLM rerank for the phenotype knowledge-graph task."""
import json
import os
import re

import numpy as np

from src.common import config as config_mod
from src.common import llm, metrics, prompts
from src.kg import build as kg_build
from src.kg import method

KS = (1, 3, 5, 10, 20)


def retrieve(cfg):
    kg_build.configure(cfg)
    method.configure(cfg)
    idx = kg_build.ensure()
    cases = method.load_cases(idx)
    print(f"[data] {cfg.name}: {len(cases)} queries; "
          f"graph {idx['nD']} diseases x {idx['nP']} phenotypes")
    S = method.run(idx, cases)

    dis, rd_name, C = idx["diseases"], idx["rd_name"], idx["C"].tocsr()
    members = {}
    for code, i in idx["di"].items():
        members.setdefault(i, []).append(code)

    def code_of(d):
        for c in members.get(d, []):
            if c.startswith("OMIM:"):
                return c
        m = members.get(d, [])
        return m[0] if m else dis[d]

    names, cur = {}, None
    for line in open(str(cfg.path("obo")), encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = None
        elif line.startswith("id: HP:"):
            cur = line[4:].strip()
        elif line.startswith("name:") and cur:
            names[cur] = line[5:].strip()
    pn = lambda t: names.get(idx["phenotypes"][t], idx["phenotypes"][t])

    out_dir = cfg.path("results_dir")
    os.makedirs(out_dir, exist_ok=True)
    top_k = int(cfg.get("top_k", 20))
    rows, ranks = [], []
    with open(out_dir / "retrieval.jsonl", "w", encoding="utf-8") as f:
        for b, c in enumerate(cases):
            order = np.argsort(-S[:, b])
            rank = np.empty(idx["nD"], np.int64)
            rank[order] = np.arange(idx["nD"])
            gold_rank = int(min(rank[d] for d in c["gold"])) + 1
            ranks.append(gold_rank)
            rec = {"id": b,
                   "phenotypes": [pn(t) for t in c["phenotypes"]],
                   "gold": [{"rd": dis[d], "code": code_of(d),
                             "name": rd_name.get(dis[d], "")} for d in sorted(c["gold"])],
                   "gold_rank": gold_rank,
                   "topk": [{"rd": dis[int(d)], "code": code_of(int(d)),
                             "name": rd_name.get(dis[int(d)], ""),
                             "score": float(S[int(d), b]),
                             "phenos": [pn(int(t)) for t in
                                        C.indices[C.indptr[int(d)]:C.indptr[int(d) + 1]]],
                             "is_gold": bool(int(d) in c["gold"])} for d in order[:top_k]]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)
    m = metrics.score(ranks)
    metrics.table("retrieval", [("PPR", m)], n=len(rows))
    json.dump(m, open(out_dir / "retrieval.json", "w"), indent=2)
    print(f"[save] {out_dir}/retrieval.jsonl")
    return rows


def parse_order(text, candidates):
    code2i, names = {}, []
    for i, c in enumerate(candidates):
        for cc in (c["code"], c["rd"]):
            code2i.setdefault(str(cc).lower(), i)
        names.append((str(c["name"]).split("/")[0].strip().lower(), i))
    out, used = [], set()
    for line in (text or "").splitlines():
        hit = None
        for t in re.findall(r"[A-Za-z]+:\d+", line):
            j = code2i.get(t.lower())
            if j is not None and j not in used:
                hit = j
                break
        if hit is None:
            low = line.lower()
            for nm, j in names:
                if j not in used and nm and len(nm) > 6 and nm[:24] in low:
                    hit = j
                    break
        if hit is not None:
            out.append(hit)
            used.add(hit)
    return out


def report(cfg, rows, cache, top_k):
    import pandas as pd
    recs, after, before, malformed = [], [], [], 0
    line_re = re.compile(r"^\s*(\d{1,2})\s*[.)]\s+(.*)$")
    code_re = re.compile(r"\b[A-Za-z]+:\d+\b")
    for r in rows:
        cand = r["topk"][:top_k]
        txt = cache.get(str(r["id"]), "")
        order = parse_order(txt, cand)
        gold = {g["code"] for g in r["gold"]} | {g["rd"] for g in r["gold"]}
        hit = next((i + 1 for i, j in enumerate(order)
                    if cand[j]["code"] in gold or cand[j]["rd"] in gold), None)
        after.append(hit)
        before.append(r["gold_rank"] if r["gold_rank"] <= top_k else None)
        given = {str(c["code"]).lower() for c in cand} | {str(c["rd"]).lower() for c in cand}
        listed = [code_re.findall(m.group(2))[0].lower()
                  for m in map(line_re.match, txt.splitlines()) if m and code_re.findall(m.group(2))]
        if len(listed) < top_k or len(listed) != len(set(listed)) or any(c not in given for c in listed):
            malformed += 1
        recs.append({"id": r["id"],
                     "gold": " ; ".join(g["code"] for g in r["gold"]),
                     "gold_names": " ; ".join(g["name"] for g in r["gold"]),
                     "retrieval_rank": r["gold_rank"], "rerank_rank": hit or "",
                     "n_parsed": len(order),
                     "candidates": " | ".join(f"{c['code']} {c['name']}" for c in cand),
                     "ranking": " | ".join(f"{cand[j]['code']} {cand[j]['name']}" for j in order),
                     "response": txt})
    mb, ma = metrics.score(before, ks=KS), metrics.score(after, ks=KS)
    print(f"\nmalformed responses (short, repeated or invented code): {malformed} of {len(rows)}")
    metrics.table("rerank", [("retrieval", mb), ("+ LLM rerank", ma)], ks=KS, n=len(rows))
    out_dir = cfg.path("results_dir")
    pd.DataFrame(recs).to_csv(out_dir / "ranking.csv", index=False)
    json.dump({"model": cfg.get("rerank_model"), "top_k": top_k, "n": len(rows),
               "malformed": malformed, "retrieval": mb, "rerank": ma},
              open(out_dir / "rerank.json", "w"), indent=2)
    print(f"[save] {out_dir}/ranking.csv")


def main(cfg, args):
    out_dir = cfg.path("results_dir")
    top_k = int(cfg.get("top_k", 20))
    src = out_dir / "retrieval.jsonl"
    if (args.score or args.dry_run) and src.exists():
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        print(f"[data] {cfg.name}: {len(rows)} queries (from {src.name})")
    else:
        rows = retrieve(cfg)
    if args.limit:
        rows = rows[:args.limit]
    if not (args.rerank or args.score or args.dry_run):
        return

    inK = sum(1 for r in rows if r["gold_rank"] <= top_k)
    print(f"\nrerank: top-{top_k}, model {cfg.get('rerank_model')}. "
          f"Gold inside the top-{top_k}: {inK} ({100 * inK / len(rows):.2f}%)")

    def build(r):
        cands = [(c["code"], c["name"], c["phenos"]) for c in r["topk"][:top_k]]
        return prompts.rerank(r["phenotypes"], cands, top_k)

    if args.dry_run:
        _, p = build(rows[0])
        mean = float(np.mean([len(build(r)[1]) for r in rows]))
        cost, tin = llm.estimate(len(rows), mean, 500, cfg.get("rerank_model", "gpt-4o"))
        print("\n" + "=" * 78 + "\nSAMPLE PROMPT (truncated)\n" + "=" * 78)
        print(p[:1200] + f"\n   ... [{len(p)} chars]")
        print("=" * 78)
        print(f"mean prompt ~{tin:.0f} tokens -> ${cost:.2f} for {len(rows)} calls")
        return

    cache_path = str(out_dir / "responses.jsonl")
    if args.rerank:
        os.makedirs(out_dir, exist_ok=True)
        cache = llm.run_batch([(r["id"], r) for r in rows], build, cache_path,
                              cfg.get("rerank_model", "gpt-4o"), config_mod.api_key(),
                              max_tokens=2000, workers=args.workers)
    else:
        cache = llm.load_cache(cache_path)
    report(cfg, rows, cache, top_k)
