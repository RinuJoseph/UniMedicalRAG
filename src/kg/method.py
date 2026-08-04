
import json
import os

import numpy as np
import scipy.sparse as sp


CASES = None            
SPLITS = []
DAMPING = 0.15          
PRUNE_K = 20           

def configure(cfg):
    global CASES, SPLITS, DAMPING, PRUNE_K
    CASES = str(cfg.path("cases_dir"))
    SPLITS = list(cfg.get("splits", []))
    DAMPING = float(cfg.get("damping", DAMPING))
    PRUNE_K = cfg.get("prune_k", PRUNE_K)


def load_cases(idx):
    di, pi, alt = idx["di"], idx["pi"], idx["alt"]
    cases = []
    for split in SPLITS:
        fp = os.path.join(CASES, f"{split}.jsonl")
        if not os.path.exists(fp):
            continue
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # alt-id resolve, then keep only codes in the P vocabulary (drops non-P findings)
            phe = sorted({pi[h] for h in (alt.get(x, x) for x in r["Phenotype"]) if h in pi})
            gold = {di[d] for d in r["RareDisease"] if d in di}
            if phe and gold:
                cases.append(dict(split=split, phenotypes=phe, gold=gold))
    return cases


def ancestors(idx):
    H = idx["H"].tocsr(); nP = idx["nP"]
    anc = [None] * nP
    for s in range(nP):
        if anc[s] is not None:
            continue
        seen, st = set(), [s]
        while st:
            x = st.pop()
            for p in H.indices[H.indptr[x]:H.indptr[x + 1]].tolist():
                if p not in seen:
                    seen.add(p); st.append(p)
        anc[s] = seen
    return anc


def seed_matrix(idx, cases, anc):
    C = idx["C"].tocsr(); IC = idx["IC"].astype(np.float32)
    nD, nP, B = idx["nD"], idx["nP"], len(cases)
    Rp = sp.lil_matrix((nP, B), dtype=np.float32)   # phenotype restart mass
    Wd = sp.lil_matrix((nP, B), dtype=np.float32)   # findings only, to warm-start diseases
    for b, c in enumerate(cases):
        reported = set(c["phenotypes"])
        own = {}
        for t in c["phenotypes"]:
            Rp[t, b] += IC[t]; Wd[t, b] += IC[t]
            for a in anc[t]:
                a = int(a)
                if a not in reported:      # ONE weight per node, never two: a term that is
                    own[a] = IC[a]         # both reported and an ancestor keeps only its
        for a, v in own.items():           # direct-match weight
            Rp[a, b] += v
    W = np.zeros((nD + nP, B), np.float32)
    W[nD:] = np.asarray(Rp.todense())
    W[:nD] = np.asarray((C @ Wd.tocsr()).todense())     # disease warm-start = shared-phenotype IC
    col = W.sum(0); dead = col <= 0
    if dead.any():
        W[:, dead] = 1.0 / (nD + nP); col = W.sum(0)
    return (W / col).astype(np.float32)


def _softmax_x_n(scores):
    z = np.asarray(scores, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum() * len(z)


def graph(idx, prune_k=None):
    import igraph as ig
    prune_k = PRUNE_K if prune_k is None else prune_k
    C, H = idx["C"].tocsr(), idx["H"].tocsr()
    Hc = H.tocsc()
    IC = idx["IC"]
    nD, nP = idx["nD"], idx["nP"]

    E, wt = [], []
    for d in range(nD):
        ts = C.indices[C.indptr[d]:C.indptr[d + 1]]
        if ts.size == 0:
            continue
        for t, w in zip(ts.tolist(), _softmax_x_n(IC[ts]).tolist()):
            E.append((d, nD + t)); wt.append(w)
    for p in range(nP):
        kids = Hc.indices[Hc.indptr[p]:Hc.indptr[p + 1]]
        if kids.size == 0:
            continue
        for k, w in zip(kids.tolist(), _softmax_x_n(IC[kids]).tolist()):
            E.append((nD + p, nD + k)); wt.append(w)

    if prune_k is not None:
        inc = {}
        for i, (u, v) in enumerate(E):
            inc.setdefault(u, []).append(i)
            inc.setdefault(v, []).append(i)
        wa = np.array(wt)
        keep = np.zeros(len(E), bool)
        for ids in inc.values():
            if len(ids) <= prune_k:
                keep[ids] = True
            else:
                a = np.array(ids)
                keep[a[np.argpartition(-wa[a], prune_k - 1)[:prune_k]]] = True
        E = [e for i, e in enumerate(E) if keep[i]]
    return ig.Graph(n=nD + nP, edges=E, directed=False)   # unweighted


def ppr(idx, W, G=None):
    G = graph(idx) if G is None else G
    nD, B = idx["nD"], W.shape[1]
    S = np.zeros((nD, B), np.float32)
    for b in range(B):
        pr = G.personalized_pagerank(damping=DAMPING,
                                     reset=W[:, b].astype(np.float64).tolist())
        S[:, b] = np.asarray(pr[:nD], np.float32)
    return S


def run(idx, cases):
    return ppr(idx, seed_matrix(idx, cases, ancestors(idx)))
