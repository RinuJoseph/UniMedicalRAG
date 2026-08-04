"""Offline graph build for the phenotype knowledge-graph task."""
import csv
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

PROOT = "HP:0000118"          # keep only the phenotypic-abnormality subtree
_P = {}                       # paths, filled in by configure()


def configure(cfg):
    """Point the builder at the directories named in the config."""
    _P["RB"] = str(cfg.path("nodes_dir"))
    _P["HPOA"] = str(cfg.path("hpoa"))
    _P["OBO"] = str(cfg.path("obo"))
    _P["INDEX"] = str(cfg.path("index_dir"))


def _rows(path):
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for row in r:
            if row:
                yield row


def _obo_alt(path):
    alt, cur = {}, None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("id: HP:"):
            cur = line[4:]
        elif line.startswith("alt_id: HP:") and cur:
            alt[line[8:]] = cur
    return alt


def _ancestors(H, nP):
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


def build():
    # --- nodes ---
    phenos_all = [row[0] for row in _rows(os.path.join(_P['RB'], "nodes_phenotype.tsv"))]
    pia = {h: i for i, h in enumerate(phenos_all)}
    nPa = len(phenos_all)
    diseases, di, rd_name = [], {}, {}
    for i, row in enumerate(_rows(os.path.join(_P['RB'], "nodes_disease.tsv"))):
        diseases.append(row[0]); rd_name[row[0]] = row[2] if len(row) > 2 else ""
        for c in row[1].split("|"):
            di[c.strip()] = i
    nD = len(diseases)

    # --- C (disease->phenotype), phenotype.hpoa aspect-P, member codes merged onto RD ---
    r, c = [], []
    for line in open(_P['HPOA'], encoding="utf-8"):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[0] == "database_id" or f[2] == "NOT" or f[10] != "P":
            continue
        if f[0] in di and f[3] in pia:
            r.append(di[f[0]]); c.append(pia[f[3]])
    Ca = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nD, nPa)); Ca.sum_duplicates(); Ca.data[:] = 1.0

    # --- H (is_a, child->parent) over all HP ---
    r, c = [], []
    for row in _rows(os.path.join(_P['RB'], "edges_hpo_isa.tsv")):
        if row[0] in pia and row[1] in pia:
            r.append(pia[row[0]]); c.append(pia[row[1]])
    Ha = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nPa, nPa))

    # --- restrict to the Phenotypic-abnormality subtree ---
    anc = _ancestors(Ha.tocsr(), nPa)
    p118 = pia[PROOT]
    Pidx = [t for t in range(nPa) if t == p118 or p118 in anc[t]]
    Pset = set(Pidx); o2n = {t: i for i, t in enumerate(Pidx)}
    phenotypes = [phenos_all[t] for t in Pidx]; nP = len(Pidx)

    Cc = Ca.tocoo(); r, c = [], []
    for d, t in zip(Cc.row.tolist(), Cc.col.tolist()):
        if t in Pset:
            r.append(d); c.append(o2n[t])
    C = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nD, nP)); C.sum_duplicates(); C.data[:] = 1.0

    Hc = Ha.tocoo(); r, c = [], []
    for a, b in zip(Hc.row.tolist(), Hc.col.tolist()):
        if a in Pset and b in Pset:
            r.append(o2n[a]); c.append(o2n[b])
    H = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nP, nP))

    # --- IC: n_t = #diseases with the term or a descendant (ancestor-closed) ---
    ancn = _ancestors(H.tocsr(), nP)
    Cr = C.tocsr(); r, c = [], []
    for d in range(nD):
        closed = set()
        for t in Cr.indices[Cr.indptr[d]:Cr.indptr[d + 1]]:
            closed.add(int(t)); closed |= ancn[int(t)]
        for t in closed:
            r.append(d); c.append(t)
    C_closed = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nD, nP))
    n_t = np.asarray(C_closed.sum(0)).ravel()
    # add-one (Laplace) smoothing over ALL terms; n_t = 0 -> ln(nD+1)
    IC = (-np.log((n_t + 1.0) / (nD + 1.0))).astype(np.float32)

    os.makedirs(_P['INDEX'], exist_ok=True)
    sp.save_npz(os.path.join(_P['INDEX'], "C.npz"), C)
    sp.save_npz(os.path.join(_P['INDEX'], "H.npz"), H)
    np.save(os.path.join(_P['INDEX'], "IC.npy"), IC)
    json.dump({"diseases": diseases, "phenotypes": phenotypes, "di": di, "alt": _obo_alt(_P["OBO"]),
               "rd_name": rd_name, "note": "P-only (HP:0000118 subtree); add-one IC"},
              open(os.path.join(_P['INDEX'], "meta.json"), "w"))
    print(f"[index] built {nD} RD x {nP} P-phenotypes  ({int(C.nnz)} disease-phenotype edges, "
          f"{int(H.nnz)} is_a; add-one IC)  -> {_P[chr(34)+chr(34)]}")


def load():
    m = json.load(open(os.path.join(_P['INDEX'], "meta.json")))
    return dict(diseases=m["diseases"], phenotypes=m["phenotypes"], di=m["di"], alt=m["alt"],
                rd_name=m.get("rd_name", {}), pi={p: i for i, p in enumerate(m["phenotypes"])},
                C=sp.load_npz(os.path.join(_P['INDEX'], "C.npz")), H=sp.load_npz(os.path.join(_P['INDEX'], "H.npz")),
                IC=np.load(os.path.join(_P['INDEX'], "IC.npy")), nD=len(m["diseases"]), nP=len(m["phenotypes"]))


def ensure(rebuild=False):
    """Load the cached graph, rebuilding it if missing or if it was built by an older
    version that used a different IC."""
    meta = os.path.join(_P['INDEX'], "meta.json")
    stale = os.path.exists(meta) and "add-one IC" not in json.load(open(meta)).get("note", "")
    if stale:
        print("[index] cache was not built with add-one IC -- rebuilding")
    if rebuild or stale or not os.path.exists(meta):
        build()
    return load()


