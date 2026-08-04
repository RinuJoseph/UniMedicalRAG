"""Ranking metrics"""
import numpy as np

KS = (1, 3, 5, 10, 20, 50, 100)


def score(ranks, ks=KS, miss=10 ** 9):
    r = np.array([x if x else miss for x in ranks], dtype=float)
    out = {f"Hit@{k}": round(100 * float(np.mean(r <= k)), 2) for k in ks}
    out["median_rank"] = float(np.median(r))
    out["MRR"] = round(float(np.mean(1.0 / r)), 4)
    return out


def table(title, rows, ks=KS, n=None):
    head = f"\n{title}" + (f"   n={n}" if n else "")
    print(head)
    print("  " + f"{'':<22}" + "".join(f"{'Hit@' + str(k):>9}" for k in ks) +
          f"{'median':>9}{'MRR':>9}")
    print("  " + "-" * (22 + 9 * len(ks) + 18))
    for label, m in rows:
        print("  " + f"{label:<22}" + "".join(f"{m[f'Hit@{k}']:>9.2f}" for k in ks) +
              f"{m['median_rank']:>9.0f}{m['MRR']:>9.4f}")
