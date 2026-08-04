
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as config_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--topn", type=int, help="predictions shown to the judge")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = config_mod.load(args.config, judge_topn=args.topn)

    if cfg.task == "kg_diagnosis":
        from src.kg import evaluate
        evaluate.main(cfg, args)
    elif cfg.task == "text_qa":
        from src.text import evaluate
        evaluate.main(cfg, args)
    else:
        raise SystemExit(f"unknown task: {cfg.task}")


if __name__ == "__main__":
    main()
