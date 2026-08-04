
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as config_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rerank", action="store_true", help="run the LLM reranker")
    ap.add_argument("--score", action="store_true", help="recompute from saved responses")
    ap.add_argument("--dry-run", action="store_true", help="prompts and cost only, no calls")
    ap.add_argument("--limit", type=int, default=0, help="first N queries only")
    ap.add_argument("--top_k", type=int)
    ap.add_argument("--damping", type=float)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = config_mod.load(args.config, top_k=args.top_k, damping=args.damping)

    if cfg.task == "kg_diagnosis":
        from src.kg import pipeline
        pipeline.main(cfg, args)
    elif cfg.task == "text_qa":
        from src.text import pipeline
        pipeline.main(cfg, args)
    else:
        raise SystemExit(f"unknown task: {cfg.task}")


if __name__ == "__main__":
    main()
