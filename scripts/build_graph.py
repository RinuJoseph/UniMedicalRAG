"""Build the graph for a dataset

    python scripts/build_graph.py --config configs/rarebench.yaml
    python scripts/build_graph.py --config configs/medical.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as config_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if a cache exists")
    args = ap.parse_args()
    cfg = config_mod.load(args.config)

    if cfg.task == "kg_diagnosis":
        from src.kg import build as kg_build
        kg_build.configure(cfg)
        kg_build.ensure(rebuild=args.rebuild)
    elif cfg.task == "text_qa":
        from src.text import build_index
        build_index.build(cfg, rebuild=args.rebuild)
    else:
        raise SystemExit(f"unknown task: {cfg.task}")


if __name__ == "__main__":
    main()
