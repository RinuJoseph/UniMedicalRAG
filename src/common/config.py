
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Config(dict):

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def path(self, key, default=None):
        v = self.get(key, default)
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else ROOT / p


def load(config_file, **overrides):
    p = Path(config_file)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"config not found: {p}")
    cfg = Config(yaml.safe_load(p.read_text()) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["_config_file"] = str(p)
    if "task" not in cfg:
        raise SystemExit(f"{p} is missing a `task:` field (kg_diagnosis or text_qa)")
    return cfg


def api_key():
    k = os.environ.get("OPENAI_API_KEY")
    if not k:
        raise SystemExit("set OPENAI_API_KEY in your environment")
    return k
