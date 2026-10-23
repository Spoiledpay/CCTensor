"""Carregamento do config.json (parametros de modelo, dados e treino).

O config.json concentra os parametros; os scripts de treino o usam como
**defaults** e a linha de comando continua podendo sobrepor cada valor.

Estrutura do arquivo:
    {
      "model":       { ... },
      "data":        { ... },
      "generalista": { ... },
      "especialista":{ ... }
    }

Uso:
    from minigpt.config import load_config, section
    cfg = load_config(args.config)
    model_cfg = section(cfg, "model")
"""

import json
import os

DEFAULT_CONFIG = "config.json"


def load_config(path=DEFAULT_CONFIG):
    """Le o config.json. Retorna {} se o arquivo nao existir."""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def section(cfg, name):
    """Retorna a secao `name` do config (ou {})."""
    if isinstance(cfg, dict) and isinstance(cfg.get(name), dict):
        return cfg[name]
    return {}
