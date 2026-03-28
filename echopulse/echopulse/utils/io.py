from __future__ import annotations

from pathlib import Path
import json


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
