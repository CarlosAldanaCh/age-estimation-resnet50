# src/config.py

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    raw: dict

    @property
    def seed(self) -> int:
        return int(self.raw["proyect"]["seed"])

    @property
    def csv_path(self) -> Path:
        return Path(self.raw["proyect"]["csv_path"])

    @property
    def images_dir(self) -> Path:
        return Path(self.raw["proyect"]["images_dir"])


def load_config(path: str | Path) -> Config:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(raw=raw)
