from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"


@dataclass(frozen=True)
class ExperimentConfig:
    quantile_levels: list[float]
    data_size_days: list[int | str]
    seeds: list[int]
    data_size_sweep_seeds: list[int]
    source_path: Path


@dataclass(frozen=True)
class DataConfig:
    euppbench_version: str
    targets: list[str]
    max_lead_hours: int
    split_name: str
    source_path: Path


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    resolved_path = path or DEFAULT_EXPERIMENT_CONFIG
    raw = yaml.safe_load(resolved_path.read_text())
    return ExperimentConfig(
        quantile_levels=raw["quantile_levels"],
        data_size_days=raw["data_size_days"],
        seeds=raw["seeds"],
        data_size_sweep_seeds=raw["data_size_sweep_seeds"],
        source_path=Path(resolved_path),
    )


def load_data_config(path: Path | None = None) -> DataConfig:
    resolved_path = path or DEFAULT_DATA_CONFIG
    raw = yaml.safe_load(resolved_path.read_text())
    return DataConfig(
        euppbench_version=raw["euppbench_version"],
        targets=raw["targets"],
        max_lead_hours=raw["max_lead_hours"],
        split_name=raw["split_name"],
        source_path=Path(resolved_path),
    )
