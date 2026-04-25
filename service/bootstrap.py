from __future__ import annotations

import json
import os
import csv
from pathlib import Path

from service.config import (
    DEFAULT_CONFIG,
    META_CLEAN_PATH,
    METADATA_PATH,
    REPORT_DATA_PATH,
    SEQUENCE_SCALER_PATH,
    TABULAR_SCALER_PATH,
    WINDOW_DATA_PATH,
    PipelineConfig,
)
from service.ml_pipeline import train_and_save_artifacts
from service.synthetic_data import generate_synthetic_data


def _csv_has_columns(path: Path, columns: set[str]) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except OSError:
        return False

    return columns.issubset(set(header))


def artifacts_are_ready() -> bool:
    if not METADATA_PATH.exists():
        return False

    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    if (
        "feature_influence" not in metadata
        or metadata.get("feature_influence_version") != 2
        or metadata.get("entity_cols") != ["app_du", "container_id"]
    ):
        return False

    required_paths = [
        META_CLEAN_PATH,
        WINDOW_DATA_PATH,
        REPORT_DATA_PATH,
        TABULAR_SCALER_PATH,
        SEQUENCE_SCALER_PATH,
        Path(SEQUENCE_SCALER_PATH.parent / metadata["cpu_model_file"]),
        Path(SEQUENCE_SCALER_PATH.parent / metadata["ram_model_file"]),
    ]
    if not all(path.exists() for path in required_paths):
        return False

    return _csv_has_columns(WINDOW_DATA_PATH, {"app_du", "container_id"}) and _csv_has_columns(
        REPORT_DATA_PATH,
        {"app_du", "container_id"},
    )


def ensure_artifacts_ready(config: PipelineConfig = DEFAULT_CONFIG, force: bool = False) -> dict[str, object]:
    if not config.meta_path.exists() or not config.usage_path.exists():
        print("[bootstrap] input CSV files not found. Generating synthetic dataset...")
        generate_synthetic_data(meta_path=config.meta_path, usage_path=config.usage_path, seed=config.random_seed)

    if force or not artifacts_are_ready():
        if force:
            print("[bootstrap] force retrain enabled. Rebuilding model artifacts from current CSV files...")
        else:
            print("[bootstrap] artifacts not found. Starting model training...")
        return train_and_save_artifacts(config)

    print("[bootstrap] existing artifacts found. Reusing saved models.")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    force_retrain = os.getenv("FORCE_RETRAIN", "0").lower() in {"1", "true", "yes"}
    metadata = ensure_artifacts_ready(force=force_retrain)
    print(f"Artifacts ready. CPU model: {metadata['best_cpu_model_name']}, RAM model: {metadata['best_ram_model_name']}")
