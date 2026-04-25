from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
WINDOW_DATA_PATH = ARTIFACTS_DIR / "window_df.csv"
REPORT_DATA_PATH = ARTIFACTS_DIR / "bootstrap_report.csv"
META_CLEAN_PATH = ARTIFACTS_DIR / "meta_clean.csv"
HISTORY_PATH = ARTIFACTS_DIR / "recommendations_history.jsonl"
TABULAR_SCALER_PATH = MODELS_DIR / "tabular_scaler.joblib"
SEQUENCE_SCALER_PATH = MODELS_DIR / "sequence_scaler.joblib"


@dataclass(frozen=True)
class PipelineConfig:
    meta_path: Path = ROOT_DIR / "container_meta.csv"
    usage_path: Path = ROOT_DIR / "container_usage.csv"
    meta_nrows: int = 500000
    usage_nrows: int = 1000000
    lags: int = 3
    lookback: int = 5
    grid_sample_size: int = 50000
    training_n_jobs: int = 1
    horizon_minutes: int = 10
    cpu_safety_margin: float = 1.20
    ram_safety_margin: float = 1.15
    down_threshold: float = 0.85
    up_threshold: float = 1.05
    model_version: str = "1.0.0"
    random_seed: int = 42
    service_name: str = "resource-recommender-service"

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in data.items()
        }


DEFAULT_CONFIG = PipelineConfig()
