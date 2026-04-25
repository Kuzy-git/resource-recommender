import os

from service.bootstrap import ensure_artifacts_ready


if __name__ == "__main__":
    force_retrain = os.getenv("FORCE_RETRAIN", "0").lower() in {"1", "true", "yes"}
    metadata = ensure_artifacts_ready(force=force_retrain)
    print(f"Artifacts ready. CPU model: {metadata['best_cpu_model_name']}, RAM model: {metadata['best_ram_model_name']}")
