from __future__ import annotations

from typing import Any

from service.config import DEFAULT_CONFIG, HISTORY_PATH, PipelineConfig
from service.history import RecommendationHistoryStore
from service.ml_pipeline import (
    LoadedArtifacts,
    build_recommendation_response,
    dataframe_to_records,
    load_artifacts,
)


class RecommendationService:
    def __init__(self, artifacts: LoadedArtifacts, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.artifacts = artifacts
        self.config = config
        self.history_store = RecommendationHistoryStore(HISTORY_PATH)

    @classmethod
    def create(cls, config: PipelineConfig = DEFAULT_CONFIG) -> "RecommendationService":
        return cls(load_artifacts(), config=config)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service_name": self.artifacts.metadata["service_name"],
            "model_version": self.artifacts.metadata["model_version"],
            "trained_at": self.artifacts.metadata["trained_at"],
            "history_entries": self.history_store.count(),
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "service_name": self.artifacts.metadata["service_name"],
            "model_version": self.artifacts.metadata["model_version"],
            "trained_at": self.artifacts.metadata["trained_at"],
            "best_cpu_model_name": self.artifacts.best_cpu_model_name,
            "best_ram_model_name": self.artifacts.best_ram_model_name,
            "feature_cols": self.artifacts.feature_cols,
            "seq_cols": self.artifacts.seq_cols,
            "config": self.artifacts.metadata["config"],
            "metrics": self.artifacts.metadata["metrics"],
            "best_params": self.artifacts.metadata["best_params"],
            "training_split": self.artifacts.metadata["training_split"],
            "feature_influence": self.artifacts.metadata.get("feature_influence", {}),
        }

    def recommend(
        self,
        meta_payload: dict[str, Any],
        usage_payload: list[dict[str, Any]],
        include_features: bool,
        include_window_series: bool,
    ) -> dict[str, Any]:
        response = build_recommendation_response(
            artifacts=self.artifacts,
            meta_payload=meta_payload,
            usage_payload=usage_payload,
            include_features=include_features,
            include_window_series=include_window_series,
            config=self.config,
        )

        history_record = {
            "request_id": response["request_id"],
            "processed_at": response["processed_at"],
            "app_du": response["app_du"],
            "container_id": response["container_id"],
            "machine_id": response["machine_id"],
            "time_window": response["time_window"],
            "current_resources": response["current_resources"],
            "prediction": response["prediction"],
            "recommendation": response["recommendation"],
            "actions": response["actions"],
            "decision": response["decision"],
            "decision_label": response["decision_label"],
            "model_names": response["model_names"],
        }
        self.history_store.append(history_record)
        return response

    def history(self, limit: int = 50, container_id: str | None = None) -> dict[str, Any]:
        return {
            "items": self.history_store.read(limit=limit, container_id=container_id),
            "count": self.history_store.count(),
        }

    def data_overview(self, container_id: str | None = None, app_du: str | None = None, limit: int = 200) -> dict[str, Any]:
        window_df = self.artifacts.window_df.copy()
        meta_df = self.artifacts.meta_df.copy()
        if "app_du" not in window_df.columns:
            window_df["app_du"] = "unknown"

        window_df["cpu_usage_absolute"] = (window_df["cpu_util_mean"] / 100.0) * window_df["cpu_limit"]
        window_df["mem_usage_absolute"] = (window_df["mem_util_mean"] / 100.0) * window_df["mem_size"]

        app_container_counts = (
            meta_df[["container_id", "app_du"]]
            .drop_duplicates()
            .groupby("app_du")
            .size()
            .sort_values(ascending=False)
            .head(20)
            .reset_index(name="container_count")
        )

        machine_app_counts = (
            meta_df[["machine_id", "app_du"]]
            .drop_duplicates()
            .groupby("machine_id")["app_du"]
            .nunique()
            .sort_values(ascending=False)
            .head(20)
            .reset_index(name="app_count")
        )

        cluster_cpu_over_time = (
            window_df.groupby("time_window", as_index=False)
            .agg(
                cpu_request_total=("cpu_request", "sum"),
                cpu_limit_total=("cpu_limit", "sum"),
                cpu_usage_total=("cpu_usage_absolute", "sum"),
            )
            .sort_values("time_window")
            .tail(limit)
        )

        cluster_mem_over_time = (
            window_df.groupby("time_window", as_index=False)
            .agg(
                mem_size_total=("mem_size", "sum"),
                mem_usage_total=("mem_usage_absolute", "sum"),
            )
            .sort_values("time_window")
            .tail(limit)
        )

        resource_by_hour = (
            window_df.groupby("relative_hour", as_index=False)
            .agg(
                cpu_util_mean_avg=("cpu_util_mean", "mean"),
                mem_util_mean_avg=("mem_util_mean", "mean"),
            )
            .sort_values("relative_hour")
        )

        resource_by_day = (
            window_df.groupby("relative_day_cycle", as_index=False)
            .agg(
                cpu_util_mean_avg=("cpu_util_mean", "mean"),
                mem_util_mean_avg=("mem_util_mean", "mean"),
            )
            .sort_values("relative_day_cycle")
        )

        payload: dict[str, Any] = {
            "summary": {
                "window_minutes": self.config.horizon_minutes,
                "containers": int(meta_df[["app_du", "container_id"]].drop_duplicates().shape[0]),
                "machines": int(meta_df["machine_id"].nunique()),
                "bootstrap_recommendations": int(len(self.artifacts.bootstrap_report)),
            },
            "app_container_counts": dataframe_to_records(app_container_counts),
            "machine_app_counts": dataframe_to_records(machine_app_counts),
            "cluster_cpu_over_time": dataframe_to_records(cluster_cpu_over_time),
            "cluster_mem_over_time": dataframe_to_records(cluster_mem_over_time),
            "resource_by_hour": dataframe_to_records(resource_by_hour),
            "resource_by_day": dataframe_to_records(resource_by_day),
            "bootstrap_report_preview": dataframe_to_records(self.artifacts.bootstrap_report.head(20)),
        }

        if container_id:
            container_frame = window_df[window_df["container_id"] == container_id].copy()
            if app_du:
                container_frame = container_frame[container_frame["app_du"] == app_du].copy()
            container_frame = container_frame.sort_values(["app_du", "container_id", "time_window"]).tail(limit).copy()
            if container_frame.empty:
                raise ValueError(f"Контейнер '{container_id}' не найден в базовом наборе данных.")

            payload["container_series"] = dataframe_to_records(
                container_frame[
                    [
                        "app_du",
                        "container_id",
                        "machine_id",
                        "time_window",
                        "cpu_request",
                        "cpu_limit",
                        "cpu_util_mean",
                        "cpu_util_max",
                        "mem_size",
                        "mem_util_mean",
                        "mem_util_max",
                        "samples_in_window",
                    ]
                ]
            )

        return payload
