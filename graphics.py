import matplotlib.pyplot as plt
window_df["cpu_usage_absolute"] = (window_df["cpu_util_mean"] / 100.0) * window_df["cpu_limit"]
window_df["mem_usage_absolute"] = (window_df["mem_util_mean"] / 100.0) * window_df["mem_size"]
window_df["time_window_hours"] = window_df["time_window"] / 3600.0
meta_app_unique = meta[["container_id", "app_du"]].drop_duplicates().copy()

app_container_counts = (
    meta_app_unique
    .groupby("app_du")["container_id"]
    .nunique()
    .sort_values(ascending=False)
)

max_categories_to_show = 20

if len(app_container_counts) <= max_categories_to_show:
    app_plot_counts = app_container_counts.copy()
    app_title_suffix = "все app_du"
else:
    app_plot_counts = app_container_counts.head(max_categories_to_show).copy()
    app_title_suffix = f"топ-{max_categories_to_show} app_du"

plt.figure(figsize=(14, 6))
plt.bar(app_plot_counts.index.astype(str), app_plot_counts.values)
plt.title(f"Количество контейнеров по app_du ({app_title_suffix})")
plt.xlabel("app_du")
plt.ylabel("Количество уникальных container_id")
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()
meta_machine_app_unique = meta[["machine_id", "app_du"]].drop_duplicates().copy()

machine_app_counts = (
    meta_machine_app_unique
    .groupby("machine_id")["app_du"]
    .nunique()
    .sort_values(ascending=False)
)

if len(machine_app_counts) <= max_categories_to_show:
    machine_plot_counts = machine_app_counts.copy()
    machine_title_suffix = "все machine_id"
else:
    machine_plot_counts = machine_app_counts.head(max_categories_to_show).copy()
    machine_title_suffix = f"топ-{max_categories_to_show} machine_id"

plt.figure(figsize=(14, 6))
plt.bar(machine_plot_counts.index.astype(str), machine_plot_counts.values)
plt.title(f"Количество уникальных app_du на виртуальной машине ({machine_title_suffix})")
plt.xlabel("machine_id")
plt.ylabel("Количество уникальных app_du")
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()
cluster_cpu_over_time = (
    window_df
    .groupby("time_window_hours", as_index=False)
    .agg(
        cpu_request_total=("cpu_request", "sum"),
        cpu_limit_total=("cpu_limit", "sum"),
        cpu_usage_total=("cpu_usage_absolute", "sum")
    )
    .sort_values("time_window_hours")
)

plt.figure(figsize=(14, 6))
plt.plot(cluster_cpu_over_time["time_window_hours"], cluster_cpu_over_time["cpu_request_total"], label="CPU Request")
plt.plot(cluster_cpu_over_time["time_window_hours"], cluster_cpu_over_time["cpu_limit_total"], label="CPU Limit")
plt.plot(cluster_cpu_over_time["time_window_hours"], cluster_cpu_over_time["cpu_usage_total"], label="CPU Usage")
plt.title("CPU request / limit / usage от времени")
plt.xlabel("Время от начала trace, часы")
plt.ylabel("CPU")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
cluster_mem_over_time = (
    window_df
    .groupby("time_window_hours", as_index=False)
    .agg(
        mem_size_total=("mem_size", "sum"),
        mem_usage_total=("mem_usage_absolute", "sum")
    )
    .sort_values("time_window_hours")
)

plt.figure(figsize=(14, 6))
plt.plot(cluster_mem_over_time["time_window_hours"], cluster_mem_over_time["mem_size_total"], label="Mem Size")
plt.plot(cluster_mem_over_time["time_window_hours"], cluster_mem_over_time["mem_usage_total"], label="Mem Usage")
plt.title("Mem size / usage от времени")
plt.xlabel("Время от начала trace, часы")
plt.ylabel("Memory")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
resource_by_hour = (
    window_df
    .groupby("relative_hour", as_index=False)
    .agg(
        cpu_util_mean_avg=("cpu_util_mean", "mean"),
        mem_util_mean_avg=("mem_util_mean", "mean")
    )
    .sort_values("relative_hour")
)

plt.figure(figsize=(12, 5))
plt.plot(resource_by_hour["relative_hour"], resource_by_hour["cpu_util_mean_avg"], marker="o", label="CPU mean usage, %")
plt.plot(resource_by_hour["relative_hour"], resource_by_hour["mem_util_mean_avg"], marker="o", label="RAM mean usage, %")
plt.title("Среднее потребление CPU и RAM по времени суток")
plt.xlabel("Час суток")
plt.ylabel("Средняя загрузка, %")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
resource_by_day = (
    window_df
    .groupby("relative_day_cycle", as_index=False)
    .agg(
        cpu_util_mean_avg=("cpu_util_mean", "mean"),
        mem_util_mean_avg=("mem_util_mean", "mean")
    )
    .sort_values("relative_day_cycle")
)

plt.figure(figsize=(10, 5))
plt.plot(resource_by_day["relative_day_cycle"], resource_by_day["cpu_util_mean_avg"], marker="o", label="CPU mean usage, %")
plt.plot(resource_by_day["relative_day_cycle"], resource_by_day["mem_util_mean_avg"], marker="o", label="RAM mean usage, %")
plt.title("Среднее потребление CPU и RAM по дню цикла")
plt.xlabel("День цикла")
plt.ylabel("Средняя загрузка, %")
plt.xticks(range(0, 7))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()