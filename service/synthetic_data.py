from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from service.config import DEFAULT_CONFIG


def generate_synthetic_data(
    meta_path: Path | None = None,
    usage_path: Path | None = None,
    n_containers: int = 500,
    seed: int = 42,
) -> dict[str, object]:
    meta_path = Path(meta_path or DEFAULT_CONFIG.meta_path)
    usage_path = Path(usage_path or DEFAULT_CONFIG.usage_path)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    usage_path.parent.mkdir(parents=True, exist_ok=True)

    random_gen = random.Random(seed)
    np.random.seed(seed)

    container_ids = [f"c_{idx}" for idx in range(1, n_containers + 1)]
    machine_ids = [f"m_{random_gen.randint(1, 100)}" for _ in range(n_containers)]

    meta_rows: list[list[object]] = []
    for idx in range(n_containers):
        app_name = f"app_{random_gen.randint(1000, 9999)}"
        cpu_request = random_gen.choice([400, 800, 1600])
        cpu_limit = cpu_request
        mem_size = random_gen.choice([1.56, 3.13, 6.26])

        for time_stamp in [0, 100000, 200000]:
            meta_rows.append(
                [
                    container_ids[idx],
                    machine_ids[idx],
                    time_stamp,
                    app_name,
                    "started",
                    cpu_request,
                    cpu_limit,
                    mem_size,
                ]
            )

    meta_frame = pd.DataFrame(meta_rows)
    meta_frame.to_csv(meta_path, index=False, header=False)

    usage_rows: list[list[object]] = []
    for idx in range(n_containers):
        base_cpu = random_gen.randint(5, 50)
        base_mem = random_gen.randint(20, 80)

        for time_stamp in range(0, 500000, 10000):
            container_id = np.nan if random_gen.random() < 0.05 else container_ids[idx]
            cpu_util = int(np.clip(np.random.normal(base_cpu, 15), 0, 100))
            mem_util = int(np.clip(np.random.normal(base_mem, 5), 0, 100))
            cpi = round(random_gen.uniform(1.0, 3.0), 2) if random_gen.random() > 0.5 else np.nan
            mem_gps = round(random_gen.uniform(0.0, 0.1), 2) if random_gen.random() > 0.5 else np.nan

            usage_rows.append(
                [
                    container_id,
                    machine_ids[idx],
                    time_stamp,
                    cpu_util,
                    mem_util,
                    cpi,
                    mem_gps,
                    0.0,
                    0.0,
                    0.0,
                    float(random_gen.randint(1, 10)),
                ]
            )

    usage_frame = pd.DataFrame(usage_rows)
    usage_frame.to_csv(usage_path, index=False, header=False)

    return {
        "meta_path": str(meta_path),
        "usage_path": str(usage_path),
        "meta_rows": int(len(meta_frame)),
        "usage_rows": int(len(usage_frame)),
    }
