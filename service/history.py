from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class RecommendationHistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = Lock()

    def append(self, record: dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")

    def read(self, limit: int = 50, container_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = [line.strip() for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

        records = [json.loads(row) for row in rows]
        if container_id:
            records = [record for record in records if record.get("container_id") == container_id]
        if limit > 0:
            records = records[-limit:]
        records.reverse()
        return records

    def count(self) -> int:
        with self._lock:
            return sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())
