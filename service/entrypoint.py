from __future__ import annotations

import os

import uvicorn

from service.bootstrap import ensure_artifacts_ready


def main() -> None:
    force_retrain = os.getenv("FORCE_RETRAIN", "0").lower() in {"1", "true", "yes"}
    host = os.getenv("SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("SERVICE_PORT", "8000"))

    ensure_artifacts_ready(force=force_retrain)
    print(f"[service] starting FastAPI on http://{host}:{port}")
    uvicorn.run("service.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
