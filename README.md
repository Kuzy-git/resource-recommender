# Resource Recommender Service

Сервис на `FastAPI` для прогноза нагрузки контейнера и выдачи рекомендаций по CPU и RAM.

## Что делает сервис

- принимает метрики контейнера
- агрегирует их в окна
- формирует признаки
- подаёт признаки в обученные модели
- возвращает прогноз и рекомендацию: `UPSCALE`, `DOWNSCALE` или `OK`

Модель обучается один раз и сохраняется в `artifacts/`. При следующих запусках сервис использует сохранённые артефакты. Если `container_meta.csv` и `container_usage.csv` отсутствуют, будет сгенерирован синтетический датасет.

## Эндпоинты

- `GET /health`
- `POST /recommendation`
- `GET /model-info`
- `GET /recommendations/history`
- `GET /data`

## Запуск без Docker

Из корня проекта:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m service.entrypoint
```

После запуска:

```text
http://127.0.0.1:8000
```

Документация Swagger:

```text
http://127.0.0.1:8000/docs
```

Графики и таблицы:

```text
http://127.0.0.1:8000/data
```

Графики по одному контейнеру:

```text
http://127.0.0.1:8000/data?container_id=c_1
```

JSON-ответ вместо HTML:

```text
http://127.0.0.1:8000/data?format=json
```

## Запуск через Docker

Собрать образ:

```powershell
docker build -t resource-recommender .
```

Запустить контейнер:

```powershell
docker run --rm -p 8000:8000 resource-recommender
```

Запустить с принудительным переобучением:

```powershell
docker run --rm -p 8000:8000 -e FORCE_RETRAIN=1 resource-recommender
```

## Полезные команды

Принудительно переобучить модели без запуска API:

```powershell
$env:FORCE_RETRAIN=1
.\.venv\Scripts\python train_models.py
```

Подготовить артефакты без запуска API:

```powershell
.\.venv\Scripts\python train_models.py
```

Сгенерировать синтетические данные вручную:

```powershell
.\.venv\Scripts\python syntethis_data.py
```

Проверить, что сервис отвечает:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## История рекомендаций

История хранится локально в файле:

```text
artifacts/recommendations_history.jsonl
```

Лимит `50` в `GET /recommendations/history` ограничивает только выдачу ответа. Старые записи не удаляются автоматически.

## Пример запроса в `/recommendation`

```json
{
  "meta": {
    "container_id": "demo-container",
    "machine_id": "demo-machine",
    "app_du": "demo-app",
    "status": "started",
    "cpu_request": 400,
    "cpu_limit": 800,
    "mem_size": 3.13
  },
  "usage": [
    {"time_stamp": 0, "cpu_util_percent": 20, "mem_util_percent": 45},
    {"time_stamp": 600, "cpu_util_percent": 25, "mem_util_percent": 47},
    {"time_stamp": 1200, "cpu_util_percent": 22, "mem_util_percent": 50},
    {"time_stamp": 1800, "cpu_util_percent": 30, "mem_util_percent": 52},
    {"time_stamp": 2400, "cpu_util_percent": 28, "mem_util_percent": 49},
    {"time_stamp": 3000, "cpu_util_percent": 35, "mem_util_percent": 53},
    {"time_stamp": 3600, "cpu_util_percent": 40, "mem_util_percent": 55},
    {"time_stamp": 4200, "cpu_util_percent": 38, "mem_util_percent": 58}
  ],
  "include_features": true,
  "include_window_series": true
}
```
