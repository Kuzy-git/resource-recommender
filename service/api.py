from __future__ import annotations

import json
from contextlib import asynccontextmanager
from html import escape
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from service.recommendation_service import RecommendationService
from service.schemas import RecommendationRequest


STYLE_BLOCK = """
<style>
  :root {
    color-scheme: light;
    --bg: #f2f5f3;
    --surface: #ffffff;
    --surface-alt: #e8efe9;
    --text: #15201b;
    --muted: #55635b;
    --line: #d2dbd5;
    --accent: #1f7a5c;
    --accent-strong: #11533d;
    --accent-soft: #dff1e8;
    --warm: #9d5a1c;
    --shadow: 0 14px 38px rgba(17, 34, 27, 0.08);
  }

  * {
    box-sizing: border-box;
  }

  html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    line-height: 1.5;
  }

  body {
    min-height: 100vh;
  }

  a {
    color: var(--accent-strong);
    text-decoration: none;
  }

  a:hover {
    text-decoration: underline;
  }

  code, pre {
    font-family: Consolas, "SFMono-Regular", Menlo, monospace;
  }

  .page {
    max-width: 1180px;
    margin: 0 auto;
    padding: 22px 18px 40px;
  }

  .topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 18px;
  }

  .brand {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
  }

  .brand-badge {
    display: inline-flex;
    align-items: center;
    min-height: 30px;
    padding: 0 10px;
    border-radius: 8px;
    background: var(--accent-soft);
    color: var(--accent-strong);
    font-size: 12px;
    font-weight: 700;
  }

  .brand-name {
    font-size: 15px;
    color: var(--muted);
  }

  .nav {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .nav a {
    display: inline-flex;
    align-items: center;
    min-height: 38px;
    padding: 0 12px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: var(--surface);
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
  }

  .nav a.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }

  .hero {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: var(--shadow);
    padding: 28px;
    margin-bottom: 18px;
  }

  .eyebrow {
    display: inline-block;
    margin-bottom: 12px;
    padding: 6px 10px;
    border-radius: 8px;
    background: var(--accent-soft);
    color: var(--accent-strong);
    font-size: 12px;
    font-weight: 700;
  }

  h1 {
    margin: 0 0 10px;
    font-size: 34px;
    line-height: 1.08;
  }

  h2 {
    margin: 0 0 12px;
    font-size: 22px;
    line-height: 1.15;
  }

  p {
    margin: 0 0 12px;
    color: var(--muted);
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
  }

  .button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 42px;
    padding: 0 14px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: var(--surface-alt);
    font-weight: 600;
    color: var(--text);
  }

  .button.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 18px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: var(--shadow);
    padding: 18px;
  }

  .card-label {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 8px;
  }

  .card-value {
    font-size: 28px;
    line-height: 1.1;
    font-weight: 700;
    word-break: break-word;
  }

  .section {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: var(--shadow);
    padding: 20px;
    margin-bottom: 18px;
  }

  .meta {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }

  .meta-item {
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: var(--surface-alt);
  }

  .meta-item strong {
    display: block;
    margin-bottom: 6px;
    font-size: 13px;
  }

  .pill-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .pill {
    display: inline-flex;
    align-items: center;
    min-height: 34px;
    padding: 0 10px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: var(--surface-alt);
    font-size: 13px;
    color: var(--text);
  }

  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--line);
    border-radius: 8px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    min-width: 640px;
    background: var(--surface);
  }

  th, td {
    padding: 11px 12px;
    text-align: left;
    border-bottom: 1px solid var(--line);
    vertical-align: top;
    font-size: 14px;
  }

  th {
    background: var(--surface-alt);
    color: var(--text);
    font-weight: 700;
  }

  tr:last-child td {
    border-bottom: none;
  }

  .console {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #16211b;
    color: #eef5f0;
    padding: 16px;
    overflow-x: auto;
  }

  .empty {
    padding: 14px;
    border-radius: 8px;
    background: var(--surface-alt);
    color: var(--muted);
    border: 1px dashed var(--line);
  }

  .two-col {
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    gap: 18px;
  }

  .note {
    color: var(--muted);
    font-size: 14px;
  }

  h3 {
    margin: 0 0 8px;
    font-size: 18px;
    line-height: 1.2;
  }

  .section-header {
    margin-bottom: 14px;
  }

  .chart-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
    margin-bottom: 18px;
  }

  .chart-card {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .chart-frame {
    width: 100%;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: linear-gradient(180deg, #fbfdfc 0%, #f4f8f5 100%);
  }

  .chart-canvas {
    display: block;
    width: 100%;
    height: 290px;
  }

  .chart-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--muted);
    font-size: 13px;
  }

  .legend-swatch {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    flex: 0 0 auto;
  }

  @media (max-width: 960px) {
    .grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .chart-grid,
    .two-col,
    .meta {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 640px) {
    .page {
      padding: 16px 12px 28px;
    }

    h1 {
      font-size: 28px;
    }

    .grid {
      grid-template-columns: 1fr;
    }

    .hero,
    .section,
    .card {
      padding: 16px;
    }
  }
</style>
"""


SCRIPT_BLOCK = """
<script>
(() => {
  const palette = ["#1f7a5c", "#c46c2a", "#2f6f8f", "#9d5a1c", "#7a55b6", "#c04d7b"];

  function toNumber(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : 0;
  }

  function formatTick(value) {
    if (!Number.isFinite(value)) {
      return "";
    }

    const abs = Math.abs(value);
    if (abs >= 1000000) {
      return `${(value / 1000000).toFixed(1).replace(/\\.0$/, "")}M`;
    }
    if (abs >= 1000) {
      return `${(value / 1000).toFixed(abs >= 10000 ? 0 : 1).replace(/\\.0$/, "")}K`;
    }
    return String(Math.round(value * 100) / 100);
  }

  function truncateLabel(value, size = 10) {
    const text = String(value ?? "");
    return text.length > size ? `${text.slice(0, size - 1)}…` : text;
  }

  function getMaxValue(series) {
    const values = [];
    series.forEach((item) => {
      (item.values || []).forEach((value) => values.push(toNumber(value)));
    });
    const max = Math.max(...values, 0);
    return max > 0 ? max * 1.1 : 1;
  }

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(280, Math.floor(rect.width));
    const height = Math.max(240, Number(canvas.dataset.chartHeight || 290));
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
  }

  function drawGrid(ctx, width, height, padding, maxY) {
    const ticks = 4;
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    ctx.font = '12px Inter, "Segoe UI", Arial, sans-serif';
    ctx.strokeStyle = "rgba(21, 32, 27, 0.10)";
    ctx.fillStyle = "#55635b";
    ctx.lineWidth = 1;

    for (let index = 0; index <= ticks; index += 1) {
      const ratio = index / ticks;
      const y = padding.top + chartHeight - chartHeight * ratio;
      const value = maxY * ratio;

      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      ctx.fillText(formatTick(value), 8, y + 4);
    }

    ctx.strokeStyle = "rgba(21, 32, 27, 0.20)";
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();
  }

  function drawXLabels(ctx, labels, width, height, padding) {
    const chartWidth = width - padding.left - padding.right;
    const bottom = height - padding.bottom + 18;
    const step = Math.max(1, Math.ceil(labels.length / 6));
    ctx.fillStyle = "#55635b";
    ctx.font = '12px Inter, "Segoe UI", Arial, sans-serif';

    labels.forEach((label, index) => {
      if (index % step !== 0 && index !== labels.length - 1) {
        return;
      }

      const x = labels.length === 1
        ? padding.left + chartWidth / 2
        : padding.left + (chartWidth * index) / (labels.length - 1);

      ctx.save();
      ctx.translate(x, bottom);
      ctx.rotate(-0.45);
      ctx.fillText(truncateLabel(label, 12), 0, 0);
      ctx.restore();
    });
  }

  function drawLineChart(canvas, config) {
    const { ctx, width, height } = setupCanvas(canvas);
    const padding = { top: 16, right: 16, bottom: 58, left: 58 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const maxY = getMaxValue(config.series);

    drawGrid(ctx, width, height, padding, maxY);
    drawXLabels(ctx, config.labels || [], width, height, padding);

    config.series.forEach((entry, seriesIndex) => {
      const values = (entry.values || []).map(toNumber);
      const color = entry.color || palette[seriesIndex % palette.length];

      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;

      values.forEach((value, index) => {
        const x = values.length === 1
          ? padding.left + chartWidth / 2
          : padding.left + (chartWidth * index) / (values.length - 1);
        const y = padding.top + chartHeight - (value / maxY) * chartHeight;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      if (values.length <= 40) {
        values.forEach((value, index) => {
          const x = values.length === 1
            ? padding.left + chartWidth / 2
            : padding.left + (chartWidth * index) / (values.length - 1);
          const y = padding.top + chartHeight - (value / maxY) * chartHeight;
          ctx.beginPath();
          ctx.fillStyle = color;
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    });
  }

  function drawBarChart(canvas, config) {
    const { ctx, width, height } = setupCanvas(canvas);
    const padding = { top: 16, right: 16, bottom: 72, left: 58 };
    const labels = config.labels || [];
    const series = config.series || [];
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const maxY = getMaxValue(series);
    const groupWidth = labels.length ? chartWidth / labels.length : chartWidth;
    const innerWidth = groupWidth * 0.72;
    const barWidth = series.length ? innerWidth / series.length : innerWidth;

    drawGrid(ctx, width, height, padding, maxY);
    ctx.font = '12px Inter, "Segoe UI", Arial, sans-serif';

    labels.forEach((label, labelIndex) => {
      const groupX = padding.left + labelIndex * groupWidth + (groupWidth - innerWidth) / 2;
      series.forEach((entry, seriesIndex) => {
        const value = toNumber((entry.values || [])[labelIndex]);
        const color = entry.color || palette[seriesIndex % palette.length];
        const barHeight = (value / maxY) * chartHeight;
        const x = groupX + seriesIndex * barWidth;
        const y = padding.top + chartHeight - barHeight;

        ctx.fillStyle = color;
        ctx.fillRect(x, y, Math.max(8, barWidth - 4), barHeight);
      });

      const labelX = padding.left + labelIndex * groupWidth + groupWidth / 2;
      ctx.save();
      ctx.translate(labelX, height - padding.bottom + 18);
      ctx.rotate(-0.55);
      ctx.fillStyle = "#55635b";
      ctx.fillText(truncateLabel(label, 12), 0, 0);
      ctx.restore();
    });
  }

  function drawLegend(container, series) {
    if (!container) {
      return;
    }

    container.innerHTML = (series || [])
      .map((entry, index) => {
        const color = entry.color || palette[index % palette.length];
        const name = String(entry.name || `Series ${index + 1}`);
        return `
          <span class="legend-item">
            <span class="legend-swatch" style="background:${color}"></span>
            <span>${name}</span>
          </span>
        `;
      })
      .join("");
  }

  function renderChart(canvas) {
    const sourceId = canvas.dataset.chartSource;
    const legendId = canvas.dataset.chartLegend;
    const source = document.getElementById(sourceId);
    if (!source) {
      return;
    }

    const config = JSON.parse(source.textContent);
    config.series = (config.series || []).map((entry, index) => ({
      ...entry,
      color: entry.color || palette[index % palette.length],
      values: (entry.values || []).map(toNumber),
    }));

    if (config.type === "bar") {
      drawBarChart(canvas, config);
    } else {
      drawLineChart(canvas, config);
    }

    drawLegend(document.getElementById(legendId), config.series);
  }

  let resizeTimer = null;
  function renderAllCharts() {
    document.querySelectorAll("canvas[data-chart-source]").forEach(renderChart);
  }

  window.addEventListener("resize", () => {
    if (resizeTimer) {
      window.clearTimeout(resizeTimer);
    }
    resizeTimer = window.setTimeout(renderAllCharts, 120);
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderAllCharts);
  } else {
    renderAllCharts();
  }
})();
</script>
"""


NAV_ITEMS = [
    ("/", "Главная"),
    ("/health", "Состояние"),
    ("/recommendation", "Рекомендации"),
    ("/model-info", "Модель"),
    ("/recommendations/history", "История"),
    ("/data", "Данные"),
    ("/docs", "Swagger"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommendation_service = RecommendationService.create()
    yield


app = FastAPI(title="Resource Recommender Service", version="1.0.0", lifespan=lifespan)


def get_service(app_instance: FastAPI) -> RecommendationService:
    return app_instance.state.recommendation_service


def wants_html(request: Request, format_value: str | None) -> bool:
    if format_value == "html":
        return True
    if format_value == "json":
        return False

    accept_header = request.headers.get("accept", "")
    return "text/html" in accept_header


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def format_time_window_label(value: Any) -> str:
    try:
        total_seconds = int(float(value))
    except (TypeError, ValueError):
        return format_value(value)

    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, _ = divmod(remainder, 60)

    if days:
        return f"D{days} {hours:02d}:{minutes:02d}"
    return f"{hours:02d}:{minutes:02d}"


def render_cards(items: list[tuple[str, str]]) -> str:
    cards = []
    for label, value in items:
        cards.append(
            f"""
            <div class="card">
              <div class="card-label">{escape(label)}</div>
              <div class="card-value">{escape(value)}</div>
            </div>
            """
        )
    return f'<section class="grid">{"".join(cards)}</section>'


def render_table(headers: list[str], rows: list[list[Any]], empty_text: str = "Нет данных") -> str:
    if not rows:
        return f'<div class="empty">{escape(empty_text)}</div>'

    thead = "".join(f"<th>{escape(header)}</th>" for header in headers)
    tbody_rows = []
    for row in rows:
        cells = "".join(f"<td>{escape(format_value(cell))}</td>" for cell in row)
        tbody_rows.append(f"<tr>{cells}</tr>")

    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{thead}</tr></thead>
        <tbody>{"".join(tbody_rows)}</tbody>
      </table>
    </div>
    """


def render_key_value_grid(items: dict[str, Any]) -> str:
    blocks = []
    for key, value in items.items():
        blocks.append(
            f"""
            <div class="meta-item">
              <strong>{escape(str(key))}</strong>
              <span>{escape(format_value(value))}</span>
            </div>
            """
        )
    return f'<div class="meta">{"".join(blocks)}</div>'


def render_chart_card(
    chart_id: str,
    title: str,
    description: str,
    chart_type: str,
    labels: list[str],
    series: list[dict[str, Any]],
    chart_height: int = 290,
) -> str:
    if not labels or not series:
        return ""

    script_id = f"{chart_id}-data"
    legend_id = f"{chart_id}-legend"
    config = {
        "type": chart_type,
        "labels": labels,
        "series": series,
    }

    return f"""
    <section class="section chart-card">
      <div class="section-header">
        <h2>{escape(title)}</h2>
        <p>{escape(description)}</p>
      </div>
      <div class="chart-frame">
        <canvas
          class="chart-canvas"
          data-chart-source="{escape(script_id)}"
          data-chart-legend="{escape(legend_id)}"
          data-chart-height="{chart_height}"
        ></canvas>
      </div>
      <div class="chart-legend" id="{escape(legend_id)}"></div>
      <script type="application/json" id="{escape(script_id)}">{json_for_script(config)}</script>
    </section>
    """


def render_layout(
    service: RecommendationService,
    title: str,
    subtitle: str,
    body_html: str,
    active_path: str,
) -> str:
    health_info = service.health()
    nav_links: list[str] = []
    for path, label in NAV_ITEMS:
        active_attr = ' class="active"' if path == active_path else ""
        nav_links.append(f'<a href="{escape(path)}"{active_attr}>{escape(label)}</a>')
    nav_html = "".join(nav_links)

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} | {escape(str(health_info["service_name"]))}</title>
  {STYLE_BLOCK}
</head>
<body>
  <main class="page">
    <header class="topbar">
      <div class="brand">
        <span class="brand-badge">Resource Recommender</span>
        <span class="brand-name">{escape(str(health_info["service_name"]))}</span>
      </div>
      <nav class="nav">{nav_html}</nav>
    </header>

    <section class="hero">
      <div class="eyebrow">Локальный сервис рекомендаций ресурсов</div>
      <h1>{escape(title)}</h1>
      <p>{escape(subtitle)}</p>
    </section>

    {body_html}
    {SCRIPT_BLOCK}
  </main>
</body>
</html>"""


def render_home_page(service: RecommendationService) -> str:
    health_info = service.health()
    model_info = service.model_info()
    data_summary = service.data_overview(limit=3)["summary"]

    cards_html = render_cards(
        [
            ("Статус", "OK"),
            ("Версия модели", format_value(health_info["model_version"])),
            ("Контейнеров", format_value(data_summary["containers"])),
            ("Машин", format_value(data_summary["machines"])),
        ]
    )

    body = f"""
    {cards_html}

    <section class="two-col">
      <div class="section">
        <h2>Сервис</h2>
        <p>
          Сервис принимает метрики контейнера, формирует признаки и возвращает
          рекомендацию по CPU и памяти через API.
        </p>
        <p>
          Лучшая модель для CPU: <strong>{escape(format_value(model_info["best_cpu_model_name"]))}</strong>.
          Лучшая модель для RAM: <strong>{escape(format_value(model_info["best_ram_model_name"]))}</strong>.
          Окно прогноза: <strong>{escape(format_value(data_summary["window_minutes"]))} минут</strong>.
        </p>
        <div class="actions">
          <a class="button primary" href="/docs">Открыть Swagger</a>
          <a class="button" href="/data">Открыть графики</a>
          <a class="button" href="/recommendation">Посмотреть пример запроса</a>
          <a class="button" href="/model-info">Открыть описание модели</a>
        </div>
      </div>

      <div class="section">
        <h2>Сводка</h2>
        <div class="meta">
          <div class="meta-item">
            <strong>Обучено</strong>
            <span>{escape(format_value(health_info["trained_at"]))}</span>
          </div>
          <div class="meta-item">
            <strong>История запросов</strong>
            <span>{escape(format_value(health_info["history_entries"]))}</span>
          </div>
          <div class="meta-item">
            <strong>Контейнеров</strong>
            <span>{escape(format_value(data_summary["containers"]))}</span>
          </div>
          <div class="meta-item">
            <strong>Машин</strong>
            <span>{escape(format_value(data_summary["machines"]))}</span>
          </div>
        </div>
      </div>
    </section>
    """

    return render_layout(
        service=service,
        title="Сервис рекомендаций ресурсов",
        subtitle="API для прогноза CPU и RAM контейнеров и выдачи рекомендаций.",
        body_html=body,
        active_path="/",
    )


def render_health_page(service: RecommendationService) -> str:
    payload = service.health()
    cards_html = render_cards(
        [
            ("Статус", format_value(payload["status"]).upper()),
            ("Версия", format_value(payload["model_version"])),
            ("История запросов", format_value(payload["history_entries"])),
            ("Сервис", format_value(payload["service_name"])),
        ]
    )

    body = f"""
    {cards_html}
    <section class="section">
      <h2>Подробности</h2>
      {render_key_value_grid(payload)}
    </section>
    """

    return render_layout(
        service=service,
        title="Состояние сервиса",
        subtitle="Проверка доступности сервиса и текущего состояния загруженной модели.",
        body_html=body,
        active_path="/health",
    )


def render_model_info_page(service: RecommendationService) -> str:
    payload = service.model_info()
    metrics = payload["metrics"]

    metric_headers = [
        "Model",
        "CPU_MAE_val",
        "CPU_RMSE_val",
        "CPU_R2_val",
        "RAM_MAE_val",
        "RAM_RMSE_val",
        "RAM_R2_val",
    ]
    metric_rows = [
        [
            model_name,
            values.get("CPU_MAE_val"),
            values.get("CPU_RMSE_val"),
            values.get("CPU_R2_val"),
            values.get("RAM_MAE_val"),
            values.get("RAM_RMSE_val"),
            values.get("RAM_R2_val"),
        ]
        for model_name, values in metrics.items()
    ]

    body = f"""
    {render_cards([
        ("Лучшая модель CPU", format_value(payload["best_cpu_model_name"])),
        ("Лучшая модель RAM", format_value(payload["best_ram_model_name"])),
        ("Признаков", str(len(payload["feature_cols"]))),
        ("Последовательных признаков", str(len(payload["seq_cols"]))),
    ])}

    <section class="section">
      <h2>Конфигурация</h2>
      {render_key_value_grid(payload["config"])}
    </section>

    <section class="section">
      <h2>Качество моделей</h2>
      {render_table(metric_headers, metric_rows, empty_text="Метрики пока недоступны")}
    </section>

    <section class="section">
      <h2>Признаки</h2>
      <div class="pill-list">
        {"".join(f'<span class="pill">{escape(feature)}</span>' for feature in payload["feature_cols"])}
      </div>
    </section>
    """

    return render_layout(
        service=service,
        title="Информация о модели",
        subtitle="Выбранные модели, признаки, параметры и метрики качества.",
        body_html=body,
        active_path="/model-info",
    )


def render_history_page(service: RecommendationService, limit: int, container_id: str | None) -> str:
    payload = service.history(limit=limit, container_id=container_id)
    rows = [
        [
            item.get("processed_at"),
            item.get("container_id"),
            item.get("decision_label"),
            item.get("actions", {}).get("cpu"),
            item.get("actions", {}).get("ram"),
            item.get("recommendation", {}).get("cpu_limit"),
            item.get("recommendation", {}).get("mem_size"),
        ]
        for item in payload["items"]
    ]

    subtitle = "История уже рассчитанных рекомендаций."
    if container_id:
        subtitle = f"История рекомендаций для контейнера {container_id}."

    body = f"""
    {render_cards([
        ("Записей в истории", format_value(payload["count"])),
        ("Показано на странице", str(len(payload["items"]))),
        ("Лимит", str(limit)),
        ("Контейнер", container_id or "все"),
    ])}

    <section class="section">
      <h2>Записи</h2>
      {render_table(
          ["Время", "Контейнер", "Решение", "CPU action", "RAM action", "CPU limit", "Mem size"],
          rows,
          empty_text="История пока пуста"
      )}
    </section>
    """

    return render_layout(
        service=service,
        title="История рекомендаций",
        subtitle=subtitle,
        body_html=body,
        active_path="/recommendations/history",
    )


def render_data_page(service: RecommendationService, container_id: str | None, limit: int) -> str:
    payload = service.data_overview(container_id=container_id, limit=limit)
    summary = payload["summary"]

    cpu_rows = [
        [row["time_window"], row["cpu_request_total"], row["cpu_limit_total"], row["cpu_usage_total"]]
        for row in payload["cluster_cpu_over_time"][: min(12, len(payload["cluster_cpu_over_time"]))]
    ]
    mem_rows = [
        [row["time_window"], row["mem_size_total"], row["mem_usage_total"]]
        for row in payload["cluster_mem_over_time"][: min(12, len(payload["cluster_mem_over_time"]))]
    ]
    app_rows = [
        [row["app_du"], row["container_count"]]
        for row in payload["app_container_counts"][: min(12, len(payload["app_container_counts"]))]
    ]
    machine_rows = [
        [row["machine_id"], row["app_count"]]
        for row in payload["machine_app_counts"][: min(12, len(payload["machine_app_counts"]))]
    ]

    cpu_chart = render_chart_card(
        chart_id="cluster-cpu",
        title="CPU от времени",
        description="Суммарные CPU request, CPU limit и фактическое CPU usage по временным окнам.",
        chart_type="line",
        labels=[format_time_window_label(row["time_window"]) for row in payload["cluster_cpu_over_time"]],
        series=[
            {
                "name": "CPU request",
                "values": [row["cpu_request_total"] for row in payload["cluster_cpu_over_time"]],
                "color": "#1f7a5c",
            },
            {
                "name": "CPU limit",
                "values": [row["cpu_limit_total"] for row in payload["cluster_cpu_over_time"]],
                "color": "#c46c2a",
            },
            {
                "name": "CPU usage",
                "values": [row["cpu_usage_total"] for row in payload["cluster_cpu_over_time"]],
                "color": "#2f6f8f",
            },
        ],
        chart_height=300,
    )

    mem_chart = render_chart_card(
        chart_id="cluster-mem",
        title="RAM от времени",
        description="На одном графике показаны суммарные mem size и фактическое mem usage по временным окнам.",
        chart_type="line",
        labels=[format_time_window_label(row["time_window"]) for row in payload["cluster_mem_over_time"]],
        series=[
            {
                "name": "Mem size",
                "values": [row["mem_size_total"] for row in payload["cluster_mem_over_time"]],
                "color": "#7a55b6",
            },
            {
                "name": "Mem usage",
                "values": [row["mem_usage_total"] for row in payload["cluster_mem_over_time"]],
                "color": "#c04d7b",
            },
        ],
        chart_height=300,
    )

    app_chart = render_chart_card(
        chart_id="app-container-counts",
        title="Контейнеры по app_du",
        description="Количество контейнеров в кластере для наиболее представленных приложений.",
        chart_type="bar",
        labels=[row["app_du"] for row in payload["app_container_counts"][:12]],
        series=[
            {
                "name": "Контейнеров",
                "values": [row["container_count"] for row in payload["app_container_counts"][:12]],
                "color": "#1f7a5c",
            }
        ],
        chart_height=310,
    )

    machine_chart = render_chart_card(
        chart_id="machine-app-counts",
        title="Приложения по machine_id",
        description="Сколько разных app_du размещено на каждой виртуальной машине из топа.",
        chart_type="bar",
        labels=[row["machine_id"] for row in payload["machine_app_counts"][:12]],
        series=[
            {
                "name": "Приложений",
                "values": [row["app_count"] for row in payload["machine_app_counts"][:12]],
                "color": "#c46c2a",
            }
        ],
        chart_height=310,
    )

    hour_chart = render_chart_card(
        chart_id="resource-by-hour",
        title="Потребление по времени суток",
        description="Средняя загрузка CPU и RAM по относительному часу суток.",
        chart_type="line",
        labels=[f"{int(row['relative_hour']):02d}:00" for row in payload["resource_by_hour"]],
        series=[
            {
                "name": "CPU avg, %",
                "values": [row["cpu_util_mean_avg"] for row in payload["resource_by_hour"]],
                "color": "#2f6f8f",
            },
            {
                "name": "RAM avg, %",
                "values": [row["mem_util_mean_avg"] for row in payload["resource_by_hour"]],
                "color": "#c04d7b",
            },
        ],
        chart_height=290,
    )

    day_chart = render_chart_card(
        chart_id="resource-by-day",
        title="Потребление по циклу дней",
        description="Средняя загрузка CPU и RAM по циклическому признаку дня от 0 до 6.",
        chart_type="line",
        labels=[f"День {int(row['relative_day_cycle'])}" for row in payload["resource_by_day"]],
        series=[
            {
                "name": "CPU avg, %",
                "values": [row["cpu_util_mean_avg"] for row in payload["resource_by_day"]],
                "color": "#1f7a5c",
            },
            {
                "name": "RAM avg, %",
                "values": [row["mem_util_mean_avg"] for row in payload["resource_by_day"]],
                "color": "#7a55b6",
            },
        ],
        chart_height=290,
    )

    bootstrap_rows = [
        [
            row["container_id"],
            row["decision_label"],
            row["recommended_cpu_limit"],
            row["recommended_mem_size"],
            row["cpu_action"],
            row["ram_action"],
        ]
        for row in payload["bootstrap_report_preview"]
    ]

    container_block = """
    <section class="section">
      <h2>Фильтр по контейнеру</h2>
      <p>
        Чтобы посмотреть отдельный временной ряд контейнера с графиками request, limit и usage,
        передайте <code>container_id</code> в адресной строке, например <code>/data?container_id=c_1</code>.
      </p>
    </section>
    """
    if "container_series" in payload:
        container_series = payload["container_series"]
        container_labels = [format_time_window_label(row["time_window"]) for row in container_series]
        container_rows = [
            [
                row["time_window"],
                row["cpu_request"],
                row["cpu_limit"],
                row["cpu_util_mean"],
                row["mem_size"],
                row["mem_util_mean"],
                row["samples_in_window"],
            ]
            for row in container_series
        ]
        container_cpu_chart = render_chart_card(
            chart_id="container-cpu",
            title=f"CPU контейнера {container_id}",
            description="Детальный ряд выбранного контейнера: request, limit и рассчитанное usage в абсолютных значениях.",
            chart_type="line",
            labels=container_labels,
            series=[
                {
                    "name": "CPU request",
                    "values": [row["cpu_request"] for row in container_series],
                    "color": "#1f7a5c",
                },
                {
                    "name": "CPU limit",
                    "values": [row["cpu_limit"] for row in container_series],
                    "color": "#c46c2a",
                },
                {
                    "name": "CPU usage",
                    "values": [(row["cpu_util_mean"] / 100.0) * row["cpu_limit"] for row in container_series],
                    "color": "#2f6f8f",
                },
            ],
            chart_height=300,
        )
        container_mem_chart = render_chart_card(
            chart_id="container-mem",
            title=f"RAM контейнера {container_id}",
            description="Детальный ряд выбранного контейнера: mem size и рассчитанное mem usage в абсолютных значениях.",
            chart_type="line",
            labels=container_labels,
            series=[
                {
                    "name": "Mem size",
                    "values": [row["mem_size"] for row in container_series],
                    "color": "#7a55b6",
                },
                {
                    "name": "Mem usage",
                    "values": [(row["mem_util_mean"] / 100.0) * row["mem_size"] for row in container_series],
                    "color": "#c04d7b",
                },
            ],
            chart_height=300,
        )
        container_block = f"""
        <section class="section">
          <h2>Контейнер {escape(container_id or '')}</h2>
          <p>
            Ниже показан тот же временной ряд, но уже для одного контейнера, чтобы можно было
            посмотреть поведение без агрегирования по всему кластеру.
          </p>
        </section>

        <section class="chart-grid">
          {container_cpu_chart}
          {container_mem_chart}
        </section>

        <section class="section">
          <h2>Таблица по контейнеру</h2>
          {render_table(
              ["time_window", "cpu_request", "cpu_limit", "cpu_util_mean", "mem_size", "mem_util_mean", "samples"],
              container_rows,
              empty_text="Нет данных по контейнеру"
          )}
        </section>
        """

    body = f"""
    {render_cards([
        ("Окно прогноза, минут", format_value(summary["window_minutes"])),
        ("Контейнеров", format_value(summary["containers"])),
        ("Машин", format_value(summary["machines"])),
        ("Рекомендаций в bootstrap", format_value(summary["bootstrap_recommendations"])),
    ])}

    <section class="section">
      <h2>Что видно по агрегированным графикам</h2>
      <p>
        Если на графиках CPU limit и mem size появляются заметные просадки, это обычно означает,
        что в этот период уменьшилось число активных контейнеров или у части контейнеров изменились лимиты.
        Для агрегированного ряда это нормальная ситуация и не обязательно аномалия нагрузки.
      </p>
      <p class="note">
        JSON-данные по этим же графикам доступны через <code>/data?format=json</code>.
      </p>
    </section>

    <section class="chart-grid">
      {app_chart}
      {machine_chart}
    </section>

    <section class="chart-grid">
      {cpu_chart}
      {mem_chart}
    </section>

    <section class="chart-grid">
      {hour_chart}
      {day_chart}
    </section>

    {container_block}

    <section class="two-col">
      <div class="section">
        <h2>Таблица CPU</h2>
        {render_table(["time_window", "cpu_request_total", "cpu_limit_total", "cpu_usage_total"], cpu_rows)}
      </div>
      <div class="section">
        <h2>Таблица RAM</h2>
        {render_table(["time_window", "mem_size_total", "mem_usage_total"], mem_rows)}
      </div>
    </section>

    <section class="two-col">
      <div class="section">
        <h2>Таблица app_du</h2>
        {render_table(["app_du", "container_count"], app_rows)}
      </div>
      <div class="section">
        <h2>Таблица machine_id</h2>
        {render_table(["machine_id", "app_count"], machine_rows)}
      </div>
    </section>

    <section class="section">
      <h2>Превью bootstrap-рекомендаций</h2>
      <p>
        Небольшой фрагмент уже рассчитанных рекомендаций по сохранённым данным, чтобы быстро проверить,
        как сервис формирует итоговые решения.
      </p>
      {render_table(
          ["container_id", "decision", "recommended_cpu_limit", "recommended_mem_size", "cpu_action", "ram_action"],
          bootstrap_rows,
          empty_text="Предварительных рекомендаций пока нет"
      )}
    </section>
    """

    subtitle = "Сводные данные, графики и таблицы по обучающему набору и расчётным рекомендациям."
    if container_id:
        subtitle = f"Сводные данные, графики и детальный временной ряд для контейнера {container_id}."

    return render_layout(
        service=service,
        title="Данные и графики",
        subtitle=subtitle,
        body_html=body,
        active_path="/data",
    )


def render_recommendation_help_page(service: RecommendationService) -> str:
    sample_payload = {
        "meta": {
            "container_id": "demo-container",
            "machine_id": "demo-machine",
            "app_du": "demo-app",
            "status": "started",
            "cpu_request": 400,
            "cpu_limit": 800,
            "mem_size": 3.13,
        },
        "usage": [
            {"time_stamp": 0, "cpu_util_percent": 20, "mem_util_percent": 45},
            {"time_stamp": 600, "cpu_util_percent": 25, "mem_util_percent": 47},
            {"time_stamp": 1200, "cpu_util_percent": 22, "mem_util_percent": 50},
            {"time_stamp": 1800, "cpu_util_percent": 30, "mem_util_percent": 52},
            {"time_stamp": 2400, "cpu_util_percent": 28, "mem_util_percent": 49},
            {"time_stamp": 3000, "cpu_util_percent": 35, "mem_util_percent": 53},
            {"time_stamp": 3600, "cpu_util_percent": 40, "mem_util_percent": 55},
            {"time_stamp": 4200, "cpu_util_percent": 38, "mem_util_percent": 58},
        ],
        "include_features": True,
        "include_window_series": True,
    }

    body = f"""
    {render_cards([
        ("Метод", "POST"),
        ("Эндпоинт", "/recommendation"),
        ("Принимает", "JSON"),
        ("Возвращает", "JSON"),
    ])}

    <section class="section">
      <h2>Описание</h2>
      <p>
        Эндпоинт рассчитывает прогноз CPU и RAM по окну метрик контейнера
        и возвращает итоговую рекомендацию.
      </p>
      <div class="actions">
        <a class="button primary" href="/docs#/default/recommendation_recommendation_post">Открыть запрос в Swagger</a>
      </div>
    </section>

    <section class="section">
      <h2>Пример тела запроса</h2>
      <div class="console"><pre>{escape(json.dumps(sample_payload, ensure_ascii=False, indent=2))}</pre></div>
    </section>
    """

    return render_layout(
        service=service,
        title="Эндпоинт рекомендаций",
        subtitle="POST /recommendation",
        body_html=body,
        active_path="/recommendation",
    )


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(render_home_page(get_service(app)))


@app.get("/health", response_model=None)
def health(
    request: Request,
    format: str | None = Query(default=None, pattern="^(html|json)$"),
) -> HTMLResponse | dict[str, Any]:
    service = get_service(app)
    if wants_html(request, format):
        return HTMLResponse(render_health_page(service))
    return service.health()


@app.get("/recommendation", response_class=HTMLResponse)
def recommendation_help() -> HTMLResponse:
    return HTMLResponse(render_recommendation_help_page(get_service(app)))


@app.post("/recommendation")
def recommendation(payload: RecommendationRequest) -> dict[str, Any]:
    service = get_service(app)
    try:
        return service.recommend(
            meta_payload=payload.meta.model_dump(),
            usage_payload=[item.model_dump() for item in payload.usage],
            include_features=payload.include_features,
            include_window_series=payload.include_window_series,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/model-info", response_model=None)
def model_info(
    request: Request,
    format: str | None = Query(default=None, pattern="^(html|json)$"),
) -> HTMLResponse | dict[str, Any]:
    service = get_service(app)
    if wants_html(request, format):
        return HTMLResponse(render_model_info_page(service))
    return service.model_info()


@app.get("/recommendations/history", response_model=None)
def recommendations_history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    container_id: str | None = Query(default=None),
    format: str | None = Query(default=None, pattern="^(html|json)$"),
) -> HTMLResponse | dict[str, Any]:
    service = get_service(app)
    if wants_html(request, format):
        return HTMLResponse(render_history_page(service, limit=limit, container_id=container_id))
    return service.history(limit=limit, container_id=container_id)


@app.get("/data", response_model=None)
def data(
    request: Request,
    container_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    format: str | None = Query(default=None, pattern="^(html|json)$"),
) -> HTMLResponse | dict[str, Any]:
    service = get_service(app)
    try:
        if wants_html(request, format):
            return HTMLResponse(render_data_page(service, container_id=container_id, limit=limit))
        return service.data_overview(container_id=container_id, limit=limit)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
