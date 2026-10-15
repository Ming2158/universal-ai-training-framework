"""Local Gradio interface that delegates training to the UATF CLI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from stat import S_IFDIR, S_IFMT, S_IFREG, S_ISLNK
from typing import Any, Iterator

import gradio as gr
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision.transforms import ToPILImage

from app.monitoring import HardwareResourceSampler, format_resource_snapshot
from uatf.data import DatasetInspection, inspect_dataset
from uatf.data.transforms import build_image_transform
from uatf.evaluation import EvaluationResult, evaluate_model_asset
from uatf.experiments import (
    compare_experiments,
    export_model_asset,
    get_experiment,
    list_experiments,
    list_model_assets,
    load_model_asset,
)
from uatf.inference import (
    predict_model_asset_directory,
    predict_model_asset_image,
    prediction_rows,
)
from uatf.tasks import ACTIVE_STATUSES, get_latest_task, get_task, request_task_stop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "ui_uploads"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ui_runs"
CONFIG_ROOT = PROJECT_ROOT / "outputs" / "ui_configs"
DATASET_ROOT = PROJECT_ROOT / "outputs" / "ui_datasets"
DATASET_REGISTRY = DATASET_ROOT / "registry"
MAX_ZIP_MEMBERS = 50_000
MAX_ZIP_UNCOMPRESSED_BYTES = 20 * 1024**3
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
_NORMALIZE_STD = (0.229, 0.224, 0.225)
WORKSPACE_PAGES = (
    "工作台",
    "数据集",
    "训练",
    "训练监控",
    "实验管理",
    "模型管理",
    "评估与推理",
)
GUI_MONITORED_TASK_IDS: set[str] = set()
WORKSPACE_NEUTRAL_HUE = gr.themes.Color(
    c50="#f4f7f5",
    c100="#eef3f0",
    c200="#d9e1dc",
    c300="#c7d2cb",
    c400="#97a49c",
    c500="#77847b",
    c600="#5f6d64",
    c700="#46534b",
    c800="#202b26",
    c900="#151d19",
    c950="#0b100d",
    name="uatf-neutral",
)
WORKSPACE_CSS = """
:root {
  --uatf-ink: #202b26;
  --uatf-muted: #63716a;
  --uatf-border: #d9e1dc;
  --uatf-surface: #ffffff;
  --uatf-canvas: #f4f7f5;
  --uatf-green: #267256;
  --uatf-green-soft: #e8f2ed;
  --uatf-orange: #bd6439;
  --uatf-danger: #ad3f42;
}
body, .gradio-container { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
body, .gradio-container { background: var(--uatf-canvas) !important; color: var(--uatf-ink); }
.gradio-container { max-width: none !important; padding: 16px 20px !important; }
#uatf-shell { gap: 18px !important; align-items: flex-start !important; flex-wrap: nowrap !important; }
#uatf-sidebar {
  position: sticky; top: 16px; height: calc(100vh - 32px); overflow-y: auto;
  flex: 0 0 218px !important; min-width: 184px !important; padding: 14px 12px;
  background: var(--uatf-surface); border: 1px solid var(--uatf-border); border-radius: 6px;
}
#uatf-main { flex: 1 1 auto !important; min-width: 0 !important; max-width: 1680px; margin: 0 auto; }
#uatf-main > .block { background: transparent !important; }
#uatf-navigation .wrap { display: flex; flex-direction: column; gap: 3px; }
#uatf-navigation label { min-height: 38px; border-radius: 5px; color: var(--uatf-muted); }
#uatf-navigation label:has(input:checked) { background: var(--uatf-green-soft); color: var(--uatf-green); font-weight: 650; }
#uatf-brand { padding: 4px 8px 14px; border-bottom: 1px solid var(--uatf-border); margin-bottom: 12px; }
#uatf-brand h2 { font-size: 18px; margin: 0 0 3px; color: var(--uatf-ink); }
#uatf-brand p { font-size: 12px; color: var(--uatf-muted); margin: 0; }
#uatf-version { border-top: 1px solid var(--uatf-border); margin-top: 12px; padding: 12px 8px 0; color: var(--uatf-muted); font-size: 13px; }
.uatf-page-header { border-bottom: 1px solid var(--uatf-border); padding: 4px 0 12px; margin-bottom: 14px; background: transparent !important; }
.uatf-page-header h1 { font-size: 25px; line-height: 1.25; margin: 0 0 5px; }
.uatf-page-header p { color: var(--uatf-muted); margin: 0; }
.uatf-section { background: var(--uatf-surface) !important; border: 1px solid var(--uatf-border); border-radius: 6px; padding: 14px; }
.uatf-section-title { font-size: 16px; margin: 0 0 10px; }
.uatf-note { color: var(--uatf-muted); font-size: 13px; }
.uatf-status { border-left: 3px solid var(--uatf-green); padding: 8px 10px; background: var(--uatf-green-soft); }
[id^="uatf-empty-state-"] { border-left: 3px solid #7b9a89; border-radius: 5px; padding: 14px 16px; background: #eef4f0 !important; color: var(--uatf-muted); }
[id^="uatf-empty-state-"] p:last-child { margin-bottom: 0; }
.uatf-results { background: transparent !important; }
.uatf-responsive-row { flex-wrap: wrap !important; }
button.primary { background: var(--uatf-green) !important; border-color: var(--uatf-green) !important; }
button.stop { background: var(--uatf-danger) !important; border-color: var(--uatf-danger) !important; }
button { box-shadow: none !important; }
input, textarea, select { border-radius: 5px !important; }
.dataframe-container, .table-wrap { max-width: 100%; overflow: auto; }
.dataframe-container table { font-size: 13px; }
.prose code { overflow-wrap: anywhere; }
@media (max-width: 1100px) {
  .gradio-container { padding: 10px !important; }
  #uatf-shell { gap: 10px !important; }
  #uatf-sidebar { flex-basis: 168px !important; min-width: 154px !important; padding: 10px 8px; }
  .uatf-responsive-row > .column { flex: 1 1 100% !important; min-width: 0 !important; width: 100% !important; }
  .uatf-responsive-row > .block { flex: 1 1 280px !important; min-width: 0 !important; }
  .uatf-page-header h1 { font-size: 22px; }
}
@media (max-width: 760px) {
  #uatf-shell { flex-wrap: wrap !important; }
  #uatf-sidebar { position: relative; top: auto; height: auto; flex: 1 0 100% !important; min-width: 0 !important; }
  #uatf-navigation .wrap { flex-direction: row; flex-wrap: wrap; }
  #uatf-navigation label { flex: 1 1 132px; }
}
"""


def _workspace_visibility(page: str) -> tuple[dict[str, Any], ...]:
    selected = page if page in WORKSPACE_PAGES else "工作台"
    return tuple(gr.update(visible=name == selected) for name in WORKSPACE_PAGES)


def _navigate_workspace(page: str) -> tuple[Any, ...]:
    selected = page if page in WORKSPACE_PAGES else "工作台"
    return (gr.update(value=selected), *_workspace_visibility(selected))


def load_dashboard_summary() -> tuple[str, pd.DataFrame]:
    task = get_latest_task(OUTPUT_ROOT)
    if task is None:
        task_summary = "当前没有训练任务。"
    else:
        labels = {
            "Pending": "等待执行",
            "Preparing": "准备中",
            "Running": "训练中",
            "Validating": "验证中",
            "Stopping": "正在停止",
            "Stopped": "已停止",
            "Completed": "已完成",
            "Failed": "失败",
        }
        task_summary = (
            f"**{labels.get(task.get('status'), task.get('status', '未知'))}**  ·  "
            f"任务 `{task.get('task_id', '')}`  ·  "
            f"Epoch {task.get('epoch') or 0} / {task.get('total_epochs') or '-'}"
        )
    columns = [
        "id",
        "name",
        "created_at_utc",
        "status",
        "task",
        "model",
        "dataset_id",
        "best_metric",
        "duration_seconds",
        "legacy",
    ]
    rows = list_experiments(OUTPUT_ROOT)
    return task_summary, pd.DataFrame(rows[:8], columns=columns)


def load_task_artifacts(
    task_id: str | None,
) -> tuple[str, Any, pd.DataFrame, str | None, str | None]:
    columns = [
        "kind",
        "epoch",
        "saved_at_utc",
        "size_bytes",
        "supports_resume",
        "inference_only",
        "path",
    ]
    if not task_id:
        return "暂无任务产物", {}, pd.DataFrame(columns=columns), None, None
    record = get_task(OUTPUT_ROOT, task_id)
    experiment_id = record.get("experiment_id") if record else None
    if not experiment_id:
        return "实验目录尚未生成", {}, pd.DataFrame(columns=columns), None, None
    try:
        detail = get_experiment(OUTPUT_ROOT, str(experiment_id))
    except (OSError, ValueError) as error:
        return f"读取任务产物失败：{error}", {}, pd.DataFrame(columns=columns), None, None
    checkpoints = detail["checkpoints"]
    rows = [
        {
            "kind": item["kind"],
            "epoch": item.get("epoch"),
            "saved_at_utc": item["saved_at_utc"],
            "size_bytes": item["size_bytes"],
            "supports_resume": item["supports_resume"],
            "inference_only": item["inference_only"],
            "path": item["path"],
        }
        for item in checkpoints
    ]
    files = {item["kind"]: item["path"] for item in checkpoints}
    return (
        str(detail["record"].get("name") or experiment_id),
        detail["configuration"] or {},
        pd.DataFrame(rows, columns=columns),
        files.get("best"),
        files.get("latest"),
    )


def start_training_for_ui(
    upload_path: str | None,
    local_path: str | None,
    dataset_state: dict[str, Any] | None,
    task: str,
    data_format: str,
    validation_fraction: float,
    seed: float,
    image_size: float,
    accept_warnings: bool,
    model_name: str,
    epochs: float,
    batch_size: float,
    learning_rate: float,
    device: str,
    augmentation_enabled: bool = False,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> Iterator[tuple[Any, ...]]:
    routed = False
    for update in start_training(
        upload_path,
        local_path,
        dataset_state,
        task,
        data_format,
        validation_fraction,
        seed,
        image_size,
        accept_warnings,
        model_name,
        epochs,
        batch_size,
        learning_rate,
        device,
        augmentation_enabled,
        progress,
    ):
        task_id = update[24] if len(update) > 24 and isinstance(update[24], str) else None
        try:
            task = get_task(OUTPUT_ROOT, task_id) if task_id else None
        except FileNotFoundError:
            task = None
        can_stop = bool(task and task.get("status") in ACTIVE_STATUSES)
        if task_id:
            if can_stop:
                GUI_MONITORED_TASK_IDS.add(task_id)
            else:
                GUI_MONITORED_TASK_IDS.discard(task_id)
        stop_update = gr.update(interactive=can_stop)
        if not routed and len(update) > 24 and isinstance(update[24], str) and update[24]:
            routed = True
            yield (*update, *_navigate_workspace("训练监控"), stop_update)
        else:
            yield (
                *update,
                gr.skip(),
                *(gr.skip() for _ in WORKSPACE_PAGES),
                stop_update,
            )


def launch_copied_configuration_for_ui(configuration_text: str | None) -> tuple[Any, ...]:
    status, task_id = launch_copied_configuration(configuration_text)
    if task_id:
        monitor_status = f"任务 {task_id} 已提交给 CLI。点击“刷新任务”读取持久状态。"
        return (status, monitor_status, task_id, *_navigate_workspace("训练监控"))
    return (
        status,
        gr.skip(),
        task_id,
        gr.skip(),
        *(gr.skip() for _ in WORKSPACE_PAGES),
    )


def _format_choices(task: str) -> dict[str, Any]:
    if task == "multi_label":
        return gr.update(choices=["csv_multilabel"], value="csv_multilabel")
    return gr.update(choices=["image_folder", "csv_classification"], value="image_folder")


def _dataset_instructions(task: str, data_format: str | None = None) -> str:
    if data_format == "csv_classification":
        return (
            "CSV 单标签格式：`image_path,label` 两列；存在 `images/` 时相对路径以该目录为基准，"
            "否则以 CSV 所在目录为基准。"
            " 可上传 ZIP，或填写 UATF 服务端可访问的 CSV/数据集目录路径。"
        )
    if task == "multi_label" or data_format == "csv_multilabel":
        return (
            "多标签 CSV 使用 `image_path` 和标签列，标签值只能是 `0` 或 `1`；"
            "ZIP 推荐包含根目录 `annotations.csv` 与 `images/`。存在 `images/` 时，"
            "相对路径以它为基准，例如 `cats/a.jpg`。"
        )
    return (
        "目录分类格式：每个类别一个子目录，图片放在对应目录中。"
        "ZIP 可直接包含类别目录，也可包含一个外层数据集目录。"
    )


def build_training_config(
    dataset_path: str | Path,
    task: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    *,
    data_format: str | None = None,
    validation_fraction: float = 0.2,
    seed: int = 42,
    image_size: int = 224,
    image_root: str | Path | None = None,
    label_columns: list[str] | None = None,
    augmentation_enabled: bool = False,
) -> dict[str, Any]:
    """Build the GUI's input mapping using the shared UATF configuration schema."""
    selected_format = data_format or (
        "image_folder" if task == "classification" else "csv_multilabel"
    )
    dataset: dict[str, Any] = {
        "format": selected_format,
        "path": str(dataset_path),
        "validation_fraction": validation_fraction,
        "image_size": image_size,
        "mean": list(_NORMALIZE_MEAN),
        "std": list(_NORMALIZE_STD),
        "augmentation": {"enabled": augmentation_enabled},
    }
    if image_root is not None:
        dataset["image_dir"] = str(image_root)
    if label_columns:
        dataset["label_columns"] = list(label_columns)
    return {
        "task": {"type": task},
        "dataset": dataset,
        "model": {"name": model_name, "pretrained": False},
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "num_workers": 0,
            "seed": seed,
        },
        "evaluation": {"primary_metric": "accuracy" if task == "classification" else "micro_f1"},
        "output": {"directory": str(OUTPUT_ROOT)},
    }


def refresh_experiment_browser(
    name_filter: str | None = "",
    model_filter: str | None = "",
    status_filter: str | None = "",
    created_after: str | None = "",
    created_before: str | None = "",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Refresh the GUI's persistent experiment list and selection controls."""
    name = name_filter.strip() if isinstance(name_filter, str) else ""
    model = model_filter.strip() if isinstance(model_filter, str) else ""
    after = created_after.strip() if isinstance(created_after, str) else ""
    before = created_before.strip() if isinstance(created_before, str) else ""
    rows = list_experiments(
        OUTPUT_ROOT,
        name=name or None,
        model=model or None,
        status=status_filter or None,
        created_after=after or None,
        created_before=before or None,
    )
    columns = [
        "id",
        "name",
        "created_at_utc",
        "status",
        "task",
        "model",
        "dataset_id",
        "best_metric",
        "duration_seconds",
        "legacy",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    choices = [
        (
            f"{row['name']} | {row['status']} | {row.get('created_at_utc') or '时间未知'}",
            row["id"],
        )
        for row in rows
    ]
    selected = choices[0][1] if choices else None
    return frame, gr.update(choices=choices, value=selected), gr.update(choices=choices, value=[])


def _history_chart_frame(history: list[dict[str, Any]], primary_metric: str | None) -> pd.DataFrame:
    series = ["train_loss", "val_loss"]
    if primary_metric:
        series.append(f"val_{primary_metric}")
    rows = [
        {"epoch": item["epoch"], "series": key, "value": item[key]}
        for item in history
        for key in series
        if key in item
    ]
    return pd.DataFrame(rows, columns=["epoch", "series", "value"])


def load_experiment_details(experiment_id: str | None) -> tuple[Any, ...]:
    if not experiment_id:
        return {}, {}, _chart_frame([]), "", pd.DataFrame(), None, None
    detail = get_experiment(OUTPUT_ROOT, experiment_id)
    record = detail["record"]
    metrics = detail["metrics"] or {}
    primary_metric = record.get("evaluation", {}).get("primary_metric")
    if not primary_metric:
        primary_metric = (detail["configuration"] or {}).get("evaluation", {}).get("primary_metric")
    chart = _history_chart_frame(detail["history"], primary_metric)
    checkpoint_rows = [
        {
            "kind": item["kind"],
            "epoch": item.get("epoch"),
            "saved_at_utc": item["saved_at_utc"],
            "size_bytes": item["size_bytes"],
            "supports_resume": item["supports_resume"],
            "inference_only": item["inference_only"],
            "path": item["path"],
        }
        for item in detail["checkpoints"]
    ]
    checkpoint_files = {item["kind"]: item["path"] for item in detail["checkpoints"]}
    return (
        record,
        metrics,
        chart,
        detail["log"],
        pd.DataFrame(checkpoint_rows),
        checkpoint_files.get("best"),
        checkpoint_files.get("latest"),
    )


def compare_selected_experiments(experiment_ids: list[str] | None) -> dict[str, Any]:
    if not experiment_ids:
        return {"comparable": False, "warnings": ["请选择至少两个实验。"], "experiments": []}
    try:
        return compare_experiments(OUTPUT_ROOT, experiment_ids)
    except (OSError, ValueError) as error:
        return {"comparable": False, "warnings": [str(error)], "experiments": []}


def copy_experiment_configuration(experiment_id: str | None) -> tuple[str, str]:
    if not experiment_id:
        return "", "请选择一个实验。"
    try:
        detail = get_experiment(OUTPUT_ROOT, experiment_id)
    except (OSError, ValueError) as error:
        return "", f"读取配置失败：{error}"
    config = detail["configuration"]
    if not config:
        return "", "该历史目录没有可复制的生效配置。"
    return yaml.safe_dump(
        config, sort_keys=False, allow_unicode=True
    ), "配置已载入，可编辑后启动新实验。"


def launch_copied_configuration(configuration_text: str | None) -> tuple[str, str]:
    """Start a fresh CLI task from the copied YAML while retaining its settings."""
    try:
        configuration = yaml.safe_load(configuration_text or "")
        if not isinstance(configuration, dict):
            raise ValueError("配置内容必须是 YAML mapping。")
        output = configuration.setdefault("output", {})
        if not isinstance(output, dict):
            raise ValueError("配置中的 output 必须是 mapping。")
        output["directory"] = str(OUTPUT_ROOT)
        output.pop("experiment_name", None)
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        active_task = get_latest_task(OUTPUT_ROOT)
        if active_task and active_task.get("status") in ACTIVE_STATUSES:
            raise RuntimeError(
                f"已有训练任务 {active_task['task_id']} 处于 {active_task['status']} 状态。"
            )
        task_id = uuid.uuid4().hex
        config_path = CONFIG_ROOT / f"{task_id}.yaml"
        log_path = CONFIG_ROOT / f"{task_id}.log"
        config_path.write_text(
            yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "uatf.cli",
            "--config",
            str(config_path),
            "--task-id",
            task_id,
        ]
        with log_path.open("w", encoding="utf-8") as log_output:
            subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_output,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        return f"已启动新训练任务 {task_id}，任务状态可在右侧训练面板查询。", task_id
    except Exception as error:
        return f"启动失败：{error}", ""


def refresh_model_asset_browser() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    assets = list_model_assets(OUTPUT_ROOT)
    columns = [
        "id",
        "name",
        "architecture",
        "task",
        "experiment_id",
        "weights_path",
        "created_at_utc",
        "available",
        "weights_kind",
        "version",
    ]
    choices = [(f"{item.get('name')} | {item.get('architecture')}", item["id"]) for item in assets]
    choices_update = gr.update(choices=choices, value=choices[0][1] if choices else None)
    return pd.DataFrame(assets, columns=columns), choices_update, choices_update


def _evaluation_format_choices(asset_id: str | None) -> dict[str, Any]:
    asset = load_model_asset_record(asset_id)
    if asset.get("task") == "multi_label":
        return gr.update(choices=["csv_multilabel"], value="csv_multilabel")
    selected_format = asset.get("dataset_format")
    if selected_format not in {"image_folder", "csv_classification"}:
        selected_format = "image_folder"
    return gr.update(choices=["image_folder", "csv_classification"], value=selected_format)


def load_model_asset_ui(asset_id: str | None) -> tuple[Any, str]:
    if not asset_id:
        return None, "请选择一个模型资产。"
    try:
        model = load_model_asset(OUTPUT_ROOT, asset_id)
    except (OSError, RuntimeError, ValueError) as error:
        return None, f"模型加载失败：{error}"
    return model, f"已加载模型资产 {asset_id}，架构：{type(model).__name__}。"


def load_model_asset_record(asset_id: str | None) -> dict[str, Any]:
    if not asset_id:
        return {}
    return next(
        (asset for asset in list_model_assets(OUTPUT_ROOT) if asset.get("id") == asset_id),
        {},
    )


def export_model_asset_ui(asset_id: str | None) -> tuple[str | None, str | None, str]:
    if not asset_id:
        return None, None, "请选择一个模型资产。"
    try:
        path = export_model_asset(OUTPUT_ROOT, asset_id)
    except (OSError, RuntimeError, ValueError) as error:
        return None, None, f"模型导出失败：{error}"
    sidecar = path.with_suffix(path.suffix + ".json")
    return str(path), str(sidecar), f"模型权重和结构/预处理配置已导出：{path}"


def _evaluation_rows(result: EvaluationResult) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_view: list[dict[str, Any]] = []
    error_view: list[dict[str, Any]] = []
    dataset_metadata = (result.metadata or {}).get("dataset", {})
    root = (
        Path(dataset_metadata.get("image_root", "")) if dataset_metadata.get("image_root") else None
    )
    for index, (sample_id, relative_path) in enumerate(
        zip(result.sample_ids, result.relative_paths, strict=True)
    ):
        row: dict[str, Any] = {"sample_id": sample_id, "relative_path": relative_path}
        if root is not None and relative_path:
            row["_image_path"] = str((root / Path(relative_path)).resolve())
        if result.task == "classification":
            target_index = int(result.targets[index])
            prediction_index = int(result.predictions[index])
            target_name = result.class_names[target_index]
            prediction_name = result.class_names[prediction_index]
            row.update(
                {
                    "target": target_name,
                    "prediction": prediction_name,
                    "confidence": float(result.probabilities[index, prediction_index]),
                    "error_labels": sorted({target_name, prediction_name})
                    if target_index != prediction_index
                    else [],
                }
            )
            row.update(
                {
                    f"probability_{name}": float(result.probabilities[index, class_index])
                    for class_index, name in enumerate(result.class_names)
                }
            )
            row.update(
                {
                    f"score_{name}": float(result.logits[index, class_index])
                    for class_index, name in enumerate(result.class_names)
                }
            )
            if target_index != prediction_index:
                error_view.append(row)
        else:
            target_values = result.targets[index].tolist()
            prediction_values = result.predictions[index].tolist()
            true_labels = [
                name
                for name, enabled in zip(result.class_names, target_values, strict=True)
                if enabled
            ]
            predicted_labels = [
                name
                for name, enabled in zip(result.class_names, prediction_values, strict=True)
                if enabled
            ]
            row.update(
                {
                    "target_labels": ", ".join(true_labels),
                    "prediction_labels": ", ".join(predicted_labels),
                    "error_labels": [
                        name
                        for name, actual, predicted in zip(
                            result.class_names, target_values, prediction_values, strict=True
                        )
                        if actual != predicted
                    ],
                }
            )
            row.update(
                {
                    f"probability_{name}": float(result.probabilities[index, class_index])
                    for class_index, name in enumerate(result.class_names)
                }
            )
            row.update(
                {
                    f"score_{name}": float(result.logits[index, class_index])
                    for class_index, name in enumerate(result.class_names)
                }
            )
            if row["error_labels"]:
                error_view.append(row)
        prediction_view.append(row)
    return prediction_view, error_view


def run_asset_evaluation_ui(
    asset_id: str | None,
    dataset_path: str | None,
    data_format: str | None,
    image_root: str | None,
    batch_size: float,
    threshold: float,
) -> tuple[Any, ...]:
    if not asset_id or not dataset_path:
        return (
            None,
            "请选择模型资产和带标签的数据集。",
            {},
            pd.DataFrame(),
            None,
            None,
            None,
            pd.DataFrame(),
            gr.update(choices=[]),
            gr.update(choices=[]),
            pd.DataFrame(),
        )
    try:
        if isinstance(batch_size, bool) or not float(batch_size).is_integer() or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        result = evaluate_model_asset(
            OUTPUT_ROOT,
            asset_id,
            dataset_path,
            data_format=data_format or None,
            image_root=image_root or None,
            batch_size=int(batch_size),
            threshold=threshold,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return (
            None,
            f"独立评估失败：{error}",
            {},
            pd.DataFrame(),
            None,
            None,
            None,
            pd.DataFrame(),
            gr.update(choices=[]),
            gr.update(choices=[]),
            pd.DataFrame(),
        )

    metric_rows = []
    for name, values in result.metrics["per_class"].items():
        metric_rows.append({"class_or_label": name, **values})
    prediction_rows, error_rows = _evaluation_rows(result)
    sample_choices = [row["sample_id"] for row in error_rows]
    filter_choices = ["全部", *result.class_names]
    artifacts = result.artifacts or {}

    def chart(key: str) -> str | None:
        return str(artifacts[key]) if key in artifacts else None

    report = {
        "task": result.task,
        "sample_count": result.metrics.get("sample_count"),
        "loss": result.loss,
        "metrics": result.metrics,
        "warnings": result.warnings,
        "metadata": result.metadata,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    state = {"predictions": prediction_rows, "errors": error_rows, "task": result.task}
    message = f"评估完成：{result.metadata['dataset']['sample_count']} 个样本。结果目录：{artifacts['metrics'].parent}"
    return (
        state,
        message,
        report,
        pd.DataFrame(metric_rows),
        chart("confusion_matrix"),
        chart("roc_curve"),
        chart("precision_recall_curve"),
        pd.DataFrame(
            [
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_") and key != "error_labels"
                }
                for row in prediction_rows
            ]
        ),
        gr.update(choices=filter_choices, value="全部"),
        gr.update(choices=sample_choices, value=sample_choices[0] if sample_choices else None),
        pd.DataFrame(
            [
                {key: value for key, value in row.items() if not key.startswith("_")}
                for row in error_rows
            ]
        ),
    )


def filter_evaluation_errors(
    state: dict[str, Any] | None, class_filter: str | None
) -> tuple[Any, Any]:
    if not state:
        return pd.DataFrame(), gr.update(choices=[], value=None)
    rows = state.get("errors", [])
    if class_filter and class_filter != "全部":
        rows = [row for row in rows if class_filter in row.get("error_labels", [])]
    choices = [row["sample_id"] for row in rows]
    visible = [
        {key: value for key, value in row.items() if not key.startswith("_")} for row in rows
    ]
    return pd.DataFrame(visible), gr.update(choices=choices, value=choices[0] if choices else None)


def preview_evaluation_error(
    sample_id: str | None, state: dict[str, Any] | None
) -> tuple[str | None, str]:
    if not sample_id or not state:
        return None, ""
    row = next((item for item in state.get("errors", []) if item["sample_id"] == sample_id), None)
    if row is None:
        return None, ""
    details = {key: value for key, value in row.items() if not key.startswith("_")}
    return row.get("_image_path"), json.dumps(details, ensure_ascii=False, indent=2)


def run_asset_image_prediction_ui(
    asset_id: str | None, image_path: str | None, threshold: float
) -> tuple[str, pd.DataFrame, str]:
    if not asset_id or not image_path:
        return "请选择模型资产和图片。", pd.DataFrame(), ""
    try:
        result = predict_model_asset_image(OUTPUT_ROOT, asset_id, image_path, threshold=threshold)
    except (OSError, RuntimeError, ValueError) as error:
        return f"单图推理失败：{error}", pd.DataFrame(), ""
    rows = [
        {"class_or_label": name, "logit_score": score, "probability": result["probabilities"][name]}
        for name, score in result["scores"].items()
    ]
    decision = result["prediction"]
    message = (
        f"最终分类结果：{decision}"
        if isinstance(decision, str)
        else f"预测标签：{', '.join(decision) or '无'}"
    )
    return message, pd.DataFrame(rows), json.dumps(result, ensure_ascii=False, indent=2)


def run_asset_directory_prediction_ui(
    asset_id: str | None, image_directory: str | None, threshold: float, batch_size: float
) -> tuple[str, pd.DataFrame, str | None]:
    if not asset_id or not image_directory:
        return "请选择模型资产和图片目录。", pd.DataFrame(), None
    try:
        if isinstance(batch_size, bool) or not float(batch_size).is_integer() or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        result, csv_path = predict_model_asset_directory(
            OUTPUT_ROOT, asset_id, image_directory, threshold=threshold, batch_size=int(batch_size)
        )
    except (OSError, RuntimeError, ValueError) as error:
        return f"批量推理失败：{error}", pd.DataFrame(), None
    rows = prediction_rows(result)
    frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key not in {"scores", "probabilities"}}
            for row in rows
        ]
    )
    return f"批量推理完成：{len(rows)} 张图片。", frame, str(csv_path)


def _archive_digest(archive_path: Path) -> str:
    digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_cache_index() -> dict[str, str]:
    index_path = UPLOAD_ROOT / ".import_index.json"
    try:
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _archive_source_key(archive_path: Path) -> str:
    info = archive_path.stat()
    source = f"{archive_path}|{info.st_size}|{info.st_mtime_ns}|{info.st_ctime_ns}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _save_archive_cache_index(index: dict[str, str]) -> None:
    index_path = UPLOAD_ROOT / ".import_index.json"
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2), encoding="utf-8")
    temporary.replace(index_path)


def _safe_extract_dataset(
    upload_path: str,
    task: str,
    data_format: str | None = None,
) -> Path:
    if not upload_path:
        raise ValueError("请先上传 ZIP 格式的数据集。")
    archive_path = Path(upload_path).expanduser().resolve()
    if archive_path.suffix.lower() != ".zip":
        raise ValueError("数据集必须是 ZIP 文件。")
    if not archive_path.is_file():
        raise ValueError(f"ZIP 文件不存在：{archive_path}")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    source_key = _archive_source_key(archive_path)
    cache_index = _archive_cache_index()
    digest = cache_index.get(source_key)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        digest = _archive_digest(archive_path)
    destination = UPLOAD_ROOT / digest
    complete_marker = UPLOAD_ROOT / f".{digest}.complete"
    if destination.is_dir() and complete_marker.is_file():
        if cache_index.get(source_key) != digest:
            cache_index[source_key] = digest
            _save_archive_cache_index(cache_index)
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
            seen_names: set[str] = set()
            advertised_size = 0
            for info in archive.infolist():
                normalized = info.filename.replace("\\", "/")
                path = PurePosixPath(normalized)
                parts = tuple(part for part in path.parts if part not in {"", "."})
                has_reserved_name = any(
                    part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES for part in parts
                )
                if (
                    path.is_absolute()
                    or ".." in parts
                    or any(
                        ":" in part or "\x00" in part or part.endswith((" ", ".")) for part in parts
                    )
                    or has_reserved_name
                ):
                    raise ValueError("ZIP 中包含不安全的文件路径。请重新打包数据集。")
                if not parts:
                    continue
                key = "/".join(parts).casefold()
                if key in seen_names:
                    raise ValueError(f"ZIP 中包含重复路径：{'/'.join(parts)}")
                seen_names.add(key)
                mode = info.external_attr >> 16
                file_type = S_IFMT(mode)
                if S_ISLNK(mode) or file_type not in {0, S_IFREG, S_IFDIR}:
                    raise ValueError("ZIP 中不能包含符号链接或特殊文件。请重新打包数据集。")
                if info.flag_bits & 0x1:
                    raise ValueError("ZIP 中的加密文件不受支持。")
                if info.file_size < 0:
                    raise ValueError("ZIP 文件条目包含无效的文件大小。")
                advertised_size += info.file_size
                if advertised_size > MAX_ZIP_UNCOMPRESSED_BYTES:
                    raise ValueError("ZIP 解压后超过 20 GiB 的安全限制。")
                members.append((info, parts))
            if len(members) > MAX_ZIP_MEMBERS:
                raise ValueError("ZIP 文件数量超过 50,000 个的安全限制。")
            if not members:
                raise ValueError("ZIP 文件中没有可导入的内容。")

            copied_bytes = 0
            for info, parts in members:
                output_path = destination.joinpath(*parts)
                if not output_path.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("ZIP 中包含不安全的文件路径。请重新打包数据集。")
                if info.is_dir():
                    output_path.mkdir(parents=True, exist_ok=True)
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, output_path.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        copied_bytes += len(chunk)
                        if copied_bytes > MAX_ZIP_UNCOMPRESSED_BYTES:
                            raise ValueError("ZIP 解压内容超过 20 GiB 的安全限制。")
                        output.write(chunk)
            if copied_bytes != sum(info.file_size for info, _ in members if not info.is_dir()):
                raise ValueError("ZIP 文件实际解压大小与目录信息不一致。")
        complete_marker.write_text("complete", encoding="ascii")
        cache_index[source_key] = digest
        _save_archive_cache_index(cache_index)
    except zipfile.BadZipFile as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise ValueError("无法读取 ZIP 文件，请确认文件没有损坏。") from error
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    _ = task, data_format
    return destination


def _dataset_identity(path: str, data_format: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"uatf:{data_format}:{Path(path).resolve()}"))


def _register_dataset(inspection: DatasetInspection) -> str:
    record_id = _dataset_identity(str(inspection.path), inspection.format)
    DATASET_REGISTRY.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset_id": record_id,
        "path": str(inspection.path),
        "format": inspection.format,
        "sample_count": inspection.sample_count,
        "valid_sample_count": inspection.valid_sample_count,
        "labels": inspection.class_names,
        "class_counts": inspection.class_counts,
        "split_strategy": inspection.split_strategy,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = DATASET_REGISTRY / f"{record_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return record_id


def _report_fingerprint(report: dict[str, Any]) -> str:
    manifest = {
        key: report[key]
        for key in (
            "path",
            "format",
            "sample_count",
            "valid_sample_count",
            "class_names",
            "class_counts",
            "split",
            "errors",
            "warnings",
            "samples",
        )
    }
    encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inspect_selected_dataset(
    upload_path: str | None,
    local_path: str | None,
    task: str,
    data_format: str,
    validation_fraction: float,
    seed: float,
    image_size: float,
) -> tuple[DatasetInspection, dict[str, Any], dict[str, Any]]:
    if upload_path and local_path and local_path.strip():
        raise ValueError("请只选择 ZIP 上传或本地目录路径中的一种数据来源。")
    if upload_path:
        path = _safe_extract_dataset(upload_path, task, data_format)
        source = "zip"
    elif local_path and local_path.strip():
        path = Path(local_path.strip()).expanduser().resolve()
        source = "local"
    else:
        raise ValueError("请上传 ZIP，或填写 UATF 服务端可访问的数据集路径。")

    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(validation_fraction)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("验证集比例必须大于 0 且小于 1。")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, float))
        or not float(seed).is_integer()
        or seed < 0
    ):
        raise ValueError("随机种子必须是非负整数。")
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, (int, float))
        or not float(image_size).is_integer()
        or image_size < 1
    ):
        raise ValueError("图像尺寸必须是正整数。")

    if task == "multi_label" and data_format != "csv_multilabel":
        raise ValueError("多标签分类要求 `csv_multilabel` 格式。")
    if task == "classification" and data_format == "csv_multilabel":
        raise ValueError("单标签分类不能使用 `csv_multilabel` 格式。")

    inspection = inspect_dataset(
        path,
        data_format,
        validation_fraction=float(validation_fraction),
        seed=int(seed),
    )
    report = inspection.to_dict(include_samples=True)
    dataset_id = _register_dataset(inspection)
    report["dataset_id"] = dataset_id
    state = {
        "source": source,
        "upload_path": upload_path,
        "local_path": local_path,
        "task": task,
        "format": data_format,
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "image_size": int(image_size),
        "dataset_path": str(inspection.path),
        "dataset_id": dataset_id,
        "fingerprint": _report_fingerprint(report),
        "report": report,
    }
    return inspection, report, state


def _save_distribution_chart(report: dict[str, Any]) -> str | None:
    rows = report["class_counts"]
    if not rows:
        return None
    matplotlib_config = DATASET_ROOT / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for family in (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "PingFang SC",
    ):
        try:
            font_path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=font_path).get_name()]
        break
    plt.rcParams["axes.unicode_minus"] = False

    chart_directory = DATASET_ROOT / report["dataset_id"]
    chart_directory.mkdir(parents=True, exist_ok=True)
    chart_path = chart_directory / "class_distribution.png"
    labels = [row["label"] for row in rows]
    positions = list(range(len(rows)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 0.55), 3.8))
    axis.bar(
        [position - width / 2 for position in positions],
        [row["train_count"] for row in rows],
        width,
        label="Train",
        color="#397c68",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [row["validation_count"] for row in rows],
        width,
        label="Validation",
        color="#d17a43",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Samples" if report["format"] != "csv_multilabel" else "Positive labels")
    axis.set_xlabel("Label")
    axis.tick_params(axis="x", labelrotation=35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(chart_path, dpi=120)
    plt.close(figure)
    return str(chart_path)


def _class_distribution_rows(report: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            row["label"],
            row["count"],
            f"{row['ratio']:.1%}",
            row["train_count"],
            row["validation_count"],
        ]
        for row in report["class_counts"]
    ]


def _preview_at(report: dict[str, Any], index: int, image_size: int) -> tuple[Any, Any, str, int]:
    samples = report["samples"]
    if not samples:
        return None, None, "没有可预览的有效图片。", 0
    index = index % len(samples)
    sample = samples[index]
    with Image.open(sample["path"]) as source:
        original = source.convert("RGB")
    transform = build_image_transform(
        image_size=image_size,
        mean=_NORMALIZE_MEAN,
        std=_NORMALIZE_STD,
    )
    normalized = transform(original)
    mean = torch.tensor(_NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(_NORMALIZE_STD).view(3, 1, 1)
    processed = ToPILImage()((normalized * std + mean).clamp(0, 1))
    label = json.dumps(sample["label"], ensure_ascii=False)
    information = f"样本 {index + 1} / {len(samples)}\n标签：{label}\n路径：{sample['path']}"
    return original, processed, information, index


def _inspection_summary(report: dict[str, Any]) -> str:
    status = {"ready": "检查通过", "warning": "检查通过，有警告", "blocked": "存在阻断错误"}[
        report["status"]
    ]
    split = report["split"]
    return (
        f"**{status}**  ·  数据集 ID `{report['dataset_id']}`\n\n"
        f"路径：`{report['path']}`  \n"
        f"格式：`{report['format']}`  ·  样本：{report['sample_count']}（有效 {report['valid_sample_count']}）  ·  "
        f"标签：{len(report['class_names'])}  \n"
        f"划分：{split['strategy']}  ·  训练 {split['train_count']} / 验证 {split['validation_count']}  ·  "
        f"验证比例 {split['validation_fraction']:.0%}  ·  随机种子 {split['seed']}"
    )


def inspect_dataset_ui(
    upload_path: str | None,
    local_path: str | None,
    task: str,
    data_format: str,
    validation_fraction: float,
    seed: float,
    image_size: float,
) -> tuple[Any, ...]:
    try:
        inspection, report, state = _inspect_selected_dataset(
            upload_path, local_path, task, data_format, validation_fraction, seed, image_size
        )
        initial_index = (
            random.Random(state["seed"]).randrange(len(report["samples"]))
            if report["samples"]
            else 0
        )
        original, processed, information, initial_index = _preview_at(
            report, initial_index, state["image_size"]
        )
        state["sample_index"] = initial_index
        return (
            state,
            initial_index,
            _inspection_summary(report),
            {key: value for key, value in report.items() if key != "samples"},
            _class_distribution_rows(report),
            _save_distribution_chart(report),
            original,
            processed,
            information,
            gr.update(value=False),
            gr.update(visible=inspection.can_train),
        )
    except Exception as error:
        return (
            None,
            0,
            f"数据集导入/检查失败：{error}",
            {"status": "blocked", "errors": [str(error)]},
            [],
            None,
            None,
            None,
            "",
            gr.update(value=False),
            gr.update(visible=False),
        )


def turn_sample(
    state: dict[str, Any] | None, sample_index: float, direction: int
) -> tuple[Any, ...]:
    if not state:
        return None, None, "请先检查数据集。", 0
    report = state["report"]
    index = int(sample_index) + direction
    original, processed, information, index = _preview_at(report, index, state["image_size"])
    state["sample_index"] = index
    return original, processed, information, index


def _friendly_error(output: str, return_code: int) -> str:
    error_lines = [
        line.removeprefix("Error: ").strip()
        for line in output.splitlines()
        if line.startswith("Error:")
    ]
    detail = error_lines[-1] if error_lines else f"训练进程退出，代码 {return_code}。"
    lowered = detail.lower()
    if "cuda out of memory" in lowered or "out of memory" in lowered:
        return f"GPU 显存不足。请减小 Batch Size 后重试。{detail}"
    if "cuda was requested but is unavailable" in lowered:
        return "当前 PyTorch 未检测到可用 CUDA GPU。请选择 CPU，或安装支持 CUDA 的 PyTorch。"
    if "unknown model" in lowered:
        return f"所选模型不存在。{detail}"
    if "dataset contains no images" in lowered or "dataset has no" in lowered:
        return f"数据集为空或没有有效图片。{detail}"
    if "dataset csv" in lowered or "csv row" in lowered or "label" in lowered:
        return f"数据集标注或标签不符合要求。{detail}"
    return f"训练失败：{detail}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "计算中"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def _event_timestamp(value: str | None) -> str:
    if not value:
        return "尚未计算"
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "尚未计算"


def _read_log_tail(path: Path, maximum_bytes: int = 64 * 1024) -> str:
    try:
        with path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - maximum_bytes))
            return source.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _technical_error(log_text: str, output_lines: list[str]) -> str:
    marker = "Traceback (most recent call last):"
    traceback_index = log_text.rfind(marker)
    if traceback_index >= 0:
        return log_text[traceback_index:].strip()
    technical_lines = [line for line in output_lines if line.strip()]
    if technical_lines:
        return "\n".join(technical_lines[-80:])
    return "训练进程未提供技术堆栈。"


def _chart_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["epoch", "series", "value"])


def refresh_task_status(task_id: str | None = None) -> tuple[Any, ...]:
    """Restore a persisted task summary after a page reload or app restart."""
    record = get_task(OUTPUT_ROOT, task_id) if task_id else get_latest_task(OUTPUT_ROOT)
    if record is None:
        return ("没有可恢复的训练任务", 0, "-", "", None, "", "", "")

    status = str(record.get("status", "未知"))
    labels = {
        "Pending": "等待执行",
        "Preparing": "准备数据和模型",
        "Running": "训练中",
        "Validating": "验证中",
        "Stopping": "正在停止（将在当前 Epoch 完成后生效）",
        "Stopped": "已停止",
        "Completed": "训练完成",
        "Failed": "训练失败或进程中断",
    }
    epoch = record.get("epoch")
    total_epochs = record.get("total_epochs")
    epoch_label = f"{epoch or 0} / {total_epochs}" if total_epochs else "-"
    progress_value = 0.0
    if total_epochs:
        completed_fraction = max(0, (int(epoch or 0) - 1)) / max(int(total_epochs), 1)
        if status == "Validating":
            completed_fraction += 0.5 / max(int(total_epochs), 1)
        elif status in {"Stopped", "Completed"} and epoch:
            completed_fraction = int(epoch) / max(int(total_epochs), 1)
        progress_value = min(100.0, completed_fraction * 100)
    result_directory = record.get("result_directory")
    result_path = Path(result_directory) if result_directory else None
    log_path = result_path / "logs" / "train.log" if result_path else None
    logs = _read_log_tail(log_path, maximum_bytes=16 * 1024) if log_path else ""
    error_summary = str(record.get("error_summary") or "")
    detail = logs if error_summary else ""
    stage = record.get("stage")
    status_text = f"{labels.get(status, status)} | 任務 {record['task_id']}"
    if stage:
        status_text += f" | {stage}"
    if result_directory:
        status_text += f" | {result_directory}"
    return (
        status_text,
        progress_value,
        epoch_label,
        logs,
        str(log_path) if log_path and log_path.is_file() else None,
        error_summary,
        detail,
        str(record["task_id"]),
    )


def request_training_stop(task_id: str | None) -> str:
    """Request a cooperative stop for the selected GUI task."""
    selected_id = task_id
    if not selected_id:
        latest = get_latest_task(OUTPUT_ROOT)
        selected_id = str(latest["task_id"]) if latest else None
    if not selected_id:
        return "没有可停止的训练任务"
    record = request_task_stop(OUTPUT_ROOT, selected_id)
    status = record.get("status")
    if status in ACTIVE_STATUSES:
        return f"任务 {selected_id} 正在停止，将在当前 Epoch 完成后停止并保留最近检查点。"
    return f"任务 {selected_id} 当前状态：{status}"


def _safe_resource_text(sampler: HardwareResourceSampler) -> str:
    try:
        return format_resource_snapshot(sampler.snapshot())
    except Exception:
        return "资源监控暂不可用。训练将继续运行。"


def start_training(
    upload_path: str | None,
    local_path: str | None,
    dataset_state: dict[str, Any] | None,
    task: str,
    data_format: str,
    validation_fraction: float,
    seed: float,
    image_size: float,
    accept_warnings: bool,
    model_name: str,
    epochs: float,
    batch_size: float,
    learning_rate: float,
    device: str,
    augmentation_enabled: bool = False,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> Iterator[tuple[Any, ...]]:
    started_at = time.perf_counter()
    logs: deque[str] = deque(maxlen=200)
    stdout_tail: deque[str] = deque(maxlen=1000)
    stdout_queue: queue.Queue[str] = queue.Queue(maxsize=2000)
    experiment_path: Path | None = None
    log_path: Path | None = None
    resource_sampler = HardwareResourceSampler()
    current_epoch = "-"
    current_batch = "尚未计算"
    current_train_loss = "尚未计算"
    current_validation_loss = "尚未计算"
    current_metric = "尚未计算"
    best_metric = "尚未计算"
    best_epoch = "尚未计算"
    learning_rate_value = "尚未计算"
    epoch_duration = "尚未计算"
    overall_progress = 0.0
    metric_updated_at = "尚未计算"
    evaluation_report: dict[str, Any] = {}
    model_file: str | None = None
    task_id: str | None = None
    loss_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    learning_rate_rows: list[dict[str, Any]] = []
    primary_key = "accuracy" if task == "classification" else "micro_f1"
    completed_primary_values: list[tuple[int, float]] = []
    initial_values: tuple[Any, ...] = (
        "检查并准备数据集",
        0,
        "-",
        "尚未计算",
        "0:00",
        "计算中",
        "尚未计算",
        "尚未计算",
        "尚未计算",
        "尚未计算",
        "尚未计算",
        "尚未计算",
        "尚未计算",
        "尚未计算",
        _chart_frame(loss_rows),
        _chart_frame(metric_rows),
        _chart_frame(learning_rate_rows),
        "",
        None,
        "",
        "",
        {},
        _safe_resource_text(resource_sampler),
        None,
        "",
    )

    def changes_tuple(changes: dict[int, Any]) -> tuple[Any, ...]:
        if task_id is not None:
            changes[24] = task_id
        return tuple(changes.get(index, gr.skip()) for index in range(len(initial_values)))

    try:
        if (
            not isinstance(epochs, (int, float))
            or isinstance(epochs, bool)
            or not float(epochs).is_integer()
            or epochs < 1
        ):
            raise ValueError("Epochs 必须是大于 0 的整数。")
        if (
            not isinstance(batch_size, (int, float))
            or isinstance(batch_size, bool)
            or not float(batch_size).is_integer()
            or batch_size < 1
        ):
            raise ValueError("Batch Size 必须是大于 0 的整数。")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("Learning Rate 必须是大于 0 的有限数值。")
        if task not in {"classification", "multi_label"}:
            raise ValueError("请选择有效的任务类型。")
        if dataset_state is None:
            raise ValueError("请先导入并检查数据集，再启动训练。")
        expected_settings = {
            "task": task,
            "format": data_format,
            "validation_fraction": float(validation_fraction),
            "seed": int(seed),
            "image_size": int(image_size),
        }
        if any(dataset_state.get(key) != value for key, value in expected_settings.items()):
            raise ValueError("数据集或划分设置已更改，请重新检查数据集后再训练。")
        expected_source = "zip" if upload_path else "local"
        if dataset_state.get("source") != expected_source:
            raise ValueError("数据集来源已更改，请重新检查数据集后再训练。")
        progress(0, desc="检查并准备数据集")
        inspection, fresh_report, fresh_state = _inspect_selected_dataset(
            upload_path,
            local_path,
            task,
            data_format,
            validation_fraction,
            seed,
            image_size,
        )
        if fresh_state["fingerprint"] != dataset_state.get("fingerprint"):
            raise ValueError("数据集内容或检查结果已变化，请重新检查并确认数据集。")
        if fresh_report["errors"]:
            raise ValueError(
                "数据集存在阻断性错误，不能启动训练：\n" + "\n".join(fresh_report["errors"])
            )
        if fresh_report["warnings"] and not accept_warnings:
            raise ValueError("检查报告包含警告。请阅读报告并勾选确认后再启动训练。")
        if not inspection.can_train:
            raise ValueError("数据集未通过检查，不能启动训练。")

        active_task = get_latest_task(OUTPUT_ROOT)
        if active_task and active_task.get("status") in ACTIVE_STATUSES:
            task_id = str(active_task["task_id"])
            raise ValueError(
                f"已有训练任务 {task_id} 处于 {active_task['status']} 状态；"
                "请等待其结束或请求停止。"
            )

        dataset_path = (
            Path(fresh_report["annotation_csv"])
            if data_format in {"csv_classification", "csv_multilabel"}
            else inspection.path
        )
        run_id = uuid.uuid4().hex
        task_id = run_id
        config_path = CONFIG_ROOT / f"{run_id}.yaml"
        events_path = CONFIG_ROOT / f"{run_id}.events.jsonl"
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        config = build_training_config(
            dataset_path,
            task,
            model_name,
            int(epochs),
            int(batch_size),
            float(learning_rate),
            device,
            data_format=data_format,
            validation_fraction=float(validation_fraction),
            seed=int(seed),
            image_size=int(image_size),
            image_root=fresh_report["image_root"]
            if data_format in {"csv_classification", "csv_multilabel"}
            else None,
            augmentation_enabled=augmentation_enabled,
        )
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        events_path.write_text("", encoding="utf-8")
        initial_values = (*initial_values[:-1], task_id)
        yield initial_values

        command = [
            sys.executable,
            "-m",
            "uatf.cli",
            "--config",
            str(config_path),
            "--events-file",
            str(events_path),
            "--task-id",
            task_id,
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        progress(0, desc="启动训练")
        assert process.stdout is not None

        def read_process_output() -> None:
            for output_line in process.stdout:
                line = output_line.rstrip("\r\n")
                stdout_tail.append(line)
                try:
                    stdout_queue.put_nowait(line)
                except queue.Full:
                    try:
                        stdout_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        stdout_queue.put_nowait(line)
                    except queue.Full:
                        pass

        output_reader = threading.Thread(target=read_process_output, daemon=True)
        output_reader.start()
        resource_sampler.attach_process(process.pid)

        def update_metrics(metrics: dict[str, Any], timestamp: str | None) -> dict[int, Any]:
            nonlocal current_train_loss, current_validation_loss, current_metric
            nonlocal learning_rate_value, metric_updated_at
            changes: dict[int, Any] = {}
            if "train_loss" in metrics:
                current_train_loss = f"{float(metrics['train_loss']):.4f}"
                changes[7] = current_train_loss
            if "val_loss" in metrics:
                current_validation_loss = f"{float(metrics['val_loss']):.4f}"
                changes[8] = current_validation_loss
            primary_value = metrics.get(f"val_{primary_key}")
            if primary_value is not None:
                current_metric = f"{primary_key}: {float(primary_value):.4f}"
                changes[9] = current_metric
            if "learning_rate" in metrics:
                learning_rate_value = f"{float(metrics['learning_rate']):.6g}"
                changes[12] = learning_rate_value
            if changes:
                metric_updated_at = _event_timestamp(timestamp)
                changes[6] = metric_updated_at
            return changes

        def progress_values(
            epoch: int, total_epochs: int, stage: str, batch_index: int, total_batches: int | None
        ) -> float:
            within_epoch = 0.0
            if total_batches and total_batches > 0:
                fraction = min(1.0, batch_index / total_batches)
                within_epoch = 0.5 * fraction if stage == "training" else 0.5 + 0.5 * fraction
            return max(0.0, min(1.0, (epoch - 1 + within_epoch) / max(total_epochs, 1)))

        last_log_offset = 0
        event_buffer = ""
        last_yield_at = time.perf_counter()
        last_resource_at = 0.0
        return_code: int | None = None
        monitor_error: str | None = None
        with events_path.open("r", encoding="utf-8") as event_source:
            try:
                while True:
                    changes: dict[int, Any] = {}
                    should_yield = False
                    event_buffer += event_source.read()
                    while "\n" in event_buffer:
                        event_line, event_buffer = event_buffer.split("\n", 1)
                        if not event_line:
                            continue
                        try:
                            event = json.loads(event_line)
                        except json.JSONDecodeError:
                            continue
                        event_type = event.get("event_type")
                        if event_type == "training_started":
                            experiment_path = Path(event["experiment_dir"])
                            log_path = experiment_path / "logs" / "train.log"
                            changes.update(
                                {
                                    0: "准备训练组件",
                                    18: str(log_path),
                                }
                            )
                            should_yield = True
                        elif event_type == "training_progress":
                            epoch = int(event.get("epoch") or 0)
                            total = int(event.get("total_epochs") or epochs)
                            stage = str(event.get("stage") or "training")
                            batch_index = int(event.get("batch_index") or 0)
                            total_batches = event.get("total_batches")
                            total_batches = (
                                int(total_batches) if total_batches is not None else None
                            )
                            current_epoch = f"{epoch} / {total}"
                            current_batch = (
                                f"{batch_index} / {total_batches}"
                                if total_batches
                                else f"{batch_index}（总数未知）"
                            )
                            overall_progress = progress_values(
                                epoch, total, stage, batch_index, total_batches
                            )
                            elapsed_seconds = time.perf_counter() - started_at
                            eta_seconds = (
                                elapsed_seconds * (1 - overall_progress) / overall_progress
                                if overall_progress > 0
                                else None
                            )
                            stage_label = "验证中" if stage == "validation" else "训练中"
                            changes.update(
                                {
                                    0: f"{stage_label} | Epoch {current_epoch}",
                                    1: overall_progress * 100,
                                    2: current_epoch,
                                    3: current_batch,
                                    4: _format_duration(elapsed_seconds),
                                    5: _format_duration(eta_seconds),
                                }
                            )
                            changes.update(
                                update_metrics(
                                    dict(event.get("metrics") or {}), event.get("timestamp_utc")
                                )
                            )
                            progress(
                                overall_progress,
                                desc=f"{stage_label} | Epoch {current_epoch} | Batch {current_batch}",
                            )
                            should_yield = True
                        elif event_type == "epoch_completed":
                            epoch = int(event.get("epoch") or 0)
                            total = int(event.get("total_epochs") or epochs)
                            record = dict(event.get("metrics") or {})
                            current_epoch = f"{epoch} / {total}"
                            current_batch = "Epoch 已完成"
                            overall_progress = epoch / max(total, 1)
                            epoch_duration = (
                                _format_duration(float(record["time_seconds"]))
                                if "time_seconds" in record
                                else "尚未计算"
                            )
                            changes.update(
                                {
                                    0: f"训练中 | Epoch {current_epoch}",
                                    1: overall_progress * 100,
                                    2: current_epoch,
                                    3: current_batch,
                                    4: _format_duration(time.perf_counter() - started_at),
                                    13: epoch_duration,
                                }
                            )
                            changes.update(update_metrics(record, event.get("timestamp_utc")))
                            if "train_loss" in record:
                                loss_rows.append(
                                    {
                                        "epoch": epoch,
                                        "series": "Train Loss",
                                        "value": record["train_loss"],
                                    }
                                )
                            if "val_loss" in record:
                                loss_rows.append(
                                    {
                                        "epoch": epoch,
                                        "series": "Validation Loss",
                                        "value": record["val_loss"],
                                    }
                                )
                            primary_value = record.get(f"val_{primary_key}")
                            if primary_value is not None:
                                primary_value = float(primary_value)
                                metric_rows.append(
                                    {"epoch": epoch, "series": primary_key, "value": primary_value}
                                )
                                completed_primary_values.append((epoch, primary_value))
                                best_epoch_number, best_value = max(
                                    completed_primary_values, key=lambda row: row[1]
                                )
                                best_metric = f"{primary_key}: {best_value:.4f}"
                                best_epoch = str(best_epoch_number)
                                changes[10] = best_metric
                                changes[11] = best_epoch
                            if "learning_rate" in record:
                                learning_rate_rows.append(
                                    {
                                        "epoch": epoch,
                                        "series": "Learning Rate",
                                        "value": record["learning_rate"],
                                    }
                                )
                            changes.update(
                                {
                                    14: _chart_frame(loss_rows),
                                    15: _chart_frame(metric_rows),
                                    16: _chart_frame(learning_rate_rows),
                                }
                            )
                            should_yield = True
                        elif event_type == "training_completed":
                            experiment_path = Path(event["experiment_dir"])
                            log_path = experiment_path / "logs" / "train.log"
                            changes.update({18: str(log_path)})
                            should_yield = True
                        elif event_type == "training_failed":
                            experiment_path = (
                                Path(event["experiment_dir"])
                                if event.get("experiment_dir")
                                else experiment_path
                            )
                            if experiment_path is not None:
                                log_path = experiment_path / "logs" / "train.log"
                                changes[18] = str(log_path)
                            changes[0] = "训练失败"
                            should_yield = True
                        elif event_type == "training_stopping":
                            changes[0] = "正在停止（将在当前 Epoch 完成后生效）"
                            should_yield = True
                        elif event_type == "training_stopped":
                            changes[0] = "训练已停止，最近完整检查点已保留"
                            should_yield = True

                    while True:
                        try:
                            line = stdout_queue.get_nowait()
                        except queue.Empty:
                            break
                        if log_path is None:
                            stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                            level = "ERROR" if line.startswith("Error:") else "INFO"
                            logs.append(f"{stamp} {level}: {line}")
                            should_yield = True

                    if log_path is not None and log_path.is_file():
                        try:
                            with log_path.open("r", encoding="utf-8", errors="replace") as source:
                                source.seek(last_log_offset)
                                while line := source.readline():
                                    line = line.rstrip("\r\n")
                                    if line:
                                        logs.append(line)
                                        should_yield = True
                                last_log_offset = source.tell()
                        except OSError:
                            pass

                    now = time.perf_counter()
                    if now - last_resource_at >= 2.0:
                        try:
                            changes[22] = _safe_resource_text(resource_sampler)
                        except Exception:
                            changes[22] = "资源监控暂不可用。训练将继续运行。"
                        last_resource_at = now
                        should_yield = True

                    if should_yield and now - last_yield_at >= 0.75:
                        changes[17] = "\n".join(logs)
                        yield changes_tuple(changes)
                        last_yield_at = time.perf_counter()

                    return_code = process.poll()
                    if return_code is not None:
                        event_buffer += event_source.read()
                        if not event_buffer:
                            break
                    time.sleep(0.2)
            except Exception as error:
                monitor_error = f"GUI 监控读取失败：{error}"
                return_code = process.wait()

        if return_code is None:
            return_code = process.wait()
        output_reader.join(timeout=1.0)
        while True:
            try:
                line = stdout_queue.get_nowait()
            except queue.Empty:
                break
            if log_path is None:
                stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                level = "ERROR" if line.startswith("Error:") else "INFO"
                logs.append(f"{stamp} {level}: {line}")
        if log_path is not None and log_path.is_file():
            try:
                with log_path.open("r", encoding="utf-8", errors="replace") as source:
                    source.seek(last_log_offset)
                    logs.extend(line.rstrip("\r\n") for line in source if line.strip())
            except OSError:
                pass

        task_record = None
        if task_id is not None:
            try:
                task_record = get_task(OUTPUT_ROOT, task_id, reconcile_interruption=False)
            except (OSError, ValueError):
                task_record = None
        if return_code != 0 and task_record and task_record.get("status") == "Stopped":
            checkpoint_path = experiment_path / "last_model.pth" if experiment_path else None
            yield changes_tuple(
                {
                    0: "训练已停止，最近完整检查点已保留",
                    1: overall_progress * 100,
                    2: current_epoch,
                    3: "已停止",
                    4: _format_duration(time.perf_counter() - started_at),
                    17: "\n".join(logs),
                    18: str(log_path) if log_path else None,
                    19: "",
                    20: "",
                    22: _safe_resource_text(resource_sampler),
                    23: (
                        str(checkpoint_path)
                        if checkpoint_path and checkpoint_path.is_file()
                        else None
                    ),
                }
            )
            return

        if return_code != 0:
            summary = _friendly_error("\n".join(stdout_tail), return_code)
            detail = _technical_error(
                _read_log_tail(log_path) if log_path else "", list(stdout_tail)
            )
            if monitor_error:
                detail = f"{monitor_error}\n\n{detail}"
            yield changes_tuple(
                {
                    0: "训练失败",
                    17: "\n".join(logs),
                    18: str(log_path) if log_path else None,
                    19: summary,
                    20: detail,
                    22: _safe_resource_text(resource_sampler),
                }
            )
            return

        if experiment_path is None:
            raise RuntimeError("训练进程已结束，但没有收到结构化的实验结果事件。")
        report_path = experiment_path / "metrics.json"
        if not report_path.is_file():
            raise RuntimeError("训练已结束，但没有生成 metrics.json。")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        evaluation_report = {
            "source": "训练结束后的独立验证集评估，不属于逐 Epoch 训练曲线。",
            "weights": "训练结束时的模型权重（last epoch）",
            "evaluation": report["evaluation"],
        }
        model_file = str(experiment_path / "best_model.pth")
        if not Path(model_file).is_file():
            raise RuntimeError("训练已结束，但没有生成最佳模型 checkpoint。")
        overall_progress = 1.0
        progress(1, desc="训练完成")
        completed_changes: dict[int, Any] = {
            0: "训练完成",
            1: 100,
            2: current_epoch,
            3: "已完成",
            4: _format_duration(time.perf_counter() - started_at),
            5: "0:00",
            6: metric_updated_at,
            7: current_train_loss,
            8: current_validation_loss,
            9: current_metric,
            10: best_metric,
            11: best_epoch,
            12: learning_rate_value,
            13: epoch_duration,
            14: _chart_frame(loss_rows),
            15: _chart_frame(metric_rows),
            16: _chart_frame(learning_rate_rows),
            17: "\n".join(logs),
            18: str(experiment_path / "logs" / "train.log"),
            19: "",
            20: monitor_error or "",
            21: evaluation_report,
            22: _safe_resource_text(resource_sampler),
            23: model_file,
            24: task_id,
        }
        yield changes_tuple(completed_changes)
    except Exception as error:
        details = traceback.format_exc()
        summary = (
            str(error)
            if str(error).startswith(("训练失败：", "GPU ", "当前 PyTorch ", "所选模型 "))
            else f"错误：{error}"
        )
        yield changes_tuple(
            {
                0: summary,
                17: "\n".join(logs),
                18: str(log_path) if log_path else None,
                19: summary,
                20: details,
                22: _safe_resource_text(resource_sampler),
                23: model_file,
                24: task_id,
            }
        )
    finally:
        close_sampler = getattr(resource_sampler, "close", None)
        if callable(close_sampler):
            try:
                close_sampler()
            except Exception:
                pass


def _preprocessing_summary(image_size: float) -> str:
    size = int(image_size)
    pipeline = (
        f"Resize ({size} × {size}) → RGB Tensor → Normalize "
        "(mean 0.485/0.456/0.406, std 0.229/0.224/0.225)"
    )
    return f"训练预处理：{pipeline}  \n验证预处理：{pipeline}"


def build_app() -> gr.Blocks:
    from .workspace import build_app as build_workspace

    return build_workspace(sys.modules[__name__])


def build_theme() -> gr.themes.Soft:
    return gr.themes.Soft(
        primary_hue="green",
        secondary_hue="gray",
        neutral_hue=WORKSPACE_NEUTRAL_HUE,
        radius_size="sm",
    ).set(
        body_background_fill="#f4f7f5",
        background_fill_primary="#f4f7f5",
        background_fill_secondary="#f4f7f5",
        panel_background_fill="#ffffff",
        block_background_fill="#ffffff",
        border_color_primary="#d9e1dc",
        block_label_background_fill="transparent",
        block_label_text_color="#4b5b52",
        block_label_border_width="0px",
        block_label_padding="0px",
        block_label_margin="0px 0 6px",
        block_label_shadow="none",
        block_title_background_fill="transparent",
        block_title_text_color="#202b26",
        block_title_padding="0px",
        block_shadow="none",
        checkbox_label_shadow="none",
        checkbox_label_shadow_hover="none",
        checkbox_label_shadow_active="none",
        button_secondary_shadow="none",
        button_secondary_shadow_hover="none",
        input_border_width="1px",
        input_border_color="#d9e1dc",
        table_even_background_fill="#ffffff",
        table_odd_background_fill="#f7f9f8",
    )


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=4).launch(
        inbrowser=True,
        css=WORKSPACE_CSS,
        theme=build_theme(),
    )
