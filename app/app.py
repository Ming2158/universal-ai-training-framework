"""Local Gradio interface that delegates training to the UATF CLI."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from stat import S_ISLNK
from typing import Any, Iterator

import gradio as gr
import yaml

from uatf.models import MODEL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "ui_uploads"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ui_runs"
CONFIG_ROOT = PROJECT_ROOT / "outputs" / "ui_configs"
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+)/(\d+).*?train_loss=([0-9.eE+-]+).*?val_loss=([0-9.eE+-]+)"
)
METRIC_PATTERN = re.compile(r"val_(accuracy|micro_f1)=([0-9.eE+-]+)")


def _dataset_instructions(task: str) -> str:
    if task == "multi_label":
        return (
            "上传 ZIP：根目录包含 `annotations.csv` 和 `images/`。CSV 第一列使用 `image_path`，"
            "其余列是标签名，值只能是 `0` 或 `1`。"
        )
    return "上传 ZIP：根目录直接包含类别子目录，每个类别目录中放该类图片，例如 `cats/a.jpg`。"


def _safe_extract_dataset(upload_path: str, task: str) -> Path:
    if not upload_path:
        raise ValueError("请先上传 ZIP 格式的数据集。")
    archive_path = Path(upload_path)
    if archive_path.suffix.lower() != ".zip":
        raise ValueError("数据集必须是 ZIP 文件。")

    destination = UPLOAD_ROOT / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
            for info in archive.infolist():
                normalized = info.filename.replace("\\", "/")
                path = PurePosixPath(normalized)
                parts = tuple(part for part in path.parts if part not in {"", "."})
                if path.is_absolute() or ".." in parts or (parts and ":" in parts[0]):
                    raise ValueError("ZIP 中包含不安全的文件路径。请重新打包数据集。")
                if not parts:
                    continue
                if S_ISLNK(info.external_attr >> 16):
                    raise ValueError("ZIP 中不能包含符号链接。请重新打包数据集。")
                members.append((info, parts))

            for info, parts in members:
                output_path = destination.joinpath(*parts)
                if info.is_dir():
                    output_path.mkdir(parents=True, exist_ok=True)
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, output_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except zipfile.BadZipFile as error:
        raise ValueError("无法读取 ZIP 文件，请确认文件没有损坏。") from error

    if task == "classification":
        class_directories = [path for path in destination.iterdir() if path.is_dir()]
        if not class_directories:
            raise ValueError(
                "单标签数据集格式错误：ZIP 根目录需要包含类别子目录，每个目录中放对应图片。"
            )
        image_count = sum(
            1
            for class_directory in class_directories
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if image_count == 0:
            raise ValueError("数据集没有找到图片。请检查类别子目录和图片文件。")
    else:
        if not (destination / "annotations.csv").is_file():
            raise ValueError("多标签数据集缺少根目录文件 `annotations.csv`。")
        if not (destination / "images").is_dir():
            raise ValueError("多标签数据集缺少根目录目录 `images/`。")

    return destination


def _friendly_error(output: str, return_code: int) -> str:
    error_lines = [line.removeprefix("Error: ").strip() for line in output.splitlines() if line.startswith("Error:")]
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


def start_training(
    upload_path: str | None,
    task: str,
    model_name: str,
    epochs: float,
    batch_size: float,
    learning_rate: float,
    device: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> Iterator[tuple[Any, ...]]:
    logs: list[str] = []
    current_epoch = "-"
    train_loss = "-"
    validation_loss = "-"
    metric_value = "-"
    metrics: dict[str, Any] = {}
    model_file: str | None = None

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
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning Rate 必须是大于 0 的有限数值。")
        if task not in {"classification", "multi_label"}:
            raise ValueError("请选择有效的任务类型。")

        progress(0, desc="检查并准备数据集")
        dataset_path = _safe_extract_dataset(upload_path or "", task)
        run_id = uuid.uuid4().hex
        config_path = CONFIG_ROOT / f"{run_id}.yaml"
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        config: dict[str, Any] = {
            "task": {"type": task},
            "dataset": {
                "format": "image_folder" if task == "classification" else "csv_multilabel",
                "path": str(dataset_path),
                "validation_fraction": 0.2,
                "image_size": 224,
            },
            "model": {"name": model_name, "pretrained": False},
            "training": {
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "device": device,
                "num_workers": 0,
            },
            "evaluation": {"primary_metric": "accuracy" if task == "classification" else "micro_f1"},
            "output": {"directory": str(OUTPUT_ROOT)},
        }
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        command = [sys.executable, "-m", "uatf.cli", "--config", str(config_path)]
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
        for line in process.stdout:
            clean_line = line.rstrip()
            logs.append(clean_line)
            epoch_match = EPOCH_PATTERN.search(clean_line)
            if epoch_match:
                current, total, train_value, validation_value = epoch_match.groups()
                current_epoch = f"{current} / {total}"
                train_loss = train_value
                validation_loss = validation_value
                metric_match = METRIC_PATTERN.search(clean_line)
                if metric_match:
                    metric_name, metric_value = metric_match.groups()
                    metric_value = f"{metric_name}: {float(metric_value):.4f}"
                progress(
                    int(current) / max(1, int(total)),
                    desc=f"Epoch {current}/{total} | train_loss={train_value} | val_loss={validation_value}",
                )
                yield (
                    "训练中",
                    current_epoch,
                    train_loss,
                    validation_loss,
                    metric_value,
                    "\n".join(logs[-300:]),
                    metrics,
                    model_file,
                )

        return_code = process.wait()
        log_text = "\n".join(logs[-300:])
        if return_code != 0:
            raise RuntimeError(_friendly_error(log_text, return_code))

        completed_line = next(
            (line for line in reversed(logs) if line.startswith("Training completed:")), None
        )
        if completed_line is None:
            raise RuntimeError("训练进程已结束，但没有找到实验结果目录。")
        experiment_path = Path(completed_line.split(":", 1)[1].strip())
        report_path = experiment_path / "metrics.json"
        if not report_path.is_file():
            raise RuntimeError("训练已结束，但没有生成 metrics.json。")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = {
            "task": report["task"],
            "model": report["model"],
            "dataset": report["dataset"],
            "training": report["training"],
            "evaluation": report["evaluation"],
            "environment": report["environment"],
        }
        model_file = str(experiment_path / "best_model.pth")
        if not Path(model_file).is_file():
            raise RuntimeError("训练已结束，但没有生成最佳模型 checkpoint。")
        progress(1, desc="训练完成")
        yield (
            "训练完成",
            current_epoch,
            train_loss,
            validation_loss,
            metric_value,
            log_text,
            metrics,
            model_file,
        )
    except Exception as error:
        yield (
            str(error) if str(error).startswith(("训练失败：", "GPU ", "当前 PyTorch ", "所选模型 ")) else f"错误：{error}",
            current_epoch,
            train_loss,
            validation_loss,
            metric_value,
            "\n".join(logs[-300:]),
            metrics,
            model_file,
        )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="UATF Training") as demo:
        gr.Markdown("# UATF 训练")
        with gr.Row():
            with gr.Column(scale=1):
                dataset_task = gr.Dropdown(
                    choices=[("单标签分类", "classification"), ("多标签分类", "multi_label")],
                    value="classification",
                    label="任务",
                )
                dataset_upload = gr.File(
                    label="数据集 ZIP",
                    file_count="single",
                    file_types=[".zip"],
                    type="filepath",
                )
                dataset_help = gr.Markdown(_dataset_instructions("classification"))
                model_select = gr.Dropdown(
                    choices=sorted(MODEL_REGISTRY),
                    value="resnet50" if "resnet50" in MODEL_REGISTRY else sorted(MODEL_REGISTRY)[0],
                    label="模型",
                )
                epochs_input = gr.Number(value=3, precision=0, minimum=1, label="Epochs")
                batch_size_input = gr.Number(value=8, precision=0, minimum=1, label="Batch Size")
                learning_rate_input = gr.Number(value=0.001, label="Learning Rate")
                device_select = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"], value="auto", label="Device"
                )
                start_button = gr.Button("开始训练", variant="primary")
            with gr.Column(scale=2):
                status_output = gr.Textbox(label="状态", value="等待数据集", interactive=False)
                with gr.Row():
                    epoch_output = gr.Textbox(label="Current Epoch", value="-", interactive=False)
                    train_loss_output = gr.Textbox(label="Train Loss", value="-", interactive=False)
                    validation_loss_output = gr.Textbox(label="Validation Loss", value="-", interactive=False)
                    metric_output = gr.Textbox(label="Metric", value="-", interactive=False)
                logs_output = gr.Textbox(label="Training Logs", lines=14, max_lines=24, interactive=False)
                metrics_output = gr.JSON(label="Metrics")
                model_download = gr.File(label="最佳模型下载")

        dataset_task.change(
            _dataset_instructions, inputs=dataset_task, outputs=dataset_help, show_progress="hidden"
        )
        start_button.click(
            start_training,
            inputs=[
                dataset_upload,
                dataset_task,
                model_select,
                epochs_input,
                batch_size_input,
                learning_rate_input,
                device_select,
            ],
            outputs=[
                status_output,
                epoch_output,
                train_loss_output,
                validation_loss_output,
                metric_output,
                logs_output,
                metrics_output,
                model_download,
            ],
            show_progress="full",
        )
    return demo


if __name__ == "__main__":
    build_app().queue().launch()
