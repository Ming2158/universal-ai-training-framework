"""Beginner-friendly, bilingual terminal setup for UATF training runs."""

from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from stat import S_ISLNK
from typing import Any, Callable

import yaml

from uatf.config import validate_config
from uatf.models import MODEL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT = {
    "zh": {
        "welcome": "\n🎨 欢迎使用 UATF 训练向导！",
        "language": "选择界面语言：1) 简体中文  2) English",
        "language_changed": "界面语言已切换。",
        "language_hint": "💡 在任意问题输入 :lang 可切换中英文。按 Ctrl+C 可退出。",
        "task": "第一步：选择训练任务",
        "format": "选择数据集格式",
        "multilabel_format_fixed": "多标签任务使用 CSV 标注格式：",
        "layout_folder": "目录示例：数据集目录/类别名/图片.jpg；也可使用 train/ 和 validation/ 子目录。",
        "layout_csv_single": "示例：annotations.csv（image_path,label 两列）和 images/ 图片目录。",
        "layout_csv_multi": "示例：annotations.csv（image_path 列 + 多个 0/1 标签列）和 images/ 图片目录。",
        "dataset_path": "数据集路径（文件夹、CSV 文件或 ZIP 压缩包）",
        "model": "第二步：选择模型",
        "pretrained": "使用预训练权重？首次使用可能需要联网下载",
        "epochs": "训练轮数（Epochs，完整学习数据集的次数）",
        "batch_size": "批次大小（Batch Size，每次送入模型的图片数）",
        "learning_rate": "学习率（Learning Rate，每次更新模型的步长）",
        "device": "训练设备",
        "choose": "请输入编号 [默认 {default}] > ",
        "enter": "请输入内容 > ",
        "enter_default": "直接按回车使用默认值 [{default}] > ",
        "invalid_choice": "选项无效，请输入列表中的编号。",
        "invalid_integer": "请输入大于 0 的整数。",
        "invalid_rate": "请输入大于 0 的有效数字，例如 0.001。",
        "invalid_path": "找不到这个路径，请检查后重新输入。",
        "wrong_path_type": "这个数据格式需要提供文件夹路径。",
        "zip_extract": "正在安全解压数据集…",
        "bad_zip": "无法读取这个 ZIP 文件，请确认文件完整且未加密。",
        "unsafe_zip": "ZIP 中包含不安全的路径或符号链接，已停止解压。",
        "summary_title": "\n📋 训练设置确认",
        "summary_task": "任务",
        "summary_format": "数据格式",
        "summary_dataset": "数据位置",
        "summary_model": "模型",
        "summary_pretrained": "预训练权重",
        "summary_epochs": "训练轮数",
        "summary_batch": "批次大小",
        "summary_rate": "学习率",
        "summary_device": "设备",
        "summary_split": "训练/验证划分",
        "summary_output": "结果保存位置",
        "confirm": "现在开始训练吗？[Y/n] > ",
        "confirm_invalid": "请输入 Y（开始）或 N（取消）。",
        "cancelled": "已取消，本次没有启动训练。",
        "config_saved": "配置已保存：{path}",
        "starting": "\n🚀 正在启动训练…",
        "completed": "\n✅ 训练完成！结果保存在：{path}",
        "task_classification": "单标签分类（每张图片一个类别）",
        "task_multi_label": "多标签分类（每张图片可有多个标签）",
        "format_folder": "图片文件夹（类别分别放在子文件夹）",
        "format_csv_classification": "CSV 标注 + 图片文件夹",
        "format_csv_multi": "多标签 CSV 标注 + 图片文件夹",
        "model_resnet50": "ResNet-50（经典通用）",
        "model_convnext": "ConvNeXt-Tiny（新一代卷积网络）",
        "model_efficientnet": "EfficientNet-B0（轻量高效）",
        "model_vit": "ViT-B/16（视觉 Transformer）",
        "yes": "是",
        "no": "否",
        "auto": "自动选择",
        "cpu": "CPU（通用，速度较慢）",
        "cuda": "CUDA GPU（需要兼容环境）",
        "classification": "单标签分类",
        "multi_label": "多标签分类",
        "image_folder": "图片文件夹",
        "csv_classification": "CSV 单标签",
        "csv_multilabel": "CSV 多标签",
        "pretrained_yes": "开启",
        "pretrained_no": "关闭",
    },
    "en": {
        "welcome": "\n🎨 Welcome to the UATF training wizard!",
        "language": "Choose a language: 1) 简体中文  2) English",
        "language_changed": "Interface language changed.",
        "language_hint": "💡 Type :lang at any prompt to switch languages. Press Ctrl+C to quit.",
        "task": "Step 1: Choose a training task",
        "format": "Choose a dataset format",
        "multilabel_format_fixed": "Multi-label tasks use CSV annotations:",
        "layout_folder": "Example: dataset/class-name/image.jpg; optional train/ and validation/ folders are supported.",
        "layout_csv_single": "Example: annotations.csv (image_path,label columns) and an images/ folder.",
        "layout_csv_multi": "Example: annotations.csv (image_path plus 0/1 label columns) and an images/ folder.",
        "dataset_path": "Dataset path (folder, CSV file, or ZIP archive)",
        "model": "Step 2: Choose a model",
        "pretrained": "Use pretrained weights? The first download may need an internet connection",
        "epochs": "Training epochs (complete passes through the dataset)",
        "batch_size": "Batch size (images processed at a time)",
        "learning_rate": "Learning rate (how large each model update is)",
        "device": "Training device",
        "choose": "Enter a number [default {default}] > ",
        "enter": "Enter a value > ",
        "enter_default": "Press Enter to keep the default [{default}] > ",
        "invalid_choice": "That choice is not on the list. Enter one of the numbers shown.",
        "invalid_integer": "Enter a whole number greater than 0.",
        "invalid_rate": "Enter a valid number greater than 0, for example 0.001.",
        "invalid_path": "Path not found. Check it and try again.",
        "wrong_path_type": "This dataset format requires a folder path.",
        "zip_extract": "Safely extracting the dataset…",
        "bad_zip": "Could not read this ZIP file. Check that it is complete and not encrypted.",
        "unsafe_zip": "The ZIP contains an unsafe path or symbolic link. Extraction was stopped.",
        "summary_title": "\n📋 Review training settings",
        "summary_task": "Task",
        "summary_format": "Dataset format",
        "summary_dataset": "Dataset location",
        "summary_model": "Model",
        "summary_pretrained": "Pretrained weights",
        "summary_epochs": "Epochs",
        "summary_batch": "Batch size",
        "summary_rate": "Learning rate",
        "summary_device": "Device",
        "summary_split": "Train/validation split",
        "summary_output": "Results folder",
        "confirm": "Start training now? [Y/n] > ",
        "confirm_invalid": "Enter Y to start or N to cancel.",
        "cancelled": "Cancelled. Training was not started.",
        "config_saved": "Configuration saved to: {path}",
        "starting": "\n🚀 Starting training…",
        "completed": "\n✅ Training complete! Results are in: {path}",
        "task_classification": "Single-label classification (one class per image)",
        "task_multi_label": "Multi-label classification (multiple labels per image)",
        "format_folder": "Image folders (one subfolder per class)",
        "format_csv_classification": "CSV annotations + image folder",
        "format_csv_multi": "Multi-label CSV annotations + image folder",
        "model_resnet50": "ResNet-50 (classic general-purpose model)",
        "model_convnext": "ConvNeXt-Tiny (newer convolutional model)",
        "model_efficientnet": "EfficientNet-B0 (lightweight and efficient)",
        "model_vit": "ViT-B/16 (vision Transformer)",
        "yes": "Yes",
        "no": "No",
        "auto": "Auto",
        "cpu": "CPU (works broadly, but slower)",
        "cuda": "CUDA GPU (requires a compatible setup)",
        "classification": "Single-label classification",
        "multi_label": "Multi-label classification",
        "image_folder": "Image folders",
        "csv_classification": "Single-label CSV",
        "csv_multilabel": "Multi-label CSV",
        "pretrained_yes": "Enabled",
        "pretrained_no": "Disabled",
    },
}

MODEL_LABELS = {
    "resnet50": "model_resnet50",
    "convnext": "model_convnext",
    "efficientnet": "model_efficientnet",
    "vit": "model_vit",
}


class TerminalUI:
    def __init__(self, language: str | None = None) -> None:
        self.language = language or "zh"

    def text(self, key: str) -> str:
        return TEXT[self.language].get(key, key)

    def say(self, key: str) -> None:
        print(self.text(key))

    def choose_language(self) -> None:
        while True:
            try:
                answer = input(TEXT["zh"]["language"] + " > ").strip().lower()
            except EOFError as error:
                raise KeyboardInterrupt from error
            if answer in {"1", "zh", "中文"}:
                self.language = "zh"
                return
            if answer in {"2", "en", "english"}:
                self.language = "en"
                return
            print("请输入 1 或 2 / Enter 1 or 2.")

    def ask(self, key: str, *, default: str | None = None) -> str:
        while True:
            if key == "choose":
                prompt = self.text("choose").format(default=default)
                question = ""
            elif key == "confirm":
                prompt = self.text("confirm")
                question = ""
            elif default is not None:
                prompt = self.text("enter_default").format(default=default)
                question = f"{self.text(key)}\n"
            else:
                prompt = self.text("enter")
                question = f"{self.text(key)}\n"
            try:
                answer = input(f"{question}{prompt}").strip()
            except EOFError as error:
                raise KeyboardInterrupt from error
            if answer.lower() in {":lang", ":language"}:
                if key == "choose":
                    return answer
                self.choose_language()
                self.say("language_changed")
                continue
            return answer or (default if default is not None else "")

    def choose(
        self,
        key: str,
        options: list[tuple[str, Callable[["TerminalUI"], str]]],
        *,
        default: str,
    ) -> str:
        while True:
            self.say(key)
            for index, (_, label) in enumerate(options, start=1):
                print(f"  {index}) {label(self)}")
            answer = self.ask("choose", default=str(next(
                index for index, (value, _) in enumerate(options, start=1) if value == default
            )))
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1][0]
            if answer.lower() in {":lang", ":language"}:
                self.choose_language()
                self.say("language_changed")
                continue
            value_lookup = {value.lower(): value for value, _ in options}
            if answer.lower() in value_lookup:
                return value_lookup[answer.lower()]
            self.say("invalid_choice")


def _extract_dataset(archive_path: Path, data_format: str) -> Path:
    destination = PROJECT_ROOT / "outputs" / "cli_uploads" / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
            for info in archive.infolist():
                normalized = info.filename.replace("\\", "/")
                path = PurePosixPath(normalized)
                parts = tuple(part for part in path.parts if part not in {"", "."})
                if (
                    path.is_absolute()
                    or ".." in parts
                    or (parts and ":" in parts[0])
                    or S_ISLNK(info.external_attr >> 16)
                ):
                    raise ValueError("unsafe_zip")
                if parts:
                    members.append((info, parts))

            for info, parts in members:
                output_path = destination.joinpath(*parts)
                if not output_path.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("unsafe_zip")
                if info.is_dir():
                    output_path.mkdir(parents=True, exist_ok=True)
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, output_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except (zipfile.BadZipFile, RuntimeError) as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise ValueError("bad_zip") from error
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    children = list(destination.iterdir())
    if len(children) == 1 and children[0].is_dir():
        nested = children[0]
        if data_format == "image_folder" and any(item.is_dir() for item in nested.iterdir()):
            return nested
        if data_format != "image_folder" and (nested / "annotations.csv").is_file():
            return nested
    return destination


def _dataset_location(ui: TerminalUI, data_format: str) -> Path:
    while True:
        raw_path = ui.ask("dataset_path")
        if not raw_path.strip():
            ui.say("invalid_path")
            continue
        path = Path(raw_path.strip('"').strip("'")).expanduser()
        if not path.exists():
            ui.say("invalid_path")
            continue
        path = path.resolve()
        if path.suffix.lower() == ".zip":
            ui.say("zip_extract")
            try:
                return _extract_dataset(path, data_format)
            except ValueError as error:
                if str(error) in {"bad_zip", "unsafe_zip"}:
                    ui.say(str(error))
                    continue
                raise
        if data_format == "image_folder" and not path.is_dir():
            ui.say("wrong_path_type")
            continue
        if data_format != "image_folder" and path.is_dir() and not (
            path / "annotations.csv"
        ).is_file():
            ui.say("invalid_path")
            continue
        if data_format != "image_folder" and not (path.is_dir() or path.is_file()):
            ui.say("invalid_path")
            continue
        if data_format != "image_folder" and path.is_file() and path.suffix.lower() != ".csv":
            ui.say("invalid_path")
            continue
        return path


def _positive_integer(ui: TerminalUI, key: str, default: int) -> int:
    while True:
        answer = ui.ask(key, default=str(default))
        try:
            value = int(answer)
            if value > 0:
                return value
        except ValueError:
            pass
        ui.say("invalid_integer")


def _learning_rate(ui: TerminalUI, default: float) -> float:
    while True:
        answer = ui.ask("learning_rate", default=str(default))
        try:
            value = float(answer)
            if 0 < value < float("inf"):
                return value
        except ValueError:
            pass
        ui.say("invalid_rate")


def _model_label(ui: TerminalUI, name: str) -> str:
    key = MODEL_LABELS.get(name)
    return f"{name} — {ui.text(key)}" if key else name


def run_wizard(language: str | None = None) -> tuple[dict[str, Any], Path, str] | None:
    """Prompt for a complete training setup; return None when the user cancels."""
    ui = TerminalUI(language)
    ui.say("welcome")
    if language is None:
        ui.choose_language()
    ui.say("language_hint")

    task = ui.choose(
        "task",
        [
            ("classification", lambda current: current.text("task_classification")),
            ("multi_label", lambda current: current.text("task_multi_label")),
        ],
        default="classification",
    )
    if task == "multi_label":
        data_format = "csv_multilabel"
        ui.say("multilabel_format_fixed")
        print(f"  • {ui.text('format_csv_multi')}")
    else:
        data_format = ui.choose(
            "format",
            [
                ("image_folder", lambda current: current.text("format_folder")),
                (
                    "csv_classification",
                    lambda current: current.text("format_csv_classification"),
                ),
            ],
            default="image_folder",
        )

    layout_key = {
        "image_folder": "layout_folder",
        "csv_classification": "layout_csv_single",
        "csv_multilabel": "layout_csv_multi",
    }[data_format]
    ui.say(layout_key)

    dataset_path = _dataset_location(ui, data_format)
    models = sorted(MODEL_REGISTRY)
    model_name = ui.choose(
        "model",
        [
            (name, lambda current, model=name: _model_label(current, model))
            for name in models
        ],
        default="resnet50" if "resnet50" in MODEL_REGISTRY else models[0],
    )
    pretrained = ui.choose(
        "pretrained",
        [
            ("yes", lambda current: current.text("yes")),
            ("no", lambda current: current.text("no")),
        ],
        default="no",
    ) == "yes"
    epochs = _positive_integer(ui, "epochs", 3)
    batch_size = _positive_integer(ui, "batch_size", 8)
    learning_rate = _learning_rate(ui, 0.001)
    device = ui.choose(
        "device",
        [
            ("auto", lambda current: current.text("auto")),
            ("cuda", lambda current: current.text("cuda")),
            ("cpu", lambda current: current.text("cpu")),
        ],
        default="auto",
    )

    configuration = validate_config(
        {
            "task": {"type": task},
            "dataset": {
                "format": data_format,
                "path": str(dataset_path),
                "validation_fraction": 0.2,
                "image_size": 224,
            },
            "model": {"name": model_name, "pretrained": pretrained},
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "device": device,
                "num_workers": 0,
            },
            "evaluation": {
                "primary_metric": "accuracy" if task == "classification" else "micro_f1"
            },
            "output": {"directory": "outputs/cli_runs"},
        }
    )

    ui.say("summary_title")
    task_label = ui.text("task_classification" if task == "classification" else "task_multi_label")
    format_key = {
        "image_folder": "format_folder",
        "csv_classification": "format_csv_classification",
        "csv_multilabel": "format_csv_multi",
    }[data_format]
    summary = [
        ("summary_task", task_label),
        ("summary_format", ui.text(format_key)),
        ("summary_dataset", str(dataset_path)),
        ("summary_model", _model_label(ui, model_name)),
        ("summary_pretrained", ui.text("pretrained_yes" if pretrained else "pretrained_no")),
        ("summary_epochs", str(epochs)),
        ("summary_batch", str(batch_size)),
        ("summary_rate", str(learning_rate)),
        ("summary_device", ui.text(device)),
        ("summary_split", "80% / 20%"),
        ("summary_output", "outputs/cli_runs"),
    ]
    for label_key, value in summary:
        separator = "：" if ui.language == "zh" else ": "
        print(f"  {ui.text(label_key)}{separator}{value}")

    while True:
        answer = ui.ask("confirm", default="Y").lower()
        if answer in {"y", "yes", "是", "1"}:
            break
        if answer in {"n", "no", "否", "0"}:
            ui.say("cancelled")
            return None
        ui.say("confirm_invalid")

    config_path = PROJECT_ROOT / "outputs" / "cli_configs" / f"cli_{uuid.uuid4().hex}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(configuration, destination, sort_keys=False, allow_unicode=True)
    print(ui.text("config_saved").format(path=config_path))
    ui.say("starting")
    return configuration, config_path, ui.language


def print_completed(language: str, experiment_path: Path) -> None:
    print(TEXT[language]["completed"].format(path=experiment_path))
