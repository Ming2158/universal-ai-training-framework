"""Configuration-driven image classification training command."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from uatf.api import run_training
from uatf.config import load_config
from uatf.logging_utils import shutdown_logging
from uatf.tasks import get_task
from uatf.training import TrainingEvent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UATF 图像训练：使用交互向导，或通过 YAML 配置运行。"
    )
    parser.add_argument("--config", type=Path, help="YAML 配置文件路径")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="启动中英双语交互式训练向导"
    )
    parser.add_argument(
        "--language", choices=("zh", "en"), help="向导语言：zh（简体中文）或 en（English）"
    )
    parser.add_argument("--epochs", type=int, help="覆盖配置中的训练轮数")
    parser.add_argument("--batch-size", type=int, help="覆盖配置中的批次大小")
    parser.add_argument("--learning-rate", type=float, help="覆盖配置中的学习率")
    parser.add_argument("--seed", type=int, help="覆盖配置中的随机种子")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="覆盖运行设备")
    parser.add_argument("--output-dir", type=str, help="覆盖实验输出根目录")
    parser.add_argument(
        "--events-file",
        type=Path,
        help="将结构化训练生命周期事件写入 JSONL 文件",
    )
    parser.add_argument("--task-id", help=argparse.SUPPRESS)
    return parser


def _event_file_writer(path: Path) -> Callable[[TrainingEvent], None]:
    destination = path.expanduser()

    def write_event(event: TrainingEvent) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(asdict(event), ensure_ascii=False, allow_nan=False))
            output.write("\n")

    return write_event


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    interactive = args.interactive or (args.config is None and sys.stdin.isatty())
    if interactive and args.config is not None:
        parser.error("交互向导与 YAML 配置模式不能同时使用。")
    if interactive and any(
        getattr(args, name) is not None
        for name in ("epochs", "batch_size", "learning_rate", "seed", "device", "output_dir")
    ):
        parser.error("交互模式请在向导中设置参数；命令行覆盖参数需要配合 --config 使用。")
    if not interactive and args.language is not None:
        parser.error("--language 仅用于交互向导，请添加 --interactive。")
    if not interactive and args.config is None:
        parser.error("请提供 --config，或使用 --interactive 启动训练向导。")

    selected_language: str | None = None
    stop_event = threading.Event()
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda _signum, _frame: stop_event.set())
    try:
        if interactive:
            from uatf.cli_wizard import print_completed, run_wizard

            wizard_result = run_wizard(args.language)
            if wizard_result is None:
                return 0
            supplied_config, source_config, selected_language = wizard_result
            overrides: dict[str, Any] = {}
        else:
            assert args.config is not None
            supplied_config = load_config(args.config)
            overrides = {}
            cli_training: dict[str, Any] = {}
            for argument, key in (
                ("epochs", "epochs"),
                ("batch_size", "batch_size"),
                ("learning_rate", "learning_rate"),
                ("seed", "seed"),
                ("device", "device"),
            ):
                value = getattr(args, argument)
                if value is not None:
                    cli_training[key] = value
            if cli_training:
                overrides["training"] = cli_training
            if args.output_dir is not None:
                overrides["output"] = {"directory": args.output_dir}
            source_config = args.config.expanduser().resolve()

        task_id = args.task_id or uuid.uuid4().hex
        experiment_dir = run_training(
            supplied_config,
            overrides=overrides,
            source_config_path=source_config,
            event_callback=_event_file_writer(args.events_file) if args.events_file else None,
            task_id=task_id,
            stop_requested=stop_event.is_set,
        )
        output_directory = (
            overrides.get("output", {}).get("directory")
            or supplied_config.get("output", {}).get("directory", "outputs")
        )
        try:
            task = get_task(output_directory, task_id, reconcile_interruption=False)
        except (OSError, ValueError):
            task = {}
        if task.get("status") == "Stopped":
            print(f"训练已停止，最近完整检查点保留在：{experiment_dir}")
            return 130
        if selected_language is not None:
            print_completed(selected_language, experiment_dir)
        return 0
    except KeyboardInterrupt:
        print("\n已取消 / Cancelled.")
        return 130
    except (FileNotFoundError, FileExistsError, ValueError, OSError, RuntimeError) as error:
        label = "错误" if selected_language == "zh" else "Error"
        print(f"{label}: {error}", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        shutdown_logging()




if __name__ == "__main__":
    raise SystemExit(main())
