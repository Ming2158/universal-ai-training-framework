"""Gradio workspace layout and event wiring."""

from __future__ import annotations

import queue
import threading
from typing import Any

import gradio as gr

from uatf import __version__
from uatf.models import MODEL_REGISTRY, get_model_adapters


def build_app(backend: Any) -> gr.Blocks:
    pages = backend.WORKSPACE_PAGES
    experiment_columns = ["name", "status", "task", "model", "best_metric", "created_at_utc"]
    model_columns = ["name", "architecture", "task", "available", "weights_kind", "experiment_id"]

    def compact_frame(frame: Any, columns: list[str]) -> Any:
        return frame.reindex(columns=columns)

    def page_header(title: str, description: str) -> None:
        gr.Markdown(
            f"# {title}\n\n{description}",
            elem_classes=["uatf-page-header"],
        )

    def navigate(page: str) -> tuple[Any, ...]:
        return backend._navigate_workspace(page)

    def refresh_dashboard(page: str) -> tuple[Any, Any]:
        if page != "工作台":
            return gr.skip(), gr.skip()
        summary, experiments = backend.load_dashboard_summary()
        return summary, compact_frame(experiments, experiment_columns)

    def refresh_experiments(*filters: Any) -> tuple[Any, Any, Any]:
        frame, selected, comparison = backend.refresh_experiment_browser(*filters)
        return compact_frame(frame, experiment_columns), selected, comparison

    def refresh_assets() -> tuple[Any, Any, Any]:
        frame, model_selection, inference_selection = backend.refresh_model_asset_browser()
        return compact_frame(frame, model_columns), model_selection, inference_selection

    def inspect_dataset(
        upload_path: str | None,
        local_path: str | None,
        task: str,
        data_format: str,
        validation_fraction: float,
        seed: float,
        image_size: float | None,
        resize_enabled: bool,
        verify_image_content: bool,
    ) -> Any:
        updates: queue.Queue[tuple[float | None, str]] = queue.Queue(maxsize=1)
        outcome: dict[str, Any] = {}
        finished = threading.Event()

        def report_progress(fraction: float | None, description: str) -> None:
            backend._offer_latest(updates, (fraction, description))

        def run_import() -> None:
            try:
                outcome["result"] = backend.inspect_dataset_ui(
                    upload_path,
                    local_path,
                    task,
                    data_format,
                    validation_fraction,
                    seed,
                    image_size,
                    progress_callback=report_progress,
                    verify_image_content=verify_image_content,
                    resize_enabled=resize_enabled,
                )
            except Exception as error:
                outcome["error"] = error
            finally:
                finished.set()

        skip_outputs = tuple(gr.skip() for _ in range(len(inspect_outputs) - 1))
        yield (
            *skip_outputs,
            backend._progress_bar_html(None, "准备读取数据集", title="数据集导入与检查"),
        )
        worker = threading.Thread(target=run_import, name="uatf-dataset-import", daemon=True)
        worker.start()
        while not finished.is_set() or not updates.empty():
            try:
                fraction, description = updates.get(timeout=0.25)
            except queue.Empty:
                continue
            yield (
                *skip_outputs,
                backend._progress_bar_html(
                    fraction,
                    description,
                    title="数据集导入与检查",
                    state="error" if "失败" in description else "running",
                ),
            )

        if "error" in outcome:
            error = outcome["error"]
            result = (
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
        else:
            result = outcome["result"]
        state = result[0]
        can_train = bool(state and state.get("report", {}).get("status") in {"ready", "warning"})
        report_status = state.get("report", {}).get("status") if state else None
        completion_text = (
            "导入与检查完成"
            if report_status in {"ready", "warning"}
            else "导入完成，发现阻断错误"
            if report_status == "blocked"
            else f"数据集导入/检查失败：{result[2]}"
        )
        if state and not state.get("report", {}).get("image_content_verified", True):
            completion_text = (
                "导入完成，发现阻断错误；未逐张验证图片内容"
                if report_status == "blocked"
                else "导入完成，未逐张验证图片内容"
            )
        yield (
            *result,
            gr.update(visible=True),
            gr.update(visible=not can_train),
            backend._progress_bar_html(
                1.0,
                completion_text,
                title="数据集导入与检查",
                state="error" if not can_train else "running",
            ),
        )

    def open_experiment_details(experiment_id: str | None) -> tuple[Any, ...]:
        result = backend.load_experiment_details(experiment_id)
        has_selection = bool(experiment_id)
        return (
            *result,
            gr.update(visible=has_selection),
            gr.update(visible=not has_selection),
        )

    def open_model_details(asset_id: str | None) -> tuple[Any, ...]:
        details = backend.load_model_asset_record(asset_id)
        has_details = bool(details)
        return (
            details,
            gr.update(visible=has_details),
            gr.update(visible=not has_details),
        )

    def export_model(asset_id: str | None) -> tuple[Any, ...]:
        result = backend.export_model_asset_ui(asset_id)
        has_files = bool(result[0])
        return (*result, gr.update(visible=has_files))

    def launch_copied_configuration(configuration_text: str | None) -> tuple[Any, ...]:
        result = backend.launch_copied_configuration_for_ui(configuration_text)
        return (*result, stop_enabled(result[2] or None))

    def run_evaluation(*inputs: Any) -> tuple[Any, ...]:
        result = backend.run_asset_evaluation_ui(*inputs)
        has_results = result[0] is not None
        return (
            *result,
            gr.update(visible=has_results),
            gr.update(visible=not has_results),
        )

    def run_image_prediction(*inputs: Any) -> tuple[Any, ...]:
        result = backend.run_asset_image_prediction_ui(*inputs)
        has_results = not result[1].empty
        return (
            *result,
            gr.update(visible=has_results),
            gr.update(visible=not has_results),
        )

    def run_directory_prediction(*inputs: Any) -> tuple[Any, ...]:
        result = backend.run_asset_directory_prediction_ui(*inputs)
        has_results = bool(result[2])
        return (
            *result,
            gr.update(visible=has_results),
            gr.update(visible=has_results),
            gr.update(visible=not has_results),
        )

    def read_task(task_id: str | None) -> dict[str, Any] | None:
        if not task_id:
            return None
        try:
            return backend.get_task(backend.OUTPUT_ROOT, task_id)
        except FileNotFoundError:
            return None

    def stop_enabled(task_id: str | None) -> dict[str, Any]:
        if not task_id:
            record = backend.get_latest_task(backend.OUTPUT_ROOT)
        else:
            record = read_task(task_id)
        can_stop = bool(record and record.get("status") in backend.ACTIVE_STATUSES)
        return gr.update(interactive=can_stop)

    def refresh_monitor(task_id: str | None = None) -> tuple[Any, ...]:
        result = backend.refresh_task_status(task_id)
        selected_task = task_id or result[7]
        record = read_task(selected_task)
        if not record:
            resource_state = gr.update(value="当前没有可监控的训练进程。")
        elif record.get("status") in backend.ACTIVE_STATUSES:
            if selected_task in backend.GUI_MONITORED_TASK_IDS:
                resource_state = gr.skip()
            else:
                resource_state = gr.update(
                    value="任务仍在运行，但当前界面没有连接它的实时资源采样器。"
                )
        else:
            resource_state = gr.update(value="任务已结束，运行时资源采样已停止。")
        return (
            *result,
            resource_state,
            stop_enabled(selected_task or None),
            gr.update(
                active=bool(record and record.get("status") in backend.ACTIVE_STATUSES)
            ),
        )

    def request_stop(task_id: str | None) -> tuple[str, dict[str, Any]]:
        return backend.request_training_stop(task_id), gr.update(interactive=False)

    with gr.Blocks(title="UATF Training Workspace") as demo:
        dataset_state = gr.State(None)
        sample_index = gr.State(0)
        loaded_model_asset = gr.State(None)
        evaluation_state = gr.State(None)

        with gr.Row(elem_id="uatf-shell"):
            with gr.Column(scale=0, elem_id="uatf-sidebar"):
                gr.Markdown(
                    "## UATF\n\nAI Training Workspace",
                    elem_id="uatf-brand",
                )
                page_navigation = gr.Radio(
                    choices=list(pages),
                    value="工作台",
                    label="工作区",
                    elem_id="uatf-navigation",
                    show_label=False,
                )
                gr.Markdown(
                    f"V{__version__}  ·  本地训练",
                    elem_id="uatf-version",
                )

            with gr.Column(scale=1, elem_id="uatf-main"):
                with gr.Group(visible=True, elem_id="page-dashboard") as dashboard_page:
                    page_header("工作台", "任务状态与近期实验")
                    with gr.Row(elem_classes=["uatf-responsive-row"]):
                        with gr.Column(scale=2, elem_classes=["uatf-section"]):
                            gr.Markdown("### 最近训练任务", elem_classes=["uatf-section-title"])
                            dashboard_task_summary = gr.Markdown("正在读取任务状态…")
                            with gr.Row():
                                dashboard_open_training = gr.Button("创建训练", variant="primary")
                                dashboard_open_monitor = gr.Button("训练监控")
                                dashboard_refresh = gr.Button("刷新")
                        with gr.Column(scale=3, elem_classes=["uatf-section"]):
                            gr.Markdown("### 最近实验", elem_classes=["uatf-section-title"])
                            dashboard_experiments = gr.Dataframe(
                                headers=experiment_columns,
                                datatype=["str", "str", "str", "str", "number", "str"],
                                label="",
                                interactive=False,
                                max_height=260,
                                wrap=True,
                                column_widths=["150px", "88px", "105px", "105px", "100px", "175px"],
                            )
                    with gr.Row(elem_classes=["uatf-responsive-row"]):
                        dashboard_open_dataset = gr.Button("数据集检查")
                        dashboard_open_experiments = gr.Button("查看实验")
                        dashboard_open_models = gr.Button("查看模型资产")

                with gr.Group(visible=False, elem_id="page-datasets") as datasets_page:
                    page_header("数据集", "导入、检查并确认训练数据")
                    with gr.Row(elem_classes=["uatf-responsive-row"]):
                        with gr.Column(scale=5, elem_classes=["uatf-section"]):
                            gr.Markdown("### 数据来源", elem_classes=["uatf-section-title"])
                            dataset_task = gr.Dropdown(
                                choices=[
                                    ("单标签分类", "classification"),
                                    ("多标签分类", "multi_label"),
                                ],
                                value="classification",
                                label="任务类型",
                            )
                            dataset_format = gr.Dropdown(
                                choices=["image_folder", "csv_classification"],
                                value="image_folder",
                                label="数据格式",
                            )
                            with gr.Row(elem_classes=["uatf-responsive-row"]):
                                dataset_upload = gr.File(
                                    label="上传 ZIP 数据集",
                                    file_count="single",
                                    file_types=[".zip"],
                                    type="filepath",
                                )
                                local_dataset_path = gr.Textbox(
                                    label="服务端目录或 CSV 路径",
                                    placeholder=r"D:\datasets\cats-and-dogs",
                                )
                            dataset_help = gr.Markdown(
                                backend._dataset_instructions("classification", "image_folder"),
                                elem_classes=["uatf-note"],
                            )
                            gr.Markdown(
                                "划分和预处理",
                                elem_classes=["uatf-section-title"],
                            )
                            with gr.Row(elem_classes=["uatf-responsive-row"]):
                                validation_fraction = gr.Slider(
                                    minimum=0.05,
                                    maximum=0.5,
                                    value=0.2,
                                    step=0.05,
                                    label="验证集比例",
                                )
                                seed_input = gr.Number(
                                    value=42,
                                    precision=0,
                                    minimum=0,
                                    label="随机种子",
                                )
                            image_size_input = gr.Number(
                                value=224,
                                precision=0,
                                minimum=1,
                                label="缩放尺寸",
                            )
                            resize_enabled = gr.Checkbox(
                                value=True,
                                label="训练时缩放至固定尺寸",
                                info="关闭后保留原始尺寸并按批次补边；可能增加显存占用，ViT 需要开启。",
                            )
                            preprocessing_output = gr.Markdown(
                                backend._preprocessing_summary(224, True),
                                elem_classes=["uatf-note"],
                            )
                            verify_image_content = gr.Checkbox(
                                value=False,
                                label="导入时逐张验证图片内容（耗时）",
                            )
                            inspect_button = gr.Button("导入数据集", variant="primary")
                            dataset_import_progress = gr.HTML(
                                value=backend._progress_bar_html(
                                    0.0,
                                    "等待导入数据集",
                                    title="数据集导入与检查",
                                )
                            )
                            gr.Markdown(
                                "ZIP 解压缓存位于 `outputs/ui_uploads/`；数据集记录和分布图位于 "
                                "`outputs/ui_datasets/`。清理缓存不会移动或删除本地数据集源文件。",
                                elem_classes=["uatf-note"],
                            )
                        with gr.Column(scale=7, elem_classes=["uatf-section"]):
                            gr.Markdown(
                                "**尚未导入数据集**\n\n填写数据目录或上传 ZIP，然后导入数据集。标签、样本统计和预览会显示在这里。",
                                elem_id="uatf-empty-state-dataset",
                            )
                            with gr.Group(
                                visible=False,
                                elem_classes=["uatf-results"],
                            ) as dataset_inspection_results:
                                inspection_summary = gr.Markdown("等待导入数据集")
                                with gr.Tabs():
                                    with gr.Tab("检查结果"):
                                        inspection_details = gr.JSON(label="数据集检查报告")
                                        with gr.Row(elem_classes=["uatf-responsive-row"]):
                                            distribution_plot = gr.Image(
                                                type="filepath",
                                                label="类别分布与训练/验证拆分",
                                            )
                                            class_distribution = gr.Dataframe(
                                                headers=[
                                                    "Label",
                                                    "Samples",
                                                    "Ratio",
                                                    "Train",
                                                    "Validation",
                                                ],
                                                datatype=[
                                                    "str",
                                                    "number",
                                                    "str",
                                                    "number",
                                                    "number",
                                                ],
                                                label="类别统计",
                                                interactive=False,
                                                max_height=300,
                                                wrap=True,
                                            )
                                        accept_warnings = gr.Checkbox(
                                            label="我已阅读并接受检查报告中的警告",
                                            value=False,
                                        )
                                    with gr.Tab("样本预览"):
                                        with gr.Row(elem_classes=["uatf-responsive-row"]):
                                            original_preview = gr.Image(
                                                type="pil", label="原始样本"
                                            )
                                            processed_preview = gr.Image(
                                                type="pil", label="训练/验证预处理结果"
                                            )
                                        with gr.Row(elem_classes=["uatf-responsive-row"]):
                                            previous_sample = gr.Button("上一张")
                                            sample_information = gr.Textbox(
                                                label="样本标签和路径",
                                                lines=3,
                                                interactive=False,
                                            )
                                            next_sample = gr.Button("下一张")

                with gr.Group(visible=False, elem_id="page-training") as training_page:
                    page_header("训练", "配置并启动一个训练任务")
                    with gr.Row(elem_id="uatf-empty-state-training") as training_dataset_empty_state:
                        gr.Markdown(
                            "**训练配置尚未开放**\n\n请先导入一个数据集。完成数据集结构检查后，模型与训练参数会在此处显示。"
                        )
                        training_open_dataset = gr.Button("前往数据集", variant="secondary")
                    with gr.Group(visible=False) as training_config_group:
                        training_dataset_summary = gr.Markdown()
                        with gr.Row(elem_classes=["uatf-responsive-row"]):
                            with gr.Column(scale=5, elem_classes=["uatf-section"]):
                                gr.Markdown(
                                    "### 模型与训练参数", elem_classes=["uatf-section-title"]
                                )
                                model_adapters = get_model_adapters()
                                model_choices = [
                                    (adapter.display_name, adapter.identifier)
                                    for adapter in model_adapters
                                ]
                                model_select = gr.Dropdown(
                                    choices=model_choices,
                                    value=(
                                        "resnet50"
                                        if "resnet50" in MODEL_REGISTRY
                                        else model_choices[0][1]
                                    ),
                                    label="模型",
                                )
                                with gr.Row():
                                    epochs_input = gr.Number(
                                        value=3,
                                        precision=0,
                                        minimum=1,
                                        label="Epochs",
                                    )
                                    batch_size_input = gr.Number(
                                        value=8,
                                        precision=0,
                                        minimum=1,
                                        label="Batch Size",
                                    )
                                    learning_rate_input = gr.Number(
                                        value=0.001,
                                        label="Learning Rate",
                                    )
                                with gr.Row():
                                    device_choices = [("auto", "auto"), ("cpu", "cpu")]
                                    cuda_available = backend.torch.cuda.is_available()
                                    if cuda_available:
                                        device_choices.insert(1, ("cuda", "cuda"))
                                    device_select = gr.Dropdown(
                                        choices=device_choices,
                                        value="auto" if cuda_available else "cpu",
                                        label="Device",
                                        info=(
                                            None
                                            if cuda_available
                                            else (
                                                "当前 Python 环境未检测到 CUDA；"
                                                "需要 GPU 时请使用 CUDA 版 PyTorch 启动 UATF。"
                                            )
                                        ),
                                    )
                                    augmentation_enabled = gr.Checkbox(
                                        value=False,
                                        label="启用训练数据增强",
                                    )
                                    worker_count_input = gr.Number(
                                        value=min(4, max(0, (backend.os.cpu_count() or 1) - 1)),
                                        precision=0,
                                        minimum=0,
                                        maximum=16,
                                        label="DataLoader workers",
                                        info="并行读取与缩放；大数据集可从 4 开始。内存较小时设为 0。",
                                    )
                            with gr.Column(scale=4, elem_classes=["uatf-section"]):
                                gr.Markdown("### 启动前确认", elem_classes=["uatf-section-title"])
                                gr.Markdown(
                                    "任务类型、数据格式、拆分、种子与图像尺寸沿用数据集导入结果。\n\n"
                                    "配置会通过共享 UATF schema 写入 YAML，再由 CLI 启动训练。",
                                    elem_classes=["uatf-note"],
                                )
                                start_button = gr.Button("开始训练", variant="primary")
                                gr.Markdown(
                                    f"输出目录：`{backend.OUTPUT_ROOT}`",
                                    elem_classes=["uatf-note"],
                                )

                with gr.Group(visible=False, elem_id="page-monitor") as monitor_page:
                    page_header("训练监控", "观察运行状态、指标和任务产物")
                    with gr.Row():
                        status_output = gr.Textbox(
                            label="任务状态与阶段",
                            value="等待训练任务",
                            interactive=False,
                        )
                        task_id_output = gr.Textbox(
                            label="任务 ID",
                            value="",
                            interactive=False,
                        )
                        refresh_task_button = gr.Button("刷新任务", variant="secondary")
                        stop_button = gr.Button("停止训练", variant="stop", interactive=False)
                    training_monitor_timer = gr.Timer(value=1.0, active=False)
                    overall_progress_output = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=0.1,
                        precision=1,
                        label="总体进度 (%)",
                        interactive=False,
                    )
                    with gr.Tabs():
                        with gr.Tab("运行指标"):
                            gr.Markdown(
                                "损失和验证指标曲线会在训练产生首个完整 Epoch 指标后出现。",
                                elem_classes=["uatf-note"],
                            )
                            with gr.Row():
                                epoch_output = gr.Textbox(
                                    label="Epoch", value="-", interactive=False
                                )
                                batch_output = gr.Textbox(
                                    label="批次", value="尚未计算", interactive=False
                                )
                                elapsed_output = gr.Textbox(
                                    label="已用时间", value="0:00", interactive=False
                                )
                                eta_output = gr.Textbox(
                                    label="预计剩余", value="计算中", interactive=False
                                )
                                metric_updated_output = gr.Textbox(
                                    label="最近指标更新时间",
                                    value="尚未计算",
                                    interactive=False,
                                )
                            with gr.Row():
                                train_loss_output = gr.Textbox(
                                    label="Train Loss", value="尚未计算", interactive=False
                                )
                                validation_loss_output = gr.Textbox(
                                    label="Validation Loss",
                                    value="尚未计算",
                                    interactive=False,
                                )
                                metric_output = gr.Textbox(
                                    label="主要验证指标",
                                    value="尚未计算",
                                    interactive=False,
                                )
                                best_metric_output = gr.Textbox(
                                    label="历史最佳主要指标",
                                    value="尚未计算",
                                    interactive=False,
                                )
                            with gr.Row():
                                best_epoch_output = gr.Textbox(
                                    label="最佳 Epoch", value="尚未计算", interactive=False
                                )
                                learning_rate_output = gr.Textbox(
                                    label="当前学习率",
                                    value="尚未计算",
                                    interactive=False,
                                )
                                epoch_duration_output = gr.Textbox(
                                    label="最近 Epoch 耗时",
                                    value="尚未计算",
                                    interactive=False,
                                )
                            with gr.Row():
                                loss_curve = gr.LinePlot(
                                    value=backend._chart_frame([]),
                                    x="epoch",
                                    y="value",
                                    color="series",
                                    title="Train / Validation Loss",
                                    height=280,
                                    buttons=["fullscreen", "export"],
                                )
                                primary_curve = gr.LinePlot(
                                    value=backend._chart_frame([]),
                                    x="epoch",
                                    y="value",
                                    color="series",
                                    title="主要验证指标",
                                    height=280,
                                    buttons=["fullscreen", "export"],
                                )
                            learning_rate_curve = gr.LinePlot(
                                value=backend._chart_frame([]),
                                x="epoch",
                                y="value",
                                color="series",
                                title="Learning Rate",
                                height=240,
                                buttons=["fullscreen", "export"],
                            )
                            resource_output = gr.Textbox(
                                label="硬件与训练进程资源",
                                value="等待读取任务状态",
                                lines=3,
                                interactive=False,
                            )
                        with gr.Tab("日志与诊断"):
                            logs_output = gr.Textbox(
                                label="实时训练日志（最近 200 行）",
                                lines=16,
                                max_lines=28,
                                interactive=False,
                            )
                            with gr.Row():
                                log_file_output = gr.File(label="完整日志文件")
                                error_summary_output = gr.Textbox(
                                    label="错误摘要", lines=3, interactive=False
                                )
                            technical_error_output = gr.Textbox(
                                label="技术堆栈与诊断",
                                lines=10,
                                max_lines=20,
                                interactive=False,
                            )
                        with gr.Tab("配置与产物"):
                            with gr.Row():
                                monitor_experiment_name = gr.Textbox(
                                    label="关联实验", interactive=False
                                )
                                monitor_configuration = gr.JSON(label="生效配置")
                            monitor_checkpoints = gr.Dataframe(
                                headers=[
                                    "kind",
                                    "epoch",
                                    "saved_at_utc",
                                    "size_bytes",
                                    "supports_resume",
                                    "inference_only",
                                    "path",
                                ],
                                label="检查点",
                                interactive=False,
                                max_height=240,
                            )
                            with gr.Row():
                                monitor_best_checkpoint = gr.File(label="最佳检查点")
                                monitor_latest_checkpoint = gr.File(label="最近检查点")
                                model_download = gr.File(label="最佳模型下载")
                            metrics_output = gr.JSON(
                                label="训练完成后的独立验证集评估（与逐 Epoch 曲线分开）"
                            )

                with gr.Group(visible=False, elem_id="page-experiments") as experiments_page:
                    page_header("实验管理", "筛选历史记录，查看详情或比较实验")
                    with gr.Row(elem_classes=["uatf-responsive-row"]):
                        with gr.Column(scale=6, elem_classes=["uatf-section"]):
                            gr.Markdown("### 实验列表", elem_classes=["uatf-section-title"])
                            with gr.Row():
                                experiment_name_filter = gr.Textbox(label="名称包含")
                                experiment_model_filter = gr.Textbox(label="模型包含")
                                experiment_status_filter = gr.Dropdown(
                                    choices=[
                                        "",
                                        "Preparing",
                                        "Running",
                                        "Validating",
                                        "Stopped",
                                        "Completed",
                                        "Failed",
                                        "Unknown",
                                    ],
                                    value="",
                                    label="状态",
                                )
                            with gr.Row():
                                experiment_after_filter = gr.Textbox(
                                    label="开始时间不早于（ISO-8601）"
                                )
                                experiment_before_filter = gr.Textbox(
                                    label="开始时间不晚于（ISO-8601）"
                                )
                                refresh_experiments_button = gr.Button("刷新列表")
                            experiment_table = gr.Dataframe(
                                headers=experiment_columns,
                                datatype=["str", "str", "str", "str", "number", "str"],
                                label="实验记录",
                                interactive=False,
                                max_height=360,
                                wrap=True,
                                column_widths=["150px", "88px", "105px", "105px", "100px", "175px"],
                            )
                            selected_experiment = gr.Dropdown(choices=[], label="选择实验查看详情")
                            inspect_experiment_button = gr.Button("打开实验详情")
                        with gr.Column(scale=7, elem_classes=["uatf-section"]):
                            with gr.Tabs():
                                with gr.Tab("实验详情"):
                                    experiment_detail_empty_state = gr.Markdown(
                                        "**尚未打开实验详情**\n\n先在左侧选择一条记录，再点击“打开实验详情”。完整配置、训练历史、日志和检查点会显示在这里。",
                                        elem_id="uatf-empty-state-experiment",
                                    )
                                    with gr.Group(
                                        visible=False,
                                        elem_classes=["uatf-results"],
                                    ) as experiment_detail_group:
                                        with gr.Row(elem_classes=["uatf-responsive-row"]):
                                            experiment_record_output = gr.JSON(
                                                label="实验记录与生效信息"
                                            )
                                            experiment_metrics_output = gr.JSON(label="评估结果")
                                        experiment_history_plot = gr.LinePlot(
                                            value=backend._chart_frame([]),
                                            x="epoch",
                                            y="value",
                                            color="series",
                                            title="训练历史",
                                            height=260,
                                            buttons=["fullscreen", "export"],
                                        )
                                        with gr.Row(elem_classes=["uatf-responsive-row"]):
                                            experiment_log_output = gr.Textbox(
                                                label="训练日志",
                                                lines=10,
                                                max_lines=20,
                                                interactive=False,
                                            )
                                            experiment_checkpoints_output = gr.Dataframe(
                                                headers=[
                                                    "kind",
                                                    "epoch",
                                                    "saved_at_utc",
                                                    "size_bytes",
                                                    "supports_resume",
                                                    "inference_only",
                                                    "path",
                                                ],
                                                label="检查点",
                                                interactive=False,
                                                max_height=260,
                                                wrap=True,
                                            )
                                        with gr.Row(elem_classes=["uatf-responsive-row"]):
                                            best_checkpoint_download = gr.File(label="最佳检查点")
                                            latest_checkpoint_download = gr.File(label="最近检查点")
                                        copied_configuration = gr.Code(
                                            label="复制的 YAML 配置",
                                            language="yaml",
                                            lines=14,
                                        )
                                        with gr.Row(elem_classes=["uatf-responsive-row"]):
                                            copy_configuration_button = gr.Button("载入此实验配置")
                                            launch_copy_button = gr.Button(
                                                "以此配置启动新实验",
                                                variant="primary",
                                            )
                                            copied_task_status = gr.Textbox(
                                                label="新任务状态",
                                                interactive=False,
                                            )
                                with gr.Tab("实验比较"):
                                    compare_experiments_select = gr.Dropdown(
                                        choices=[],
                                        multiselect=True,
                                        label="选择要比较的实验",
                                    )
                                    compare_experiments_button = gr.Button("比较所选实验")
                                    comparison_output = gr.JSON(
                                        value={
                                            "message": "选择至少两个实验后，查看配置和指标差异。"
                                        },
                                        label="比较结果与可比性说明",
                                    )

                with gr.Group(visible=False, elem_id="page-models") as models_page:
                    page_header("模型管理", "浏览训练生成的模型资产")
                    with gr.Row(elem_classes=["uatf-responsive-row"]):
                        with gr.Column(scale=6, elem_classes=["uatf-section"]):
                            gr.Markdown("### 模型资产", elem_classes=["uatf-section-title"])
                            with gr.Row():
                                model_asset_select = gr.Dropdown(choices=[], label="选择模型资产")
                                refresh_assets_button = gr.Button("刷新资产")
                            model_asset_table = gr.Dataframe(
                                headers=model_columns,
                                label="模型资产记录",
                                interactive=False,
                                max_height=380,
                                wrap=True,
                                column_widths=["150px", "125px", "110px", "90px", "110px", "175px"],
                            )
                        with gr.Column(scale=5, elem_classes=["uatf-section"]):
                            gr.Markdown("### 资产详情", elem_classes=["uatf-section-title"])
                            view_asset_button = gr.Button("查看资产详情")
                            model_detail_empty_state = gr.Markdown(
                                "选择一个模型资产并查看详情。训练记录、权重位置和评估信息会在这里展示。",
                                elem_id="uatf-empty-state-model",
                            )
                            with gr.Group(
                                visible=False,
                                elem_classes=["uatf-results"],
                            ) as model_detail_group:
                                model_asset_details = gr.JSON(label="配置、训练摘要与评估结果")
                                with gr.Row(elem_classes=["uatf-responsive-row"]):
                                    load_asset_button = gr.Button("加载模型")
                                    export_asset_button = gr.Button(
                                        "导出推理权重", variant="secondary"
                                    )
                            model_asset_status = gr.Textbox(label="模型资产状态", interactive=False)
                            with gr.Group(visible=False) as model_export_group:
                                with gr.Row(elem_classes=["uatf-responsive-row"]):
                                    exported_asset_file = gr.File(label="导出权重")
                                    exported_asset_sidecar = gr.File(label="JSON 配置侧车")

                with gr.Group(visible=False, elem_id="page-evaluation") as evaluation_page:
                    page_header("评估与推理", "使用已登记的模型资产执行评估和预测")
                    inference_asset_select = gr.Dropdown(choices=[], label="模型资产")
                    with gr.Tabs():
                        with gr.Tab("独立评估"):
                            with gr.Row(elem_classes=["uatf-responsive-row"]):
                                with gr.Column(scale=5, elem_classes=["uatf-section"]):
                                    evaluation_dataset_path = gr.Textbox(
                                        label="带标签的数据集目录或 CSV 路径（服务端）"
                                    )
                                    evaluation_format = gr.Dropdown(
                                        choices=[
                                            "image_folder",
                                            "csv_classification",
                                            "csv_multilabel",
                                        ],
                                        value="image_folder",
                                        label="数据格式",
                                    )
                                    evaluation_image_root = gr.Textbox(
                                        label="CSV 图片目录（可选）",
                                        placeholder="默认使用 CSV 同级 images/ 或目录",
                                    )
                                    with gr.Row():
                                        evaluation_batch_size = gr.Number(
                                            value=32,
                                            precision=0,
                                            minimum=1,
                                            label="Batch Size",
                                        )
                                        evaluation_threshold = gr.Slider(
                                            minimum=0,
                                            maximum=1,
                                            value=0.5,
                                            step=0.01,
                                            label="多标签阈值",
                                        )
                                    evaluate_asset_button = gr.Button(
                                        "运行独立评估", variant="primary"
                                    )
                                    evaluation_status = gr.Textbox(
                                        label="评估状态", interactive=False
                                    )
                                with gr.Column(scale=7, elem_classes=["uatf-section"]):
                                    evaluation_empty_state = gr.Markdown(
                                        "**尚未运行评估**\n\n选择模型资产，填写带标签的数据集路径，然后运行评估。指标、曲线和错误样本会在下方显示。",
                                        elem_id="uatf-empty-state-evaluation",
                                    )
                            with gr.Group(
                                visible=False,
                                elem_classes=["uatf-results"],
                            ) as evaluation_results_group:
                                with gr.Row(elem_classes=["uatf-responsive-row"]):
                                    evaluation_summary = gr.JSON(
                                        label="总体指标、数据集、检查点与配置"
                                    )
                                    per_class_metrics = gr.Dataframe(
                                        headers=[
                                            "class_or_label",
                                            "precision",
                                            "recall",
                                            "f1",
                                            "support",
                                            "auc",
                                            "average_precision",
                                            "positive_support",
                                            "negative_support",
                                        ],
                                        label="逐类别或逐标签指标",
                                        interactive=False,
                                        max_height=250,
                                        wrap=True,
                                    )
                                with gr.Row(elem_classes=["uatf-responsive-row"]):
                                    evaluation_confusion_plot = gr.Image(
                                        type="filepath", label="混淆矩阵"
                                    )
                                    evaluation_roc_plot = gr.Image(
                                        type="filepath", label="ROC 曲线"
                                    )
                                    evaluation_pr_plot = gr.Image(type="filepath", label="PR 曲线")
                                evaluation_predictions = gr.Dataframe(
                                    label="评估预测结果",
                                    interactive=False,
                                    max_height=300,
                                    wrap=True,
                                )
                                with gr.Row(elem_classes=["uatf-responsive-row"]):
                                    error_class_filter = gr.Dropdown(
                                        choices=[], label="错误类别/标签"
                                    )
                                    error_sample_select = gr.Dropdown(choices=[], label="错误样本")
                                    error_sample_information = gr.Textbox(
                                        label="真实标签、预测和分数",
                                        lines=6,
                                        interactive=False,
                                    )
                                with gr.Row(elem_classes=["uatf-responsive-row"]):
                                    error_predictions = gr.Dataframe(
                                        label="错误预测",
                                        interactive=False,
                                        max_height=250,
                                        wrap=True,
                                    )
                                    error_sample_image = gr.Image(type="filepath", label="原始图片")
                        with gr.Tab("单张图片"):
                            with gr.Row(elem_classes=["uatf-responsive-row"]):
                                single_image_input = gr.Image(
                                    type="filepath",
                                    label="选择或上传图片",
                                    sources=["upload"],
                                )
                                with gr.Column(elem_classes=["uatf-section"]):
                                    single_image_button = gr.Button("执行推理", variant="primary")
                                    single_image_status = gr.Textbox(
                                        label="预测结果", interactive=False
                                    )
                                    single_image_empty_state = gr.Markdown(
                                        "上传图片并执行推理后，模型分数和详细输出会显示在这里。",
                                        elem_id="uatf-empty-state-single-inference",
                                    )
                                    with gr.Group(visible=False) as single_image_results:
                                        single_image_scores = gr.Dataframe(
                                            headers=[
                                                "class_or_label",
                                                "logit_score",
                                                "probability",
                                            ],
                                            label="模型分数与后处理概率",
                                            interactive=False,
                                            max_height=300,
                                            wrap=True,
                                        )
                                        single_image_report = gr.JSON(label="推理明细")
                        with gr.Tab("批量图片"):
                            with gr.Row(elem_classes=["uatf-responsive-row"]):
                                with gr.Column(scale=4, elem_classes=["uatf-section"]):
                                    batch_image_directory = gr.Textbox(
                                        label="图片目录（UATF 服务端）",
                                        placeholder=r"D:\datasets\images",
                                    )
                                    with gr.Row():
                                        batch_threshold = gr.Slider(
                                            minimum=0,
                                            maximum=1,
                                            value=0.5,
                                            step=0.01,
                                            label="多标签阈值",
                                        )
                                        inference_batch_size = gr.Number(
                                            value=32,
                                            precision=0,
                                            minimum=1,
                                            label="Batch Size",
                                        )
                                    batch_prediction_button = gr.Button(
                                        "批量推理", variant="primary"
                                    )
                                    batch_prediction_status = gr.Textbox(
                                        label="批量推理状态", interactive=False
                                    )
                                    batch_prediction_empty_state = gr.Markdown(
                                        "填写图片目录并运行批量推理后，结果表和 CSV 下载会显示在这里。",
                                        elem_id="uatf-empty-state-batch-inference",
                                    )
                                    with gr.Group(visible=False) as batch_prediction_results:
                                        batch_predictions_file = gr.File(label="预测 CSV")
                                with gr.Column(scale=8, elem_classes=["uatf-section"]):
                                    with gr.Group(visible=False) as batch_prediction_table_group:
                                        batch_predictions = gr.Dataframe(
                                            label="批量预测结果",
                                            interactive=False,
                                            max_height=380,
                                            wrap=True,
                                        )

        page_groups = [
            dashboard_page,
            datasets_page,
            training_page,
            monitor_page,
            experiments_page,
            models_page,
            evaluation_page,
        ]
        navigation_outputs = [page_navigation, *page_groups]
        page_navigation.change(
            lambda page: backend._workspace_visibility(page),
            inputs=page_navigation,
            outputs=page_groups,
            show_progress="hidden",
        )
        page_navigation.change(
            refresh_dashboard,
            inputs=page_navigation,
            outputs=[dashboard_task_summary, dashboard_experiments],
            show_progress="hidden",
        )
        for button, page in (
            (dashboard_open_dataset, "数据集"),
            (dashboard_open_training, "训练"),
            (dashboard_open_monitor, "训练监控"),
            (dashboard_open_experiments, "实验管理"),
            (dashboard_open_models, "模型管理"),
        ):
            button.click(
                lambda target=page: navigate(target),
                outputs=navigation_outputs,
                show_progress="hidden",
            )
        training_open_dataset.click(
            lambda: navigate("数据集"),
            outputs=navigation_outputs,
            show_progress="hidden",
        )
        dashboard_refresh.click(
            lambda: refresh_dashboard("工作台"),
            outputs=[dashboard_task_summary, dashboard_experiments],
            show_progress="hidden",
        )

        experiment_filter_inputs = [
            experiment_name_filter,
            experiment_model_filter,
            experiment_status_filter,
            experiment_after_filter,
            experiment_before_filter,
        ]
        experiment_browser_outputs = [
            experiment_table,
            selected_experiment,
            compare_experiments_select,
        ]
        refresh_experiments_button.click(
            refresh_experiments,
            inputs=experiment_filter_inputs,
            outputs=experiment_browser_outputs,
            show_progress="hidden",
        )
        inspect_experiment_button.click(
            open_experiment_details,
            inputs=selected_experiment,
            outputs=[
                experiment_record_output,
                experiment_metrics_output,
                experiment_history_plot,
                experiment_log_output,
                experiment_checkpoints_output,
                best_checkpoint_download,
                latest_checkpoint_download,
                experiment_detail_group,
                experiment_detail_empty_state,
            ],
            show_progress="hidden",
        )
        compare_experiments_button.click(
            backend.compare_selected_experiments,
            inputs=compare_experiments_select,
            outputs=comparison_output,
            show_progress="hidden",
        )
        copy_configuration_button.click(
            backend.copy_experiment_configuration,
            inputs=selected_experiment,
            outputs=[copied_configuration, copied_task_status],
            show_progress="hidden",
        )
        launch_copy_button.click(
            launch_copied_configuration,
            inputs=copied_configuration,
            outputs=[
                copied_task_status,
                status_output,
                task_id_output,
                *navigation_outputs,
                stop_button,
            ],
            show_progress="hidden",
        )

        refresh_assets_button.click(
            refresh_assets,
            outputs=[model_asset_table, model_asset_select, inference_asset_select],
            show_progress="hidden",
        )
        view_asset_button.click(
            open_model_details,
            inputs=model_asset_select,
            outputs=[model_asset_details, model_detail_group, model_detail_empty_state],
            show_progress="hidden",
        )
        load_asset_button.click(
            backend.load_model_asset_ui,
            inputs=model_asset_select,
            outputs=[loaded_model_asset, model_asset_status],
            show_progress="hidden",
        )
        export_asset_button.click(
            export_model,
            inputs=model_asset_select,
            outputs=[
                exported_asset_file,
                exported_asset_sidecar,
                model_asset_status,
                model_export_group,
            ],
            show_progress="hidden",
        )

        inference_asset_select.change(
            backend._evaluation_format_choices,
            inputs=inference_asset_select,
            outputs=evaluation_format,
            show_progress="hidden",
        )
        evaluate_asset_button.click(
            run_evaluation,
            inputs=[
                inference_asset_select,
                evaluation_dataset_path,
                evaluation_format,
                evaluation_image_root,
                evaluation_batch_size,
                evaluation_threshold,
            ],
            outputs=[
                evaluation_state,
                evaluation_status,
                evaluation_summary,
                per_class_metrics,
                evaluation_confusion_plot,
                evaluation_roc_plot,
                evaluation_pr_plot,
                evaluation_predictions,
                error_class_filter,
                error_sample_select,
                error_predictions,
                evaluation_results_group,
                evaluation_empty_state,
            ],
            show_progress="minimal",
        )
        error_class_filter.change(
            backend.filter_evaluation_errors,
            inputs=[evaluation_state, error_class_filter],
            outputs=[error_predictions, error_sample_select],
            show_progress="hidden",
        )
        error_sample_select.change(
            backend.preview_evaluation_error,
            inputs=[error_sample_select, evaluation_state],
            outputs=[error_sample_image, error_sample_information],
            show_progress="hidden",
        )
        single_image_button.click(
            run_image_prediction,
            inputs=[inference_asset_select, single_image_input, evaluation_threshold],
            outputs=[
                single_image_status,
                single_image_scores,
                single_image_report,
                single_image_results,
                single_image_empty_state,
            ],
            show_progress="minimal",
        )
        batch_prediction_button.click(
            run_directory_prediction,
            inputs=[
                inference_asset_select,
                batch_image_directory,
                batch_threshold,
                inference_batch_size,
            ],
            outputs=[
                batch_prediction_status,
                batch_predictions,
                batch_predictions_file,
                batch_prediction_results,
                batch_prediction_table_group,
                batch_prediction_empty_state,
            ],
            show_progress="minimal",
        )

        dataset_task.change(
            backend._format_choices,
            inputs=dataset_task,
            outputs=dataset_format,
            show_progress="hidden",
        )
        dataset_task.change(
            backend._dataset_instructions,
            inputs=[dataset_task, dataset_format],
            outputs=dataset_help,
            show_progress="hidden",
        )
        dataset_format.change(
            backend._dataset_instructions,
            inputs=[dataset_task, dataset_format],
            outputs=dataset_help,
            show_progress="hidden",
        )
        image_size_input.change(
            backend._preprocessing_summary,
            inputs=[image_size_input, resize_enabled],
            outputs=preprocessing_output,
            show_progress="hidden",
        )
        resize_enabled.change(
            backend._preprocessing_summary,
            inputs=[image_size_input, resize_enabled],
            outputs=preprocessing_output,
            show_progress="hidden",
        )
        inspect_outputs = [
            dataset_state,
            sample_index,
            inspection_summary,
            inspection_details,
            class_distribution,
            distribution_plot,
            original_preview,
            processed_preview,
            sample_information,
            accept_warnings,
            training_config_group,
            dataset_inspection_results,
            training_dataset_empty_state,
            dataset_import_progress,
        ]
        inspect_button.click(
            inspect_dataset,
            inputs=[
                dataset_upload,
                local_dataset_path,
                dataset_task,
                dataset_format,
                validation_fraction,
                seed_input,
                image_size_input,
                resize_enabled,
                verify_image_content,
            ],
            outputs=inspect_outputs,
            show_progress="hidden",
        ).then(
            lambda state: (
                backend._inspection_summary(state["report"])
                if state and state.get("report")
                else "请检查并修正数据集问题。"
            ),
            inputs=dataset_state,
            outputs=training_dataset_summary,
            show_progress="hidden",
        )
        for button, direction in ((previous_sample, -1), (next_sample, 1)):
            button.click(
                lambda state, index, step=direction: backend.turn_sample(state, index, step),
                inputs=[dataset_state, sample_index],
                outputs=[original_preview, processed_preview, sample_information, sample_index],
                show_progress="hidden",
            )

        training_inputs = [
            dataset_upload,
            local_dataset_path,
            dataset_state,
            dataset_task,
            dataset_format,
            validation_fraction,
            seed_input,
            image_size_input,
            resize_enabled,
            accept_warnings,
            model_select,
            epochs_input,
            batch_size_input,
            learning_rate_input,
            device_select,
            augmentation_enabled,
            worker_count_input,
        ]
        training_state_outputs = [
            status_output,
            overall_progress_output,
            epoch_output,
            batch_output,
            elapsed_output,
            eta_output,
            metric_updated_output,
            train_loss_output,
            validation_loss_output,
            metric_output,
            best_metric_output,
            best_epoch_output,
            learning_rate_output,
            epoch_duration_output,
            loss_curve,
            primary_curve,
            learning_rate_curve,
            logs_output,
            log_file_output,
            error_summary_output,
            technical_error_output,
            metrics_output,
            resource_output,
            model_download,
            task_id_output,
        ]
        training_outputs = [
            *training_state_outputs,
            *navigation_outputs,
            stop_button,
            training_monitor_timer,
        ]
        start_button.click(
            backend.start_training_for_ui,
            inputs=training_inputs,
            outputs=training_outputs,
            show_progress="hidden",
            concurrency_limit=1,
        )

        monitor_outputs = [
            status_output,
            overall_progress_output,
            epoch_output,
            logs_output,
            log_file_output,
            error_summary_output,
            technical_error_output,
            task_id_output,
            resource_output,
            stop_button,
            training_monitor_timer,
        ]
        training_monitor_timer.tick(
            backend.poll_training_for_ui,
            inputs=task_id_output,
            outputs=[*training_state_outputs, stop_button, training_monitor_timer],
            show_progress="hidden",
            concurrency_limit=1,
        )
        demo.load(
            lambda: refresh_dashboard("工作台"),
            outputs=[dashboard_task_summary, dashboard_experiments],
            show_progress="hidden",
        )
        demo.load(
            refresh_experiments,
            inputs=experiment_filter_inputs,
            outputs=experiment_browser_outputs,
            show_progress="hidden",
        )
        demo.load(
            refresh_assets,
            outputs=[model_asset_table, model_asset_select, inference_asset_select],
            show_progress="hidden",
        )
        demo.load(
            refresh_monitor,
            outputs=monitor_outputs,
            show_progress="hidden",
        ).then(
            backend.load_task_artifacts,
            inputs=task_id_output,
            outputs=[
                monitor_experiment_name,
                monitor_configuration,
                monitor_checkpoints,
                monitor_best_checkpoint,
                monitor_latest_checkpoint,
            ],
            show_progress="hidden",
        )
        refresh_task_button.click(
            refresh_monitor,
            inputs=task_id_output,
            outputs=monitor_outputs,
            show_progress="hidden",
        ).then(
            backend.load_task_artifacts,
            inputs=task_id_output,
            outputs=[
                monitor_experiment_name,
                monitor_configuration,
                monitor_checkpoints,
                monitor_best_checkpoint,
                monitor_latest_checkpoint,
            ],
            show_progress="hidden",
        )
        stop_button.click(
            request_stop,
            inputs=task_id_output,
            outputs=[status_output, stop_button],
            show_progress="hidden",
        )

    return demo
