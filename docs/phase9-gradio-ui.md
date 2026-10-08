# Phase 9：Gradio 训练界面

## 作用

Gradio 为 UATF 已有训练流程提供本地 Web 界面。界面收集参数后启动 `uatf.cli` 子进程，不会复制 Dataset、Model、Trainer 或 Evaluation 的实现。

## 启动

在仓库根目录安装依赖并启动：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m app.app
```

Gradio 会打印本地访问地址，通常为 `http://127.0.0.1:7860`。

## 数据集 ZIP 格式

上传一个 ZIP 文件，并通过任务选项指定其中一种固定结构。

单标签分类：ZIP 根目录下每个类别一个目录。

```text
dataset.zip
  cats/
    cat_001.jpg
  dogs/
    dog_001.jpg
```

多标签分类：ZIP 根目录下有 `annotations.csv` 和 `images/`。CSV 使用 `image_path` 列加一个或多个标签列，标签值只能是 `0` 或 `1`。

```text
dataset.zip
  annotations.csv
  images/
    image_001.jpg
```

图片数据会按 80%/20% 划分训练集和验证集。小数据集需要保证每个类别在划分后训练集和验证集中都有图片。

## 事件与回调

任务下拉框的 change 事件会更新数据格式提示。Start Training 的 click 回调检查参数、安全解压 ZIP、写出 YAML 配置，并执行 `python -m uatf.cli --config ...`。回调逐行读取子进程输出，在每个 epoch 完成时更新 Epoch、Loss、Metric、进度条和训练日志。训练结束后读取指标报告，并提供最佳 checkpoint 下载。

数据集校验、模型创建、训练、checkpoint 保存和评估仍由现有的 `uatf.data`、`uatf.models`、`uatf.training`、`uatf.evaluation` 模块经由 CLI 执行。把 UI 和 Training Engine 分开，可以独立调整操作界面，同时让命令行和 Web 界面运行相同的训练流程。

数据格式、标签、CUDA 不可用、模型名称和 GPU 显存不足等常见错误会显示简明信息。实验目录保留训练日志以便进一步排查。

## 输出

上传内容、界面生成的配置和训练实验分别写入 Git 忽略的 `outputs/ui_uploads/`、`outputs/ui_configs/` 和 `outputs/ui_runs/`。成功的实验目录包含常规配置、日志、指标和 best/last checkpoint。

## 范围

V1 只支持以上两种 ZIP 结构，不自动识别其他数据格式，也不下载预训练权重。CUDA 训练需要 CUDA 版 PyTorch；小型实验也可使用 CPU。
