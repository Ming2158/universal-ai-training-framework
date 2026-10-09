<p align="center"><img src="UATF_logo.png" alt="UATF logo" width="360"></p>

# 通用 AI 训练框架（UATF）

[English](README.md) | [简体中文](README_zh.md)

首次使用请查看[UATF 使用文档](使用文档.md)，了解 GUI 和 CLI 两种操作方式。

UATF 1.0.0 是一个基于 PyTorch 的本地框架，用于训练和评估图像分类模型。单标签与多标签分类共用同一套数据集、模型、训练和评估流程。

`v1.0.0` 标签之后的改动尚未发布。交互式 CLI 向导记录在[更新日志的 Unreleased 部分](CHANGELOG.md)。

## 支持功能

- 支持文件夹和 CSV 数据集，包含图像校验和可复现的训练/验证集划分。
- 支持 ResNet50、ConvNeXt-Tiny、EfficientNet-B0 和 ViT-B/16。
- 支持 YAML 配置、命令行参数覆盖、CPU/CUDA 设备选择、可选 CUDA AMP、StepLR 和提前停止。
- 保存最佳和最近检查点；Python `Trainer` API 支持恢复训练；提供指标、预测 CSV 和图表。
- 提供本地 Gradio 界面，并复用相同的 CLI 训练流程。
- 每次运行记录最终配置、数据集摘要、模型信息、指标、检查点名称、环境信息、时间戳和日志文件。

当前版本尚未实现目标检测和图像分割。相关扩展模块请参阅[架构说明](docs/architecture.md)。

## 安装

需要 Python 3.10 或更高版本。以下 PowerShell 示例创建虚拟环境并安装适用于 CUDA 12.8 的 PyTorch；请根据机器环境选择合适的 PyTorch 版本：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

仅使用 CPU 时，请根据平台安装对应的 CPU 版 PyTorch。项目依赖和开发工具版本范围见 `pyproject.toml`。

CPU/GPU 环境验证步骤见[完整安装指南](docs/installation.md)。

## 数据集格式

单标签文件夹数据集按类别分别存放图像。可以使用一个数据集根目录，也可以显式提供训练集和验证集目录：

```text
dataset/
  train/
    cats/
    dogs/
  validation/
    cats/
    dogs/
```

如果没有划分目录，UATF 会根据类别文件夹进行带随机种子的随机划分。单标签 CSV 使用 `image_path,label` 两列；多标签 CSV 使用 `image_path` 列和每个标签对应的二值列。图像可以与 CSV 放在一起，或放在 `images/` 子目录中：

```text
dataset/
  annotations.csv
  images/
    image-001.jpg
```

CSV 中的图像路径相对于配置的图像目录。更多示例和校验规则见[数据集说明](docs/phase4-dataset-system.md)。

## 使用方法

在 Windows 上，双击仓库根目录中的 `启动UATF.bat` 即可打开本地界面。首次运行会创建 `venv` 环境并安装 CPU 版依赖；需要 Python 3.10、3.11 或 3.12，以及可用的网络连接。界面会自动在浏览器中打开。关闭启动器窗口即可停止 UATF。

示例配置默认从 `datasets/example/` 读取多标签数据集，其中包含 `annotations.csv` 和 `images/`：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/example.yaml
```

也可以通过命令行覆盖常用配置，无需修改 YAML：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/example.yaml --epochs 3 --batch-size 8 --device cpu
```

也可以使用纯终端交互向导创建训练任务：

```powershell
.\.venv\Scripts\python.exe train.py --interactive
```

安装项目后也可运行 `uatf --interactive`；在终端直接运行 `uatf` 或 `python train.py` 也会进入向导。若项目之前已安装，更新后执行一次 `python -m pip install -e .` 来生成新的 `uatf` 命令。向导支持简体中文和英文，在任意问题输入 `:lang` 即可切换。它会引导选择单标签/多标签任务、数据格式，输入本地数据集目录、CSV 文件或 ZIP 路径，选择模型、预训练权重、Epochs、Batch Size、Learning Rate 和 CPU/CUDA 设备，并在训练前展示设置摘要。分类文件夹 ZIP 的根目录应包含类别子目录；CSV ZIP 应包含 `annotations.csv`，图片放在 `images/` 或与 CSV 同级。向导把 ZIP 解压到 `outputs/cli_uploads/`，训练结果保存到 `outputs/cli_runs/`。

每次运行都会在 `outputs/` 下创建新目录，包含最终生效的 `config.yaml`、`metrics.json`、`predictions.csv`、图表、`best_model.pth`、`last_model.pth` 和 `logs/train.log`。默认不启用预训练权重；启用后可能会从 torchvision 下载权重。

从仓库根目录启动本地界面：

```powershell
.\.venv\Scripts\python.exe -m app.app
```

界面接受符合文档中两种数据集目录结构的 ZIP 文件。详情见 [Phase 9 界面说明](docs/phase9-gradio-ui.md)。

如需运行一个简单的入门示例，可以使用 MNIST 脚本。首次运行时，如果本地没有缓存数据集，脚本会下载数据：

```powershell
.\.venv\Scripts\python.exe scripts/train_mnist.py --epochs 1 --batch-size 128
```

## 测试与代码检查

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/gpu_smoke_test.py
```

如果 CUDA 不可用，GPU smoke test 会给出明确提示。单元测试不会下载预训练权重。MNIST 示例首次运行时可能会下载数据集。

## 项目结构

```text
app/app.py                 Gradio 用户界面
启动UATF.bat               Windows 双击启动器
configs/                   默认和示例 YAML 配置
datasets/                  可选的本地数据集；Git 会忽略数据文件
docs/                      架构、故障排查和各阶段说明
outputs/                   实验产物；Git 会忽略该目录
scripts/                   MNIST 示例和 GPU smoke test
tests/unit/                数据集、模型、评估和配置单元测试
tests/integration/         Trainer 和 CLI 工作流集成测试
uatf/data/                 图像数据集、变换和 DataLoader 辅助代码
uatf/models/               模型注册表和任务损失函数
uatf/training/             Trainer、验证、检查点、恢复训练和 AMP
uatf/evaluation/           指标、报告、预测和图表
uatf/config.py             YAML 加载、合并和校验
uatf/logging_utils.py      控制台和实验日志配置
uatf/cli.py                交互式与基于配置的训练入口
uatf/cli_wizard.py         双语终端训练向导
```

## 文档

- [架构说明](docs/architecture.md)
- [安装指南](docs/installation.md)
- [故障排查](docs/troubleshooting.md)
- [模型系统](docs/phase5-model-system.md)
- [训练引擎](docs/phase6-training-engine.md)
- [评估系统](docs/phase7-evaluation-system.md)
- [配置与 CLI](docs/phase8-configuration-system.md)
- [Gradio 界面](docs/phase9-gradio-ui.md)
- [历史工程审查](docs/engineering-review.md)
- [更新日志](CHANGELOG.md)
- [版本管理与发布规则](docs/versioning.md)
- [UATF 1.0.0 发布说明](docs/release-notes-v1.0.0.md)

## 范围与局限

UATF 1.0.0 面向单机图像分类。随机划分不支持分层抽样或分组划分。CLI 暂不提供恢复训练选项，但 Python API 中的 `Trainer.resume()` 可以使用。框架尚未提供精确的数据集内容校验和托管式实验存储。环境、数据和检查点相关问题见[故障排查](docs/troubleshooting.md)。
