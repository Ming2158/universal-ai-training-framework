# Universal AI Training Framework (UATF)

UATF 是一个面向初学者的本地 AI 图像训练框架。目标是把数据集检查、任务与模型配置、训练、验证、指标计算、最佳模型保存和结果查看，整理成一条容易理解的流程。

Gitee 仓库：[universal-ai-training-framework](https://gitee.com/Dreamcat_DYT/universal-ai-training-framework)

**当前状态：** Phase 9 提供 Gradio 本地训练界面，可上传数据集、配置并启动训练、查看进度和指标，以及下载最佳模型。训练由既有 CLI、Trainer 和 Evaluation 模块执行。

## 项目目标

UATF 希望提供一条清晰、可配置、可测试的图像模型训练流程。第一阶段聚焦单标签和多标签图像分类，并为后续任务类型保留清楚的模块边界。

## Features

当前仓库包含通用数据集适配器、模型工厂、Training Engine 和 MNIST 教学训练脚本。MNIST 流程不依赖预训练权重；它用于理解 Dataset、DataLoader、CNN、训练循环、验证与 checkpoint。

V1 当前支持：

- 单标签图像分类：训练集和验证集按类别分文件夹组织。
- 多标签图像分类：图片目录加 CSV 标注文件，一张图片可以有多个标签。

后续目标还包括更多基线模型。模型默认不下载预训练权重。命令行、配置文件和 Gradio UI 复用同一套 Python 训练代码。

多标签 CSV 使用 `image_path` 加多个二进制标签列，例如 `image_path,cat,outdoor`；每行用 `0`/`1` 表示图片是否包含对应标签。数据集适配器会验证列名、图片路径和标签值。

数据与生成的训练结果保留在本机，不提交到 Git。用户可以在任意位置保存数据集，并在配置中填写路径。

## Project Structure

```text
configs/                 训练配置示例
datasets/                本地数据集的可选存放位置，数据内容不纳入 Git
outputs/                 checkpoint 和训练结果的默认输出位置，内容不纳入 Git
tests/
  unit/                  单个模块的测试
  integration/           多个模块组合后的流程测试
uatf/
  data/                  检查数据、读取样本、转换图像并构造 DataLoader
  tasks/                 定义任务的标签格式、损失函数和指标规则
  models/                按配置创建模型并约定模型输出
  training/              通用训练循环、验证、checkpoint、resume、scheduler 和 AMP
  evaluation/            验证流程与任务指标
  utils/                 随机种子、路径等少量通用工具
  config.py              读取并检查训练配置
  cli.py                 命令行入口
app/
  app.py                 Gradio 本地训练界面
```

Training Engine 用 `Trainer` 接收 Dataset、Model、Loss、Optimizer、Scheduler 和 Metrics，不包含胸部 X 光等数据集知识。使用方法、checkpoint 字段和 resume 行为见 [Phase 6 训练引擎说明](docs/phase6-training-engine.md)。评估入口、指标解释和输出文件见 [Phase 7 评估系统说明](docs/phase7-evaluation-system.md)。配置优先级、CLI 和实验目录见 [Phase 8 配置系统说明](docs/phase8-configuration-system.md)。

## Architecture

```text
Dataset
  -> 检查图片、标签和目录/标注格式
  -> Dataset + DataLoader
  -> Model
  -> Training epochs
  -> Validation
  -> Evaluation metrics
  -> 保存最佳 Checkpoint
  -> 输出配置、指标和训练结果供 UI 查看
```

数据模块负责把文件变成批次；任务模块规定标签表示、损失和指标含义；模型模块输出预测值；训练循环更新模型参数；验证和评估只计算模型表现，不更新参数；checkpoint 和结果模块记录最佳模型及其配置。UI 负责收集用户设置、启动同一训练流程并显示进度和结果。

未来检测和分割的目标格式、损失与指标会和分类不同。通过任务边界为它们留出扩展位置即可，V1 不提前实现相关算法。

## Installation

Windows 开发环境使用 Python 3.12 和项目目录中的 `.venv` 虚拟环境。虚拟环境把项目的 Python 包与系统 Python 分开；本机不需要 Conda。RTX 5080 属于 Blackwell 架构，需要支持 CUDA 12.8 的 PyTorch。Phase 1 使用 PyTorch 2.7.1、torchvision 0.22.1 和官方 CUDA 12.8 轮子。NVIDIA 驱动已包含运行 GPU 程序所需的驱动部分；PyTorch 轮子自带 CUDA runtime，因此不需要另外安装 CUDA Toolkit。

在 PowerShell 中安装用户级 Python 3.12，然后创建项目虚拟环境：

```powershell
winget install --id Python.Python.3.12 --exact --scope user
py -3.12 -m venv .venv
```

关闭并重新打开 PowerShell 后，将 PyTorch 和 torchvision 安装到 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

## Usage

使用 YAML 配置启动通用训练：

```powershell
python train.py --config configs/example.yaml
```

命令行参数可以覆盖常用训练配置，例如 `--epochs 3 --batch-size 8 --device cpu`。每次运行都会在 `outputs/experiment_NNN/` 保存有效配置、best/last checkpoint、指标和训练日志。配置格式、数据集目录和示例见 [Phase 8 配置系统说明](docs/phase8-configuration-system.md)。

启动本地 Gradio 训练界面：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m app.app
```

界面只接受 ZIP 数据集。单标签分类 ZIP 的根目录应直接包含类别子目录；多标签分类 ZIP 的根目录应包含 `annotations.csv` 和 `images/`，CSV 必须有 `image_path` 与至少一个 0/1 标签列。更多说明见 [Phase 9 Gradio 界面](docs/phase9-gradio-ui.md)。

运行 MNIST 教学训练（首次运行会下载约 12 MB 的 MNIST 数据）：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_mnist.py
```

脚本默认运行 5 个 epoch，优先使用 CUDA GPU，并将验证准确率最高的模型保存到 `outputs/phase3-mnist/best_model.pth`。CPU 和训练参数也可通过命令行选项设置。逐模块代码说明见 [Phase 3 学习说明](docs/phase3-first-training-pipeline.md)。

用 GPU Smoke Test 检查环境：

```powershell
.\.venv\Scripts\python.exe .\scripts\gpu_smoke_test.py
```

项目依赖声明保留在 `pyproject.toml`，因此不再添加一份重复的 `requirements.txt`。项目配置声明 PyTorch、torchvision、Pillow、PyYAML、Matplotlib 和 Gradio，`dev` 选项会增加 pytest。因此，执行 `pip install -e ".[dev]"` 时会安装配置中声明的全部项目依赖。`torchaudio` 只用于音频工作，本项目当前不需要。

GPU 验证通过后，开发阶段可在该虚拟环境中安装项目和测试依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

此安装命令会安装完整项目依赖和开发测试依赖。通用训练代码也可直接通过 Python API 调用。

## 测试路线

Phase 6 提供不依赖网络的集成测试：它实际训练小型 TensorDataset，运行验证、StepLR、best/last checkpoint、best model reload 和 resume。MNIST 教学脚本仍需在首次运行时下载数据。数据集及模型权重不提交到 Git。

## Git Workflow

UATF 由单人开发。文档、配置或小范围修复可以直接在 `main` 上完成；开始数据集、模型或训练器等独立功能时，用 `git switch -c feature/dataset` 这类命令创建并切换到 `feature/*` 分支。不要为了练习 Git 给每个小改动单独建分支。

每项功能按实际开发顺序检查并记录：

```text
修改 -> 运行相关测试 -> git status -> git diff -> git add <确认过的文件> -> git diff --cached -> git commit
```

只暂存当前功能需要的文件。`git status` 会列出未跟踪文件；`git diff` 默认只显示已跟踪文件的修改，未跟踪文件要先检查内容再决定是否暂存。测试通过并审阅暂存区后，再用 `feat:`、`fix:`、`docs:`、`refactor:`、`test:` 或 `chore:` 开头的提交说明记录一个完整、可理解的改动。

工作区是磁盘上的项目文件；暂存区选出下一个 commit 要记录的内容；`.git` 保存 commit、分支和 tag 历史。每个 commit 应对应一项经过测试、容易说明和回看的改动。用 `git branch` 查看本地分支，用 `git log --oneline` 回顾提交，用 `git remote -v` 检查当前配置的 Gitee remote。

功能分支完成测试后，可以切回 `main` 并合并，例如 `git switch main`、`git merge feature/dataset`。开始工作前，如果 Gitee 上可能有新提交，先 `git pull`；本地提交完成后，想备份或发布时再 `git push`。在版本里程碑上创建 tag，例如 `git tag v0.1.0`。当前仓库已经初始化，因此无需再次运行 `git init`。

## Roadmap

- [x] 建立 Python 项目骨架、示例配置和 GPU 检查脚本。
- [x] 完成 Git 忽略规则、项目文档和基础开发流程。
- [x] 实现数据集格式检查与分类数据加载。
- [x] 完成 MNIST 单标签分类的首个教学训练闭环。
- [x] 实现通用 Trainer、训练/验证、checkpoint、resume、StepLR、可选 Early Stopping 和 AMP。
- [x] 实现单标签和多标签评估、指标、预测导出与可视化。
- [x] 实现 YAML 配置合并、校验、CLI 训练入口和本地实验记录。
- [ ] 完善通用分类任务与基线模型选择。
- [x] 提供本地 Gradio UI。
- [ ] 完善单元测试和集成测试。
