# Universal AI Training Framework (UATF)

UATF 是一个面向初学者的本地 AI 图像训练框架。目标是把数据集检查、任务与模型配置、训练、验证、指标计算、最佳模型保存和结果查看，整理成一条容易理解的流程。

Gitee 仓库：[universal-ai-training-framework](https://gitee.com/Dreamcat_DYT/universal-ai-training-framework)

**当前状态：** 目前完成的是项目规划和目录骨架。配置文件只是示例；数据读取、模型训练、评估和 Web UI 尚未实现。

## V1 范围

V1 先支持两类图像任务：

- 单标签图像分类：训练集和验证集按类别分文件夹组织。
- 多标签图像分类：图片目录加 CSV 标注文件，一张图片可以有多个标签。

V1 还会提供数据集结构检查、ResNet18 和 MobileNetV3-Small 两个基线模型、统一的训练与验证流程、按验证指标保存最佳 checkpoint，以及本地 Streamlit 操作界面。模型默认不下载预训练权重，避免首次运行需要网络。命令行和配置文件是先打通核心流程的方式；UI 复用同一套 Python 代码，不另写训练逻辑。

多标签 CSV 的 V1 格式建议为 `image,labels` 两列，每行一张图片，多个标签用 `|` 分隔，例如 `img001.jpg,cat|outdoor`。数据集检查会验证图片路径、空标签和标签集合；后续实现时再把格式检查规则固化成代码。

数据与生成的训练结果保留在本机，不提交到 Git。用户可以在任意位置保存数据集，并在配置中填写路径。

## 目录说明

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
  training/              训练循环、优化器和 checkpoint 选择
  evaluation/            验证流程与任务指标
  ui/                    精简的本地 Streamlit 界面
  utils/                 随机种子、路径等少量通用工具
  config.py              读取并检查训练配置
  cli.py                 命令行入口
```

## 核心数据流

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

## 环境与依赖

计划使用 Python 3.10 或更新版本、PyTorch、torchvision、Pillow、PyYAML 和 Streamlit；pytest 用于开发测试。PyTorch 的安装方式取决于 CPU/GPU 环境，实际安装时应使用 PyTorch 官方安装选择器给出的命令。本次初始化没有安装任何依赖。

待核心代码实现后，可在虚拟环境中安装项目和测试依赖：

```powershell
python -m pip install -e ".[dev]"
```

当前还没有可运行的训练或 UI 入口，因此此命令留待实现阶段使用。

## 测试路线

先用不依赖 GPU 的小型临时图片检查数据集扫描、格式识别和标签解析；再检查模型输出维度；最后用少量临时样本做一次短 CPU 训练、验证和 checkpoint 保存。测试数据应自动生成或足够小，不把真实数据集和权重文件放进仓库。

## 首次提交到 Gitee

确认 `.gitignore` 已生效，并检查待提交文件后，在项目目录手动执行：

```powershell
git status
git add .
git commit -m "chore: initialize UATF project structure"
git push -u origin main
```

`git add .` 不会添加被忽略的数据集、训练输出或模型权重。以上步骤假设 Gitee 仓库还没有自己的初始提交；如果你在 Gitee 页面选择创建了 README 等文件，需要先同步远程提交再推送。
