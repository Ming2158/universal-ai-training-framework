# UATF 训练吞吐性能审查与优化参考方案

> 本文面向后续 AI 编码代理和性能工程师，作为 UATF 项目对标成熟 PyTorch 训练系统的源码审查记录和分阶段优化输入。它记录的是审查时工作区中的源码行为；实际运行配置、PyTorch/CUDA 版本、数据盘类型和硬件功耗限制可能改变结论。本文不代表性能修改已经实施或验证。

## 1. 目标与已知背景

本次性能分析的已知背景：

- 机器：NVIDIA RTX 5080，16 GB 显存。
- 数据：约 30,000 张胸部 X 光 JPG，原图平均约 3000 px，用户称已提前缩放至 384 px。
- 模型：ConvNeXt-Tiny。
- 批大小：32。
- 训练：截图时处于 Epoch 6/10、Batch 817/901，已用 1:00:18、预计剩余 50:16。按界面 ETA 外推，本次 10 epoch 运行约需 1:50:34；这是当时的动态预计值，不是最终实测总耗时。
- 期望：每个 epoch（含训练和验证）约 2–3 分钟。

“业界平均速度”不是一个脱离环境的固定数值。数据格式、图像尺寸、训练/验证比例、显卡是桌面还是移动版、功耗墙、CPU、存储、PyTorch/CUDA 版本和精度都会影响速度。因此，本文把 2–3 分钟作为此项目的目标验收值，不把它宣称为所有 RTX 5080 环境都必然达到的常数。

按截图的已用时间加预计剩余时间，当前投影总时长为 1 小时 50 分 34 秒。若把 10 个 epoch 近似视为等时，约为 11 分钟/epoch；按每个 epoch 遍历 30,000 张图粗算，吞吐约 45 张/秒。目标 2–3 分钟/epoch 对应约 167–250 张/秒，因此相对截图时的 ETA，目标约需 3.7–5.5 倍提速，而不是先前误算的 22–33 倍。这里仅用于修正文档中的数量级判断：训练进度中的 epoch 可能不等时，ETA 会随负载变化，截图也不能证明最终总耗时或区分每轮训练、验证和其他阶段各自耗时。

吞吐换算还有一项口径限制：30,000 张应理解为每个完整 epoch 中训练集与验证集合计处理的样本量。代码在 fit 周期后还会再次运行最终验证评估；截图上的 ETA 是否包含该阶段、checkpoint 写入和报告生成，无法仅从截图判断。准确基线仍应从运行目录的 epoch 记录和总运行记录中拆分计算。默认验证比例为 20% 时，每轮通常约有 24,000 张训练图和 6,000 张验证图。

## 2. 结论摘要

不客气地说，当前代码已经具备一个完整的本地训练应用所需的功能闭环，但还没有达到经过性能工程打磨的训练平台水平。它能够加载图像、启动训练、记录实验和显示硬件资源，却缺少完整的阶段计时、吞吐基准、数据缓存和针对 GPU 的快速训练路径。

源码可以确认以下问题：

1. Dataset 每次取样都重新打开 JPG；缩放、转 Tensor 和归一化也在取样时执行，没有图像内容缓存。
2. AMP 在默认值和当前 Web 配置生成路径中均关闭。
3. CUDA 训练设置为确定性 cuDNN 算法，并关闭 cuDNN benchmark，即优先确定性而非固定尺寸卷积吞吐。
4. CUDA 训练循环逐 batch 从 GPU 取标量，产生同步等待。
5. DataLoader 已具备 pinned memory、异步拷贝、persistent workers 和基础预取，但默认值因入口不同而变化，预取深度固定为 2。
6. Web 界面默认图像尺寸是 224、Batch Size 是 8；用户提供的 384 和 batch 32 只有写进该次运行的有效配置才会生效。
7. 训练每轮都验证，最终还会再评估一次；报告的完整墙钟耗时包含训练以外的工作。
8. 当前实现是单设备训练；没有发现 DDP、DataParallel、DALI、LMDB、WebDataset、TFRecord、MONAI CacheDataset、`channels_last` 或 `torch.compile` 路径。
9. 资源监控展示 CPU、GPU、显存、功耗和温度等快照，但没有拆解 DataLoader 等待、拷贝、前向、反向、优化器和验证分别耗时。

因此，不能负责任地承诺“把 worker 调大、开 AMP 或加入缓存，就一定达到 2–3 分钟”。截图修正后，目标差距约为 3.7–5.5 倍；这仍足以要求用 profile 找出真正瓶颈，但不支持此前“吞吐低到 11 小时”这一错误判断。首先仍应核对实际运行设备、输入尺寸、GPU 利用率、功耗/温度和数据等待，排除 `auto` 实际退回 CPU、环境启动了 CPU 版 PyTorch、显卡被功耗或热限制、有效图像尺寸与预期不一致等情况。

## 3. 源码链路与问题证据

### 3.1 数据集保存路径，不保存解码后的图像

`uatf/data/datasets.py` 的 `_ImageDataset` 将 `samples` 保存为图片路径和标签列表。`__getitem__` 每次被调用都会执行 `Image.open(path)`、`source.convert("RGB")` 和当前 transform。训练迭代器每个 epoch 都会再次调用 `__getitem__`，所以每轮都会重新读取和解码相应的 JPG。

关键位置：

- `uatf/data/datasets.py:78-124`：基础图像 Dataset、`with_transform()`、`__getitem__()`。
- `uatf/data/datasets.py:111-119`：逐次打开图片、转 RGB、调用 transform。
- `uatf/data/datasets.py:63-75`：只有显式启用内容校验时才做 `verify()` 和完整 `load()`。
- `app/workspace.py:365-368`：Web 导入时“逐张验证图片内容”默认关闭。
- `uatf/experiment.py:96-102`、`uatf/experiment.py:185-200`：训练构造 Dataset 时传入 `verify_images=False`。

这表示默认 Web 训练的导入检查通常不会在训练前把每张图完整解码一次，但训练阶段仍会在每次取样时重读和解码。若用户开启导入内容验证，则导入阶段会多一次全量检查；这是一笔一次性成本，不是每个 epoch 的缓存。

用户报告“文件已缩至 384”不能消除重复 JPG 解码。它只能减少每次解码的数据量。实际读取的是当前配置指向的文件；如果这些文件已经是 384×384，就不会在训练时重新读取此前的 3000 px 源文件，除非配置路径仍指向原图。

### 3.2 Resize、augmentation、ToTensor 和 Normalize 每次取样执行

`uatf/data/transforms.py:23-87` 构造 torchvision Compose：

1. 启用且只对训练集生效时，执行随机水平翻转、随机旋转或 ColorJitter。
2. `resize_enabled=True` 时执行 `transforms.Resize(tuple(size))`。
3. 执行 `ToTensor()` 和 `Normalize()`。

`uatf/experiment.py:55-77` 在创建数据加载器前分别构造训练和评估 transform；训练和验证数据集各自持有 transform。transform 对象本身会复用，但每个样本的 transform 运算会重做。`persistent_workers=True` 仅保持 worker 进程，不会保留已经解码的图像，也不会把变换结果缓存到 Dataset。

图像已是 384、配置也为 384 时，仍会调用 Resize transform。具体是否进行实际像素重采样取决于 Pillow/torchvision 对同尺寸输入的处理；不能把这一点当成主要瓶颈的确定证据。重复解码和每样本 CPU 转换则是明确事实。

更容易被忽略的是 UI 默认设置：`app/workspace.py:350-359` 的图像缩放默认值为 224，框架默认值也为 224（`uatf/config.py:23-25`）。用户把磁盘图片改为 384，不代表模型以 384 输入；需检查本次运行目录中的有效 `config.yaml`。缩放目标错误会影响模型行为和计算量。

### 3.3 DataLoader 配置：已有基础优化，但不是可测得的最佳值

当前代码并非完全缺少 DataLoader 优化：

- `uatf/experiment.py:77`：请求设备不是 CPU 且 CUDA 可用时设置 `pin_memory=True`。因此 `device=auto` 且 CUDA 可见时也会启用 pinned memory。
- `uatf/experiment.py:238-248` 和 `uatf/data/loaders.py:41-45`：`num_workers > 0` 时使用 `persistent_workers=True`、`prefetch_factor=2`。
- `uatf/training/trainer.py:432-433`：Tensor 到设备的拷贝传入 `non_blocking=True`。
- `app/workspace.py:508-515`：Web worker 数默认按 CPU 核心数计算，最高默认 4，允许用户配置到 16。
- `uatf/config.py:43`、`configs/default.yaml:29`：框架/YAML 默认 `num_workers=0`。通过 CLI 使用未覆盖该值的默认配置时，数据读取和变换都可能在训练主进程串行执行。

由此不能笼统说“Web 默认就是 0 worker”或“平台没有 pinned memory”。Web 初始值通常最多为 4，而 CLI/YAML 默认是 0；本次运行配置才是事实。worker 数没有自动 benchmark，预取深度固定为 2，也没有针对 NVMe、CPU 核心数、内存和图像解码耗时的调优策略。

worker 开到 16 也不等于更快。每个 worker 都有进程、解码和预取开销；Windows 使用 spawn 时，Dataset 状态和进程启动成本也应考虑。384×384 RGB float32 一个 batch=32 约 54 MiB；worker 数与 `prefetch_factor` 增大时，排队 batch 和 pinned 副本会占用大量主机内存。应逐档测量吞吐和 RAM，而不是将 CPU 占满作为目标。

### 3.4 没有图像预处理缓存

当前仓库中没有持久化的图像 Tensor cache，也没有 MONAI CacheDataset、LMDB、WebDataset、TFRecord 或 NVIDIA DALI 训练输入实现。`app/app.py:1127-1145` 的 archive cache index 和 `app/app.py:1180-1208` 的复用逻辑是 ZIP 解压目录缓存，不是解码/resize/normalize 缓存。操作系统文件缓存可能把近期 JPG 内容留在内存中，但它是非确定的 OS 行为，不是 UATF 提供的训练缓存。

因此当前每个 epoch 的成本会包含读取图片文件、JPEG 解码、PIL RGB 转换，以及配置指定的 transform。若磁盘图像仍然为 3000 px，每一轮都会为全部样本重新解码大图，再缩放到目标尺寸；若磁盘已经是 384 px，则主要重复成本变成小图读取/解码及其后续 CPU 转换。

### 3.5 AMP 默认关闭，Web 路径没有开关

AMP 默认值在 `uatf/config.py:51` 和 `configs/default.yaml:40-41` 中为 false。训练配置虽然支持 `training.amp.enabled`，Web 的 `build_training_config()` 在 `app/app.py:606-617` 中没有传入 amp 项，因此解析后仍然关闭。`uatf/training/trainer.py:612-615` 当前使用 CUDA FP16 autocast，构造 GradScaler，并在训练步骤中更新 scaler（`trainer.py:596-610`）。

RTX 5080 上应将 AMP 作为优先 benchmark 项。当前可直接用配置文件入口评估 `amp.enabled: true`；若目标工作流是 Web UI，则之后应增加明确的精度选项，而不是让 Web 用户误以为 GPU 已自动使用混合精度。BF16 可能更适合稳定性，但当前实现硬编码 FP16，需要另行设计 precision 配置和 scaler 行为。

### 3.6 固定图像尺寸下关闭 cuDNN benchmark

`uatf/experiment.py:661-663` 在 CUDA 可用时设置随机种子，并将 `torch.backends.cudnn.deterministic=True`、`torch.backends.cudnn.benchmark=False`。固定尺寸训练（例如所有 batch 都是 384×384）通常适合测试 `benchmark=True`，让 cuDNN 搜索卷积算法。

这是确定性的显式取舍，不是配置误读。开启 benchmark 并允许非确定性算法可能提高性能，但会降低严格复现保证。不同尺寸补边、最后一个较小 batch 或动态输入形状也会影响 benchmark 收益，因此必须以实际数据形状验证。

源码没有发现显式 `channels_last` memory format、`torch.compile` 或 TF32 精度策略。这些可作为后续可选项，但不能替代设备/配置确认、同步点优化和数据输入 profile。

### 3.7 每 batch 的 CUDA 标量提取可能压低流水线并行

`uatf/training/trainer.py` 中：

- `trainer.py:461`：每 batch 对 loss 做 `isfinite(...).all().item()`。
- `trainer.py:481-490`：每 batch 计算指标，并可能调用 Tensor `.item()`。
- `trainer.py:510-513`：每 batch 将 loss 转为 Python float 汇总。
- `trainer.py:515-550`：有 callback 时按时间间隔报告进度。

在 CUDA 上，`.item()` 会使主机等到对应 GPU 结果可用。训练 step 频繁时，反复将小标量取回 CPU，会给每 batch 加同步点，使 worker 预取更难与模型计算重叠。精确影响需用 profiler 量化，但这是源码可见的可优化热点。

改造方向是把 loss 总和、正确数、样本数等统计量保留为 detached GPU Tensor，在 epoch 结束或必要的进度间隔再一次性取回。非有限 loss 的保护仍需保留；可比较逐 batch 检查、每个优化器 step 检查或受控间隔检查的代价和诊断延迟。不要为了吞吐静默吞掉 NaN/Inf。

### 3.8 报告总耗时包含每轮验证和最终评估

`uatf/training/trainer.py:672-692` 的 fit 周期分别调用训练和验证，并记录 `time_seconds`。运行结束后 `uatf/experiment.py:888-900` 又调用 `evaluate_training_run()`，通常再次遍历验证集，并可能生成预测、指标和图表。`experiment.json` 的总体 `duration_seconds` 因此不能直接解释成纯训练时间。

性能报告至少应区分：数据准备、train epoch、validation epoch、最终验证/预测、checkpoint 写入、绘图与序列化。否则把最终验证误算成训练 epoch 会误导优化决策。

### 3.9 单卡实现；显卡资源监控不能替代阶段 profile

仓库搜索没有发现 `DistributedDataParallel`、`DataParallel` 或 `DistributedSampler`。`torch.cuda.manual_seed_all()` 只负责设置随机种子，不会把一个任务自动分到多张 GPU。当前训练路径按单设备将模型移动至 CUDA 或 CPU。

`app/monitoring.py:21-140` 会读取系统 CPU、显存、GPU 利用率、温度、功耗等 best-effort 快照，NVIDIA SMI 约每 5 秒采样一次。它没有记录每 batch/每阶段的耗时。训练进程 CPU 采样基于附加的 CLI 主进程，不等于汇总该进程所有 DataLoader 子进程，因此不能只凭该字段判断 worker 负载。

## 4. 优先级优化方案

### P0：先建立可复现的性能基线，排除运行环境错误

这一阶段不追求改模型代码，先为截图所示约 1 小时 50 分钟的运行建立可复核基线，并找出事实瓶颈。

1. 找到对应运行目录的 `config.yaml`、`experiment.json`、`metrics.json` 和 `logs/train.log`。
2. 从日志确认 `Resolved device` 是否为 `cuda`，设备名是否是 RTX 5080。检查 PyTorch 是否为 CUDA build、`torch.cuda.is_available()` 是否为 true。
3. 确认有效 `dataset.image_size`、`resize_enabled`、`batch_size`、`num_workers`、`amp.enabled`、训练/验证样本数和是否启用 augmentation。
4. 将完整运行时间拆成首次启动、每轮 train、每轮 validation、最终 evaluation、checkpoint 和报表输出。不要把“10 epoch 总耗时”当成单一训练核函数耗时。
5. 至少采样一个稳定 epoch 的 GPU 利用率、显存、功耗、温度/频率、CPU 各核利用率、RAM、磁盘读速率。监控采样应覆盖 DataLoader worker 子进程。
6. 使用 `torch.profiler` 对 warmup 后的若干 step 记录 CPU/CUDA activity、shape、memory 和 schedule；关注 `enumerate(DataLoader)` 等待时间、PIL/torchvision CPU 时间、`Memcpy HtoD`、卷积 kernel、Python scalar sync 和 optimizer。
7. 对输入与模型单独做基准：只迭代 DataLoader 测 images/s；预先构造 GPU Tensor 测纯 forward/backward samples/s。两者差距可快速显示瓶颈在输入还是 GPU 计算。

建议临时测量的核心量：

```text
train_images_per_second = train_samples / train_seconds
validation_images_per_second = validation_samples / validation_seconds
epoch_wall_seconds = train_seconds + validation_seconds + epoch_save_seconds
```

CUDA 计时需要在基准边界显式 `torch.cuda.synchronize()`，但不能把 synchronize 放进正式训练的每个 batch，否则人为引入串行等待。

**P0 的退出条件：** 能明确说出此次运行实际使用的设备/输入尺寸/精度；能分辨 DataLoader、H2D、前后向和验证占时；同一配置至少重复测量并报告中位数。若已确认使用 CUDA 但 GPU 长期低利用率，先优化数据管线/同步；若 GPU 持续饱和，先优化模型精度、内存格式和计算路径。

### P1：启用现有 CUDA 快速路径并调 worker

配置驱动路径建议从以下基准开始，随后逐项测量：

```yaml
dataset:
  image_size: 384
  resize_enabled: true

training:
  device: auto
  batch_size: 32
  num_workers: 8
  amp:
    enabled: true
```

`pin_memory` 由代码根据 CUDA 可用性自动设置，不需要虚构一个当前不存在的 YAML 键。`prefetch_factor=2` 是当前基线。

- 如果 Web 工作流是主要入口，给 AMP 增加明确 UI 控件并写入共享配置 schema；当前仅通过 YAML/CLI 才能启用 AMP。
- DataLoader worker 依次比较 0、2、4、8、12（机器有足够 CPU/RAM 时再测 16）。一次只改变 worker 数，记录样本/秒、GPU 利用率、CPU、RAM 和数据等待 p95。
- 预取深度可在实现配置项后比较 2、4，只有 GPU 因数据供应断档且 RAM 有余量时才提高。prefetch 不会让单个 CPU 解码 worker 计算得更快。
- AMP 下先保留 batch 32，确认显存和数值稳定后再测试 48/64。提高 batch 会改变优化器更新次数和有效优化动态，不能只按吞吐决定。
- 在完整模型与目标输入尺寸下测试 AMP，而非仅以 CUDA smoke test 或小 Tensor 运算代表训练性能。

### P1：修正 CUDA 训练循环里的同步统计

将 loss/accuracy 的统计转为设备端累加，减少 Python 每 batch 标量提取。示意：

```python
loss_sum_device += loss.detach() * batch_size
correct_device += (predictions.argmax(1) == targets).sum()
sample_count += batch_size

# epoch end (or a throttled progress event): transfer scalar totals once
mean_loss = (loss_sum_device / sample_count).item()
accuracy = (correct_device / sample_count).item()
```

实际实现还需考虑训练/验证、多标签指标、事件 callback、AMP scaler 和 stop/checkpoint 语义。对 loss 有限性检查应采用明确策略：保留每步检查但接受同步成本，或降低检查频率并明确检测延迟；不得静默过滤非有限值。优化后用 PyTorch profiler 确认同步事件减少，同时检查结果统计与原逻辑一致。

### P1：为固定形状增加可选的吞吐配置

给训练设置提供明确的 performance mode 与 reproducible mode，而不是在 `_execute_experiment()` 中无条件固定 cuDNN 策略。

建议基准：

```python
if fixed_input_shapes and performance_mode:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
```

再在 ConvNeXt 上比较 `channels_last`：模型与输入需使用一致的 memory format，且每条模型/恢复/评估路径都要兼容。记录格式与 precision 到有效配置和 metrics。

`torch.compile(model)` 可作为第二阶段实验。它可能带来首次编译等待、额外显存、图断裂或 Windows/PyTorch 版本兼容问题；固定输入 shape 更适合编译。对 10 epoch 的任务，需把 compile warmup 纳入总墙钟时间，不能只报告编译后的稳态 step。

### P2：增加一次构建、反复读取的 uint8 图像缓存

#### 建议格式

384×384×3 uint8 的 30,000 张固定形状缓存约 13.27 GB（12.36 GiB）；float32 缓存约 53.08 GB（49.43 GiB）。因此优先考虑磁盘上的 `numpy.memmap` 固定形状数组或 LMDB uint8 值，不建议默认把全部 float32 Tensor 放主机内存。

- **memmap：** 对固定形状、整数索引、随机读取任务实现最简单；可由 worker 懒加载同一个文件，依赖操作系统页缓存。
- **LMDB：** 适合按稳定 key 随机取值和多 worker 并发，需管理环境大小、只读事务和 map size。
- **WebDataset/tar shards：** 适合连续分片读取和大规模顺序数据流；随机访问通常需要 shuffle buffer，需重建当前 map-style split 和样本顺序契约。
- cache 仅存 uint8 RGB 或经审查的灰度表示；训练批次可先传 uint8 到 GPU，再做 float conversion、缩放到 [0,1] 和 Normalize，以降低 H2D 数据量。灰度压缩为单通道会改变当前 RGB 输入表示，默认先保持三通道兼容。

#### 推荐的数据链路

```text
离线构建：源 JPG -> RGB 解码 -> 与现有设置一致的 Resize(384, 384) -> uint8 memmap/LMDB
训练读取：cache[index] -> 随机训练 augmentation（如启用）-> uint8 batch -> pinned H2D
GPU batch transform：uint8 -> float -> /255 -> RGB Normalize -> ConvNeXt
验证读取：同一确定性图像 cache -> 不做随机增强 -> GPU Normalize -> 验证
```

磁盘 cache 构建代码的核心可按以下方式设计；这是实现示意，不是已提交代码：

```python
images = np.memmap(
    cache_path, mode="w+", dtype=np.uint8,
    shape=(sample_count, 384, 384, 3),
)
for index, path in enumerate(paths):
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        resized = resize_with_current_policy(rgb, (384, 384))
        images[index] = np.asarray(resized, dtype=np.uint8)
images.flush()
```

worker 读取端应按需打开 memmap，避免 Windows spawn 时将整张映射对象或整份数据复制进每个进程：

```python
def __getitem__(self, index):
    image = Image.fromarray(np.array(self._open_cache()[index], copy=True), "RGB")
    image = self.transform_without_resize(image)
    return image, self.targets[index]
```

启用 GPU Normalize 的实现则可让 Dataset 返回 HWC uint8，collate 后批量搬至 CUDA：

```python
x = batch_u8.to(device, non_blocking=True).permute(0, 3, 1, 2)
x = x.float().div_(255.0)
x = (x - mean[None, :, None, None]) / std[None, :, None, None]
```

上述链路要核实 CUDA 上的数据布局、dtype、AMP autocast 和 channels_last 组合；不要在 `__getitem__` 里把 CUDA Tensor 返回给 DataLoader worker。

#### Cache 一致性和平台契约

1. cache key 至少包含数据集 manifest/标签版本、每张图源标识、尺寸、RGB 转换、resize 插值策略和 cache schema 版本。不要只用目录路径作为 key。
2. 缓存写入到临时文件，全部完成后 flush、校验样本数/shape/labels，再原子 rename 并写完成标记。失败或进程中断后不可把半成品当作有效缓存。
3. 提供 cache lock 或唯一构建者，避免两个 Web 任务同时构建/覆盖同一 cache。
4. Dataset wrapper 必须保持 `.samples`、稳定的样本顺序、标签/类别映射和 `with_transform()` 行为。`uatf/experiment.py:351-376` 和 `uatf/experiment.py:379-432` 通过 Dataset 样本路径与 Subset 索引生成实验 manifest；丢失这些属性会破坏实验记录。
5. 保持当前 train/validation split 与源样本 ID 关系；缓存实现不能因排序或并行写入而改变标签配对。
6. 随机增强不能固化进全局缓存。否则所有 epoch 看到相同变换结果。更标准的做法是缓存确定性预处理结果，再按 epoch 重新执行随机增强。
7. 当前 transform 顺序是 random augmentation -> Resize -> Tensor -> Normalize。把 cache resize 提到 augmentation 前会改变原图仍为 3000 px 时的插值/增强顺序。若输入 JPG 已是 384 且目标确实是 384，该差异可能很小；否则必须作为数据行为变化单独验证，并在模型预处理元数据中记录。
8. 如果 cache 保存了 384 图像，cache Dataset 的训练/验证 transform 应跳过重复 Resize，但模型资产与推理预处理元数据仍要记录正确的 384 目标尺寸。不要简单把全局 `resize_enabled` 改成 false，否则推理流程可能开始保留任意输入尺寸。

#### 预期收益判断

缓存是否值得做取决于 profile：如果数据等待占总时间较高、GPU 经常空闲，uint8 memmap/LMDB 能减少 JPEG decode 和每文件随机 I/O；如果 GPU 已持续饱和，缓存不会显著缩短训练时间。当前图片已经是 384 JPG 时，缓存收益主要来自避免每轮 JPEG 解码和小文件访问，不会减少 ConvNeXt 的卷积 FLOPs。

### P2/P3：仅在 profile 证明数据输入受限时评估 NVIDIA DALI

DALI 可以将 JPEG decode、resize、normalize 和部分增强放在 mixed/GPU 管线执行。简化示意如下：

```python
encoded, label = fn.readers.file(file_root=image_root, random_shuffle=True)
image = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
image = fn.resize(
    image, resize_x=384, resize_y=384,
    interp_type=types.INTERP_LINEAR,
)
image = fn.crop_mirror_normalize(
    image,
    dtype=types.FLOAT,
    output_layout="CHW",
    mean=[m * 255.0 for m in rgb_mean],
    std=[s * 255.0 for s in rgb_std],
    mirror=fn.random.coin_flip(),
)
```

此片段展示算子位置，不是可直接替换当前 Dataset 的完整代码。DALI 归一化参数需匹配输入像素量纲；当前 ToTensor 先缩到 [0,1]，所以若 DALI 直接接收 [0,255] uint8，mean/std 需要对应换算。还需核对 Resize 插值、固定方形拉伸、RGB 通道、水平翻转概率和增强顺序与当前代码相同。

引入 DALI 的成本包括：Dataset/DataLoader 替换为 DALI pipeline/iterator、保持 split 与标签映射、在 manifest 中保留稳定样本路径、适配验证/最终评估顺序、处理多标签 CSV、Windows/目标部署环境兼容和 worker 生命周期。DALI 仍需每轮从磁盘读取 JPEG；GPU decode/resize 也会和 ConvNeXt 抢同一 GPU 的带宽/算力。缓存后 GPU 已饱和时，DALI 可能没有收益甚至更慢。因此它是 profile 证明 CPU decode/transform 瓶颈后的方案，不是首选开关。

### P3：未来多 GPU

如果单卡已经被有效打满而任务仍达不到目标，才评估 DDP。需要增加进程组管理、DistributedSampler、每 rank batch 与梯度语义、checkpoint rank 规则、指标聚合、断点恢复和 UI 任务生命周期。只调用 `cuda.manual_seed_all` 不会实现多卡。多卡不会修复单卡落 CPU、DataLoader 卡顿或逐 batch 同步等基础问题。

## 5. 建议的实施顺序与验收门槛

| 阶段 | 实施范围 | 最低验收标准 |
|---|---|---|
| 0. 测量基线 | 确认实际 CUDA 设备、有效配置、图像 shape、profile 和各阶段耗时 | 报告纯 train、validation、final eval 秒数和 samples/s；能判断 GPU-bound 或 input-bound |
| 1. 低风险训练配置 | 确认 CUDA；启用现有 AMP；worker 梯度测试；保留 pin memory；校验 image_size | 记录稳态吞吐、显存峰值、RAM 和验证指标；至少复测三次或报告中位数 |
| 2. 热循环/算子优化 | 减少 `.item()`；提供 cuDNN performance/reproducible 模式；测试 channels_last | profiler 显示 host sync 减少；验证 loss/metric 与原实现语义一致；恢复训练仍正确 |
| 3. uint8 持久缓存 | memmap/LMDB builder、缓存版本与失效、Dataset wrapper、GPU batch Normalize | cache 与现有确定性预处理输出匹配；label/split/manifest 稳定；多 worker 与中断构建可恢复 |
| 4. DALI（条件性） | 仅在 profile 显示 CPU decode/transform 仍是瓶颈时接入 | 对比同一数据、同一 batch、同一训练语义的 images/s 和端到端 epoch 时间；收益超过复杂度 |
| 5. 多 GPU（需求性） | DDP 与分布式任务/实验记录支持 | 有单卡已充分利用的证据和多卡端到端扩展效率数据 |

对当前 30,000 张、384、ConvNeXt-Tiny、batch 32 的目标，统一采用“每 epoch train+validation”计时。2–3 分钟对应 167–250 样本/秒的全量遍历等效速度。若验收指标只算训练 split（约 24,000 张），训练段目标另行报告约 133–200 张/秒，再单列验证时间。不要将两种口径混在一起。

每轮结果至少报告：

- `train_samples`、`validation_samples`、每阶段秒数、samples/s。
- 实际设备名、PyTorch/CUDA/cuDNN 版本、AMP dtype、输入分辨率、batch size、workers、prefetch factor、memory format。
- GPU 平均/中位利用率、显存峰值、功耗/温度/频率；CPU 每核负载、训练主进程与 worker 总 CPU、RAM、磁盘读取。
- 是否包含首 epoch warmup、compile 时间、最终验证、预测 CSV、曲线绘图和 checkpoint 保存。
- 质量对比：相同 split、相同标签、相同评价指标，比较收敛、最终验证指标和关键类别表现。

## 6. 数值、数据和兼容性风险

### 精度风险

- AMP FP16 会改变浮点舍入范围；当前 GradScaler 可缓解梯度下溢，但不能保证所有模型/数据都与 FP32 完全相同。需要观察 loss、梯度、NaN/Inf 和验证指标。
- 若增加 BF16，需明确 autocast dtype 与 GradScaler 策略。不要把 BF16 当作只改一个配置字符串；测试优化器、loss、metric、checkpoint resume。

### 随机性与可复现风险

- `cudnn.benchmark=True` 和 `deterministic=False` 可能使相同 seed 的结果不再逐位相同。
- worker 数改变随机变换/RNG 消耗顺序。检查当前 worker seeding、恢复 checkpoint 后的 DataLoader RNG 和 augmentation 重现语义。
- cache 的图像顺序、标签映射或 split 错误会造成静默数据错误，比性能变慢更危险。

### 模型语义风险

- 当前 `Resize((H, W))` 会缩放到固定矩形，可能直接改变宽高比；离线缓存必须复现相同的几何策略，否则输入分布变化。
- 384 降为 320/224 可按像素面积明显降低 ConvNet 计算量，但可能损失小病灶细节。胸片任务必须以标签和临床相关特征为准，不能只凭吞吐选尺寸。
- 当前 UI 的增强开关启用后默认可能包含 p=0.5 水平翻转。胸片侧别标识、左右侧标签和单侧表现可能使水平翻转改变语义，需任务负责人审核；这不是纯性能参数。
- 当前代码对输入执行 RGB 转换。若改用单通道缓存以省空间/带宽，必须验证 Normalize、模型 stem、预训练权重和推理预处理兼容性。

### 内存、文件与任务恢复风险

- uint8 全量缓存约 12.36 GiB，训练/验证 worker、预取 batch、文件 cache 和 Web 服务还会另占主机 RAM。float32 全量缓存约 49.43 GiB，常见主机无法无代价常驻。
- `num_workers` 和 `prefetch_factor` 直接影响队列 RAM 与 pinned memory。内存不足可能转为分页，速度比原配置更差。
- `persistent_workers` 在 Windows 多进程和模型训练/验证交替中需要长时间跑验证，检查 worker 是否泄漏、挂起或重复读取。
- 对 cache 做指纹、schema 版本、原子写入、锁和空间检查；不能覆写用户源图像或破坏现有实验目录。

### DALI/多卡架构风险

- DALI reader/iterator 不自动满足当前 map-style Dataset 的 `.samples`、Subset 索引、manifest、随机 split、CSV 多标签和 checkpoint 恢复契约。
- GPU 数据增强和 decode 会占用同一显卡资源；必须比较端到端吞吐，不可只展示数据 pipeline 的独立速度。
- DDP 会改变全局/每卡 batch 语义、梯度平均、随机数、checkpoint 与 UI 进程管理；这是架构扩展，不是无风险性能补丁。

## 7. 最终锐评

当前平台性能工程最差的地方，是缺少证据闭环：它把训练跑完并记录结果，却没有把“GPU 等数据、CPU 同步、卷积计算、验证、最终导出”分别量出来。没有阶段数据时，团队无法知道自己应该优先做 AMP、缓存、worker 调优、同步点治理还是排查设备落错。

其次，代码选择了重复解码与逐样本变换的通用输入路径，没有持久预处理缓存；Web 默认路径又没有 AMP 控件，固定尺寸 CUDA 训练还关闭 cuDNN benchmark。它并非完全没有基础优化（CUDA pinned memory、non-blocking copy、worker persistence 和 prefetch 已存在），但这些配置是少量通用默认值，不是经过数据和硬件测量的吞吐策略。

每 batch 的 CUDA 标量提取是最扎眼的训练循环问题；缺少 cache 是最扎眼的数据管线缺口；缺少阶段 profile 是最扎眼的平台工程缺口。截图对应的 ETA 粗算约 11 分钟/epoch、约 45 张/秒（假设每轮处理 30,000 张图），与 2–3 分钟/epoch 的目标仍有约 3.7–5.5 倍差距。这个差距值得认真拆解，但截图本身还不能证明瓶颈在 GPU、CPU 解码还是验证阶段，也不能据此断言平台达不到某个固定的“行业均值”。应先核对有效 `Resolved device` 和分阶段 GPU 利用率，再用数据证明优化方向。

最终应以稳态端到端 benchmark 和模型质量共同验收。不要以“AMP 已开”“DALI 已接入”“GPU 利用率看起来很高”或“worker 已经设成 16”作为性能完成标准；唯一可信的性能结论是同样数据、同样 split、同样输入与质量约束下，测得了可重复的 epoch 时间和吞吐提升。
