# UATF GUI/UX 审计与功能迁移清单

## 审计基线

- 审计对象：当前工作区中的 `app/app.py`、`app/monitoring.py`、`uatf/`、`configs/`、GUI 单元测试及 Phase 9 文档。
- 当前入口：`python -m app.app`；`app/app.py` 的 `build_app()` 构建单个 `gr.Blocks`，模块主体纵向放置，历史与推理功能收在 Accordion 中，没有一级模块导航。
- 实际运行环境：Python 3.12.14、Gradio 6.29.1、PyTorch 2.7.1+cu128，`torch.cuda.is_available()` 为 true。项目声明的最低 Gradio 版本为 4.44；实现需继续使用兼容范围内的稳定组件。
- 训练边界：GUI 生成共享配置 schema 的 YAML 并启动 `uatf.cli` 子进程；训练、数据读取、模型、检查点、实验记录、评估和推理由 `uatf` 后端承担。不得因界面迁移改写这些业务逻辑或配置字段。
- 已有 GUI 验收测试：`tests/unit/test_app_dataset_workflow.py`、`test_app_monitoring.py`、`test_app_experiment_management.py`；另有评估/推理、统一 API/配置、任务生命周期和实验管理测试。现存测试覆盖回调/API 行为，不覆盖完整浏览器导航流程。
- 工作区审计时存在大量未提交和未跟踪改动。本文和后续重构以其当前内容为基线；不清理、不覆盖无关改动。

## 页面信息架构

主工作台采用左侧模块导航，页面在同一 Blocks 应用中按需显示，以便 Gradio 组件状态在切换时继续保留。

| 新模块 | 承载内容 |
| --- | --- |
| 工作台 | 最近实验和最近任务的真实状态、进入数据集/训练/监控的快捷入口；不生成示例指标。 |
| 数据集 | 上传 ZIP 或服务端路径、任务/格式、划分/种子/图像尺寸、真实检查报告、类别分布、样本预览和警告确认。 |
| 训练 | 已检查数据集摘要、现有模型选择、Epoch、Batch Size、学习率、设备、增强设置、启动校验与启动操作。 |
| 训练监控 | 任务 ID/状态/阶段、Epoch/批次/进度/耗时/ETA、损失与验证指标、学习率、曲线、资源、日志/错误、评估结果、检查点/文件、刷新和停止。启动后导航到此页。 |
| 实验管理 | 名称/模型/状态/时间筛选、实验列表、主从式实验详情、指标/历史/日志/检查点下载、实验比较、载入 YAML 并启动新实验。 |
| 模型管理 | 资产列表、资产详情、加载模型、导出权重与 JSON 侧车下载。检查点仍归属实验/任务详情，不与模型资产混为一类。 |
| 评估与推理 | 共享真实模型资产选择器；二级视图为独立评估、单图推理、批量目录推理。保留各自输入、状态、真实结果、错误样本查看和 CSV 下载。 |
| 设置与工具 | 本轮代码未发现独立设置/诊断/帮助操作入口；当前硬件信息属于训练监控，启动说明位于 `docs/phase9-gradio-ui.md`。不新建伪功能。 |

## 新旧结构差异

- 原结构将数据集表单、训练参数、实时监控放在同一主页面，实验/模型和评估/推理放在纵向 Accordion；新结构用左侧导航把 7 个业务模块分开，保留同一 Blocks 应用和组件状态。
- 数据集检查与训练参数分属两页；训练启动只有拿到 task ID 才切换到训练监控，校验失败仍停留在训练页。实验复制配置启动也进入监控，并明确提示需刷新持久任务状态。
- 实验列表和详情采用主从布局，比较放入独立二级视图；模型资产单独浏览和操作；评估、单图、批量推理保留各自输入与真实输出。
- 本轮没有新增系统设置、诊断或帮助功能，也没有删除 GUI 之外的 YAML/CLI/Python API 能力。

## 实施文件与职责

| 文件 | 职责 |
| --- | --- |
| `app/app.py` | 保留数据、训练、实验、资产、评估与推理回调；增加工作区样式/页名、真实工作台摘要、任务产物读取及启动路由包装器；`build_app()` 委托给新布局模块，启动时配置队列并发。 |
| `app/workspace.py` | 新建 Gradio 左侧导航和 7 个页面；重新组织现有控件；保持现有回调/API 的输入输出连接。 |
| `tests/unit/test_app_workspace.py` | 验证导航包含独立训练/监控入口、训练拿到 task ID 后才切页、复制配置启动状态提示准确。 |
| `docs/development/gui-ux-audit.md` | 功能留存清单、新旧位置映射、字段基线、设计系统和验收记录。 |
| `docs/phase9-gradio-ui.md` | 新工作区与设计规范的使用说明。 |
| `docs/development/v2.0.0-phase8-manual-gui-uat.md` | 将现有人工验收步骤对应到新模块名称和导航路径。 |

## 设计系统

- 颜色：画布 `#f4f7f5`、表面白 `#ffffff`、正文 `#202b26`、次级文字 `#63716a`、边框 `#d9e1dc`；主操作/选中态使用 UATF 绿 `#267256` / `#e8f2ed`，字段标签使用透明背景和中性文字，危险停止使用 `#ad3f42`，橙色 `#bd6439` 留作品牌强调，不铺满页面。
- 字体：Inter 优先，回退到系统无衬线字体、Windows Segoe UI；页面标题 25px（窄桌面 22px），区块标题 16px，正文与辅助文字沿用 Gradio Soft 主题基础字号，不用视口宽度缩放字体。
- 间距：按 4px 基数组织；表单字段间 8–12px，内容区块 14px，主页面内边距 16–20px，左右栏间距 18px。
- 形状：常用控件圆角 5px，面板/导航边框圆角 6px；无渐变、毛玻璃和重阴影。
- 页面标题：统一 H1 与一行页面上下文，底部细边框区分导航与内容。
- 字段组：将数据来源/划分与训练参数按职责分区；高级字段不在本版伪造，现有控件仍直接写入原配置生成器。
- 状态与反馈：绿色表示可用/当前选择，停止采用危险色；校验和运行信息就地显示；未检查的数据集、未打开的实验/模型详情、尚无结果的评估/推理会显示下一步说明，真实结果组件仅在对应操作后展开。
- 表格：列表优先展示名称、状态、任务、模型、主要指标和时间等浏览字段；实验/模型完整记录仍可在详情读取。表格限制最大高度并由组件自身滚动，长文本换行，不要求在常见桌面宽度下横向滚动才能浏览关键字段。
- 图表：继续使用真实训练 Epoch 数据及现有 LinePlot，明确标题/epoch/value/series，保留 Gradio 全屏和导出操作。
- 详情：实验详情、比较、监控日志/产物使用二级视图，不把不同职责并排堆满一个页面。

## 功能保留清单

| 功能名称 | 现有位置 | 关联回调/API | 状态与数据 | 新页面位置 | 验收方法 |
| --- | --- | --- | --- | --- | --- |
| 数据集 ZIP 导入、安全解压与缓存 | `app/app.py`: `_safe_extract_dataset`; 数据集表单 | `inspect_dataset_ui` → `_inspect_selected_dataset`; `uatf.data.inspect_dataset` | 上传文件路径、任务/格式、缓存索引、解压路径、数据集 ID | 数据集 | 有效 ZIP 可重复命中缓存；不安全路径/符号链接被拒绝；已有 `test_app_dataset_workflow.py` 覆盖。 |
| 本地目录/CSV 数据集导入 | 数据集表单 | `inspect_dataset_ui`; `uatf.data.inspect_dataset` | 服务端数据路径、`image_folder`/`csv_classification`/`csv_multilabel`、可选 image root | 数据集 | 用文件夹、单标签 CSV、多标签 CSV 各跑一次检查；无效路径与不兼容任务格式得到阻断提示。 |
| 训练/验证拆分与种子 | 数据集表单 | `_inspect_selected_dataset`; `build_training_config` | `validation_fraction` 默认 0.2、`seed` 默认 42；检查报告包含真实拆分结果 | 数据集 | 改比例/种子后重检，确认报告、配置保持相同值；旧检查状态不匹配时训练被拒绝。 |
| 数据集校验、警告、类别分布及样本预览 | 数据集检查结果区 | `_inspect_selected_dataset`, `_save_distribution_chart`, `_class_distribution_rows`, `_preview_at`, `turn_sample` | `dataset_state`、检查 fingerprint、errors/warnings、统计表、真实图片与预处理预览 | 数据集 | 检查结果显示真实样本/标签与拆分；上一张/下一张可切换；警告必须确认；阻断错误不能训练。 |
| 训练配置字段及 YAML 生成 | 原训练配置组；`build_training_config` | `start_training`; `uatf.config` 共享 schema | 任务/格式/路径、验证比例、种子、尺寸、模型、Epoch 默认 3、Batch Size 默认 8、LR 默认 0.001、设备默认 `auto`、增强默认 false；输出目录和主要指标 | 训练；数据集字段仍由数据集页配置 | 对比启动后生成 YAML 与当前 GUI 字段及默认值；负 Epoch/Batch/LR、数据集状态变更均应被拒绝。 |
| 训练启动、异步子进程及防重复任务 | `start_training` 生成器 | 启动 `python -m uatf.cli --config ... --events-file ... --task-id ...`; CLI 与 `TaskManager` | 持久 task ID、进程输出队列、结构化 JSONL 事件、experiment/result path | 训练；启动后自动进入训练监控 | 短 CPU 训练启动一次；配置和事件文件绑定同 task ID；重复启动受现有活动任务锁阻止；切换其他页面时进程持续。 |
| 实时状态、进度、指标与训练曲线 | 当前主页面训练面板 | `start_training` 消费 `training_started`、`training_progress`、`epoch_completed` 等事件；`refresh_task_status` | 状态/阶段、Epoch/批次、总体进度、耗时/ETA、train/validation loss、主要指标、最佳 epoch、学习率、epoch 耗时 | 训练监控 | 短真实训练期间核对事件更新进度、图表和指标；完成后曲线包含真实 Epoch 记录；无任务显示空状态。 |
| 训练日志、错误摘要/堆栈、资源采样 | 当前主页面训练面板；`app/monitoring.py` | `start_training`, `_read_log_tail`, `_technical_error`, `HardwareResourceSampler` | 最近 200 行、完整日志路径、错误摘要、堆栈、CPU/RAM/GPU/NVML 或 `nvidia-smi` | 训练监控 | 训练期间查看实时与磁盘日志；人工制造无效路径验证提示；硬件值与采样器一致，缺少可选采样器时显示“不可用”。 |
| 停止、任务恢复/查询 | 当前主页面训练面板 | `request_training_stop`; `refresh_task_status`; `uatf.tasks.get_task/get_latest_task/request_task_stop` | 持久任务状态、epoch、stage、result dir、stop 请求；停止在 Epoch 边界生效 | 训练监控；工作台只摘要当前任务 | 运行中请求停止并确认 `Stopped`、检查点保留；重启 GUI 后用最新/指定 task ID 刷新；已有 `test_app_monitoring.py` 覆盖持久查询/停止。 |
| 最终评估结果和最佳模型下载 | 当前主页面训练面板 | `start_training` 读取 CLI 事件/产物；`uatf.evaluation` | 独立验证集 metrics JSON、最佳模型文件、日志/实验目录 | 训练监控结果区；实验详情可再访问 | 短真实训练完成后核对 metrics 与实际实验文件；模型下载路径存在且非占位值。 |
| 实验筛选与列表 | “历史实验与模型资产” Accordion | `refresh_experiment_browser`; `uatf.experiments.list_experiments` | 名称、模型、状态、创建起止时间筛选；id/name/date/status/task/model/dataset/best metric/duration/legacy | 实验管理列表 | 创建/读取历史 fixture 后按每个筛选项查询，刷新保留可选记录；列表字段对应磁盘记录。 |
| 实验详情、配置、指标、曲线、日志、检查点/下载 | 历史 Accordion 详情区 | `load_experiment_details`; `get_experiment` | record、metrics、history、日志、checkpoint 元信息与 best/latest 路径 | 实验管理详情面板/子视图；监控提供当前任务视图 | 选择真实实验逐项核对详情、曲线和文件路径；两类检查点可下载；旧实验缺文件时呈现空值而不伪造。 |
| 实验比较 | 历史 Accordion | `compare_selected_experiments`; `compare_experiments` | 多选实验 ID、可比较性、警告和指标/配置差异 | 实验管理比较视图 | 选择兼容及不兼容实验各比较一次；复用后端可比性判断；数据不足展示真实警告。 |
| 载入历史配置并启动新实验 | 历史 Accordion 的 YAML 编辑器与操作区 | `copy_experiment_configuration`; `launch_copied_configuration` | YAML 文本可编辑；重设 GUI 输出目录；新 task ID 和状态 | 实验管理详情操作；跳转训练监控追踪新任务 | 载入历史配置，检查字段保留，再启动新 task；确认仍走 CLI/任务锁；已有实验管理测试覆盖命令构造。 |
| 模型资产列表与详情 | 历史 Accordion 的模型表格 | `refresh_model_asset_browser`, `load_model_asset_record`; `list_model_assets` | 资产 ID/name/architecture/task/experiment/weights/date/available/type/version 与完整记录 | 模型管理列表/详情 | 刷新后表格和选择器使用同一真实资产集合；详情与记录字段一致；无资产时为空。 |
| 加载模型资产 | 历史 Accordion | `load_model_asset_ui`; `load_model_asset` | `loaded_model_asset` Gradio State、错误/成功状态 | 模型管理详情操作 | 对真实资产加载并检查模型架构；无效/缺失资产显示错误；推理仍使用资产 ID 的真实后端接口。 |
| 导出模型和侧车文件 | 历史 Accordion | `export_model_asset_ui`; `export_model_asset` | PyTorch 权重路径及 `.json` 侧车路径 | 模型管理详情操作 | 导出后确认两个路径存在、文件可下载且侧车描述结构/预处理；不覆盖业务资产。 |
| 独立评估 | “独立评估与本地推理” Accordion 的“独立评估” Tab | `run_asset_evaluation_ui`; `evaluate_model_asset` | 资产 ID、数据路径/格式/image root、batch 默认 32、threshold 默认 0.5；总体指标、逐类指标、混淆矩阵、ROC/PR、预测、错误过滤和原图预览 | 评估与推理 / 独立评估 | 用单标签与多标签数据运行真实评估；核对 artifacts/metrics、预测表和错误筛选/图片；现有 `test_independent_evaluation.py` 覆盖后端流程。 |
| 单张图片推理 | 同 Accordion 的“单张图片” Tab | `run_asset_image_prediction_ui`; `predict_model_asset_image` | 模型资产 ID、上传图像、阈值；预测、类别分数/概率、原始 JSON | 评估与推理 / 单张图片 | 上传实际图片执行，核对后端返回的预测和分数；缺少资产/图片及无效输入提示明确。 |
| 批量图片推理及 CSV | 同 Accordion 的“批量图片” Tab | `run_asset_directory_prediction_ui`; `predict_model_asset_directory`, `prediction_rows` | 服务端目录、阈值默认 0.5、Batch Size 默认 32；结果表和 CSV 文件 | 评估与推理 / 批量图片 | 对含至少两张图的真实目录执行；结果数量正确，CSV 存在且可下载；已有后端评估/推理测试可复用。 |

## 配置字段基线

当前 GUI 明确暴露并传入训练配置的字段如下。页面迁移不得修改字段名、默认值、含义或生效位置。

| 配置字段 | 当前来源与默认值 |
| --- | --- |
| `task.type` | 任务选择，默认 `classification`；选项 `classification` / `multi_label`。 |
| `dataset.format` | 由任务联动，默认 `image_folder`；支持 `image_folder` / `csv_classification` / `csv_multilabel`。 |
| `dataset.path` | ZIP 解压后的检查路径或服务端本地目录/CSV；CSV 使用实际解析出的 annotation CSV。 |
| `dataset.image_dir` | CSV 数据集检查结果提供时传入。 |
| `dataset.validation_fraction` | 默认 `0.2`，滑块范围 0.05–0.5，步长 0.05。 |
| `dataset.seed` / `training.seed` | 默认 `42`；检查数据集和训练配置共用该值。 |
| `dataset.image_size` | 默认 `224`，正整数；同时影响输入变换。 |
| `dataset.mean` / `dataset.std` | 固定为当前 ImageNet RGB 归一化常量 `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`。 |
| `dataset.augmentation.enabled` | 默认 `false`，由当前增强复选框控制。 |
| `model.name` | 注册模型下拉选择，优先默认 `resnet50`，否则选第一项。 |
| `model.pretrained` | 当前 GUI 生成器固定为 `false`，不是独立可编辑控件。 |
| `training.epochs` | 默认 `3`，正整数。 |
| `training.batch_size` | 默认 `8`，正整数。 |
| `training.learning_rate` | 默认 `0.001`，正有限值。 |
| `training.device` | 默认 `auto`；支持 `auto` / `cuda` / `cpu`。 |
| `training.num_workers` | 当前 GUI 固定为 `0`。 |
| `evaluation.primary_metric` | 单标签 `accuracy`；多标签 `micro_f1`。 |
| `output.directory` | 固定为应用的 `outputs/ui_runs/`。 |

完整共享 schema 以 `uatf/config.py`、`configs/default.yaml` 和 `docs/configuration.md` 为准；GUI 目前只开放该 schema 的一部分，不在重构中擅自声称覆盖全部 CLI/YAML/API 配置。

## 现有验证入口与审计结论

- 启动方式：`\.venv\Scripts\python.exe -m app.app`；当前入口调用 `build_app().queue(default_concurrency_limit=4).launch(...)`，页面与事件绑定在 `app/workspace.py`，训练/数据/管理回调留在 `app/app.py`。
- 自动化测试：从仓库根目录运行 `\.venv\Scripts\python.exe -m pytest`；模块测试可单独运行 `tests/unit/test_app_dataset_workflow.py`、`test_app_monitoring.py`、`test_app_experiment_management.py` 与 `test_independent_evaluation.py`。
- 验收日期：2026-10-15。初版 GUI 构建成功，配置中包含 274 个组件、40 个事件依赖和 7 个导航入口；Python 编译和 Ruff 检查通过。
- 完整测试：`python -m pytest -p no:cacheprovider --basetemp tests/.uatf-ui-refactor-full-20261015 -q`，124 passed（32.59 秒）。
- 服务启动：`python -m app.app` 成功保持运行，`http://127.0.0.1:7860` 根页面返回 HTTP 200。
- GPU 训练：使用现有 CLI、`resnet50`、1 Epoch、CUDA，在 NVIDIA GeForce RTX 5080 Laptop GPU 上完成训练；事件包含 `training_started`、`training_progress`、`epoch_completed`、`training_completed`，并生成 best/last checkpoint 与 `metrics.json`。训练退出码为 0；Matplotlib 对用户目录字体缓存锁报告权限警告，不影响训练结果。
- 浏览器验收限制：本机浏览器工具尝试打开 localhost 时被自动审批拦截，审批服务返回 404，未授予浏览器网络访问授权。按拦截提示没有换浏览器或使用替代请求方式；因此真实浏览器点击流程、响应式截图和页面切换期间训练仍更新尚未确认。
- 已验证事件回调迁移、页面组与 7 项导航配置、task ID 存在时自动跳转监控的回调辅助函数；数据集/监控/实验/评估/API 行为由完整测试覆盖。
- 未验证项：浏览器端 ZIP/CSV 文件交互、训练页实际点击后的自动跳转、运行中切换模块再返回、GUI 停止/恢复、模型加载/导出下载、评估与推理页面真实文件交互、1920/1440/1366/1024px 浏览器布局。GPU CLI 训练已验证，GUI CUDA 操作未验证。
- 现有范围限制：没有独立系统设置、诊断或帮助操作 API；GUI 继续只暴露共享配置的一部分。没有为缺失能力增加静态占位项。

## 截图复核与第二轮修正

- 验收日期：2026-10-16。复核用户提供的 7 张界面截图，确认 Gradio Soft 主题将每个字段标签渲染为高亮绿底、尚未执行的 JSON/图表/文件控件占据大块空间、实验与模型列表过宽、没有数据集时训练页缺少直达入口；停止任务仍显示可操作的停止按钮，已结束任务的资源字段则停留在“读取中”。
- 将主题字段标签改为透明背景中性文字，收敛表格/导航阴影并调整中性色；列表改为关键浏览字段，完整实验/模型数据仍保留在详情中；表格启用换行并为列表列设置宽度。
- 数据集检查、训练配置、实验详情、模型详情/导出、评估结果、单图/批量推理结果采用明确空状态，操作产生结果后再展开真实组件；训练空状态增加前往数据集入口。
- 训练监控将停止按钮绑定到持久任务状态，已停止/完成任务禁用停止操作；恢复已结束任务时显示资源采样已结束，运行中的 GUI 任务继续保留原实时采样。
- 本轮验证：Ruff、Python 编译和 Gradio Blocks 构建通过；构建包含 293 个组件、41 个事件依赖、7 个导航入口；GUI 工作区、数据集工作流、实验管理和任务监控的相关测试 16 项通过。
- 浏览器交互与 1920/1440/1366/1024px 布局仍未验证。用户截图证明的是修正前状态；修正后需要在新启动页面进行人工复核。

### 截图复核与第三轮视觉修正

- 验收日期：2026-10-16。复核用户提供的 4 张窗口截图，确认数据集、实验详情、模型资产和评估的空状态均出现双竖线；工作区周围出现 `#e5e7eb` 的 Gradio 默认灰色背景，与 UATF 浅绿中性色不一致。
- 后续复核用户提供的 6 张截图并采样像素，确认重叠的两条线颜色完全相同（`#7b9a89`），来自空状态 CSS 在 Gradio 组件包装层重复应用，而不是 Markdown 引用线。前一轮 blockquote 覆盖没有命中，现已移除。
- 7 个空状态组件改用各自唯一的 `elem_id`，通过 ID 选择器只装饰单个组件根节点，避免共享 class 在嵌套包装层重复绘制边框和内边距。
- Gradio Soft 主题明确设置主/次画布、白色面板和 UATF 中性色阶，替换默认偏冷的灰阶；内容表面仍为白色，边框统一为 `#d9e1dc`。
- 新增主题 token 与空状态唯一 ID 回归断言。本次 GUI 定向测试 18 项通过；Ruff、Python 编译、`git diff --check` 和 Gradio Blocks 构建通过。页面仍包含 293 个组件、41 个事件依赖、7 个导航页面；7 个空状态 ID 唯一。
- 修正版服务已在 `http://127.0.0.1:7862` 监听。浏览器自动审批此前返回 404，本轮未进行浏览器截图复核；双线修复需要在该地址由用户目视确认。

### 侧栏分隔线调整

- UATF 品牌简介使用唯一 `elem_id` 和一条底部分隔线；版本信息使用唯一 `elem_id` 和一条顶部分隔线，7 项导航菜单位于两条线之间。
- 验收覆盖侧栏三个组件 ID 及两条边界样式；需在启动页面确认无重复线条且版本区与导航区分清晰。
