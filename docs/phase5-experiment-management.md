# Phase 5：实验、模型与产物管理

## 存储结构

实验继续保存在配置指定的输出目录，不使用数据库服务：

```text
outputs/
  experiment_001/
    experiment.json
    source_config.yaml
    config.yaml
    metrics.json
    logs/train.log
    best_model.pth
    last_model.pth
    plots/                  # 按配置生成
  .uatf/
    tasks/                  # 任务生命周期记录
    datasets/<split-id>.json
    models/<asset-id>.json
    exports/<asset-id>.weights.pth
    exports/<asset-id>.weights.pth.json
```

`experiment.json` 使用 UTF-8 JSON，包含稳定实验 ID、状态与时间、错误原因、数据和划分 ID、模型与有效配置摘要、代码/环境信息及产物引用。样本清单只记录相对样本标识、标签和 train/validation 划分，不复制图像，也不计算图像内容哈希；清单按划分摘要寻址，相同划分可复用。

`.uatf/models/` 中的 JSON 是模型资产索引，权重仍保存在来源实验目录。`best_model.pth` 和 `last_model.pth` 是包含优化器状态的完整训练检查点；导出操作生成只有 `model_state_dict` 的推理权重，并附带任务、架构和来源实验 JSON。

## 旧实验读取与迁移

迁移采用只读兼容，不移动、改写或删除旧实验。没有 `experiment.json` 的目录会从 `config.yaml`、`metrics.json` 和任务记录推导列表与详情；有最佳检查点和配置的旧目录也会作为只读模型资产显示，可加载或导出。旧实验不会自动补写数据划分清单，缺少数据或指标标识时，比较界面会提示不可比。

新实验从创建时开始写入完整记录。产物路径始终按输出根目录解析，模型资产删除和实验清理本阶段不提供；导出只新增文件，不触碰来源实验。

## Python API

`uatf.api` 提供 `list_experiments`、`get_experiment`、`compare_experiments`、`list_model_assets`、`load_model_asset` 和 `export_model_asset`。比较结果只在任务类型、数据集与划分、主要指标定义及阈值一致时标记为可直接比较。
