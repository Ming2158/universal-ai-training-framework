# Phase 7: Evaluation System

Phase 7 keeps evaluation outside `Trainer`. The trainer remains responsible for optimization, validation loss, epoch history, and best/last checkpoints. `uatf.evaluation` runs a separate whole-dataset prediction pass after training, computes task-appropriate metrics, and writes reports and plots.

## Use after training

After `Trainer.fit()` returns, pass its history and validation loader to `evaluate_training_run()`. The helper loads `best_model.pth`, evaluates that checkpoint, and writes the report beside the checkpoints:

```python
from uatf.evaluation import evaluate_training_run

history = trainer.fit(train_loader, validation_loader, epochs=10)
result = evaluate_training_run(
    trainer.model,
    validation_loader,
    "outputs/experiment-1",
    task="classification",  # or "multi_label"
    class_names=validation_loader.dataset.classes,
    loss_function=trainer.loss_function,
    device=trainer.device,
    history=history,
)
```

For multi-label datasets, `classes` is the ordered label-column list. If not supplied, the evaluator tries to read the class names from the DataLoader dataset. Training metric curves use the metric keys already present in `history`; configure `Trainer(metrics=...)` for metrics such as epoch accuracy. The final classification and multi-label results always come from the independent evaluator, not from averages of per-batch F1 or AUC.

The existing MNIST teaching script calls this helper after training. Its evaluation output is saved under `outputs/phase3-mnist/`.

## Output files

```text
outputs/experiment-1/
  best_model.pth
  last_model.pth
  metrics.json
  predictions.csv
  plots/
    loss_curve.png
    metric_curve.png
    confusion_matrix.png       # single-label classification
    roc_curve.png              # multi-label classification
    roc_00_<class>.png         # one plot per label
```

`metrics.json` contains the task, class names, evaluation loss (when provided), aggregate metrics, and per-class metrics. `predictions.csv` has one row per validation example in DataLoader iteration order, including class probabilities. For a multi-label task it contains one target, prediction, and probability column per label. Dataset adapters currently yield image tensors and labels, not file paths, so the CSV uses `sample_index` rather than an image path.

If training history is omitted, loss and metric plot files are still created with a note that no series were supplied. Pass the list returned by `fit()` to get actual curves.

## The two task types

**Classification (single-label, multi-class)** assigns exactly one class to each example. The model emits one score per class, softmax converts scores to probabilities that sum to one, and argmax selects one prediction. A confusion matrix has one row and column per class.

**Multi-label classification** assigns an independent yes/no decision to every label. One image can have no labels, one label, or several labels at once. The model emits one score per label, sigmoid converts each score independently, and a threshold such as `0.5` decides each label. ROC-AUC is calculated separately for each label because each label is its own binary prediction problem.

These tasks cannot share the same decision rule. Softmax forces class probabilities to compete and sum to one, which is wrong when two findings can coexist. Sigmoid does not force that competition, which is appropriate for independent labels but does not enforce the “exactly one class” rule of ordinary classification. The evaluator uses the `task` setting to select the correct interpretation.

## Metric meanings

- **Accuracy**: fraction of examples whose predicted class is correct. For single-label classification, one wrong class makes the whole example wrong.
- **Precision**: among predictions of a class or label, the fraction that were correct. Low precision means more false positives.
- **Recall**: among true examples of a class or label, the fraction the model found. Low recall means more false negatives.
- **F1**: harmonic mean of precision and recall. It is high only when both are reasonably high.
- **Confusion matrix**: counts true classes by predicted classes. Rows are the true class; columns are the predicted class. The diagonal is correct and off-diagonal cells show which classes are confused.
- **ROC curve**: traces true positive rate against false positive rate while the score threshold changes.
- **AUC / ROC-AUC**: area under that ROC curve. `1.0` ranks positives above negatives perfectly; `0.5` is about random ranking. AUC measures ranking across thresholds, while precision, recall, and F1 describe decisions at one threshold.
- **Macro average**: calculate a metric for each class, then give every class equal weight. This makes small classes visible in the summary.
- **Micro average**: add the true-positive, false-positive, and false-negative counts across classes first, then calculate the metric. Classes with more examples have more influence.
- **Class imbalance**: classes have very different numbers of examples. A model can score well on the frequent class while missing most rare examples.

For single-label classification, the headline `precision`, `recall`, and `f1` fields are macro averages. The report also includes explicit `macro_*`, `micro_*`, and per-class values. In this setting micro precision, recall, and F1 equal accuracy. For multi-label classification, the same three headline fields are macro averages across labels; the report additionally provides `macro_auc`, `micro_auc`, and each label's AUC.

In medical-image work, accuracy alone can hide a model that predicts “negative” for nearly everyone. If 95% of images are negative, that behavior can reach 95% accuracy while finding none of the positive cases. Recall exposes missed positives; precision shows the false-alarm tradeoff; per-class results and the confusion matrix show whether a rare finding is being ignored. ROC-AUC adds a threshold-independent view of ranking, but it also does not replace threshold-specific metrics or careful validation. These are model-evaluation outputs, not clinical diagnoses.

## Reliability behavior

- Precision, recall, and F1 use zero when their denominator is zero, so an unpredicted or unsupported class does not crash evaluation.
- Classification metrics retain the configured class count and names, including classes absent from the validation split.
- A per-label AUC is `null` when that split contains only positives or only negatives for the label. Macro AUC averages the remaining defined labels and reports their count. If no label has both target values, macro AUC is `null`.
- NaN and infinite scores or targets are rejected with a clear error. `metrics.json` is written with non-finite JSON numbers disallowed.
- AUC is calculated from the complete score arrays with tied scores grouped together. It is not averaged across batches.

## Test

Run the offline metrics and artifact checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_evaluation.py
```

The tests use tiny in-memory examples and check accuracy, precision, recall, F1, the confusion matrix, per-class and macro AUC, an absent label, JSON/CSV output, and plot files.
