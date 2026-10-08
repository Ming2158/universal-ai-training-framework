# Phase 4: Dataset System

Phase 4 moves image input into reusable adapters under `uatf/data`. Existing training code receives ordinary PyTorch `DataLoader` batches, so adding a dataset format does not require changing the model or the training loop.

```text
ImageFolder / CSV / future dataset adapter
                  -> Dataset sample
                  -> DataLoader batch
                  -> training loop
```

## Supported formats

| Format | Adapter | Target |
| --- | --- | --- |
| One class directory per label | `ImageFolderDataset("dataset/train")` | Scalar `torch.long` class index |
| `image_path,label` CSV | `CsvImageClassificationDataset("labels.csv")` | Scalar `torch.long` class index |
| `image_path,cat,dog,bird` CSV | `CsvMultiLabelDataset("labels.csv")` | `torch.float32` multi-hot vector |

CSV paths are resolved relative to the CSV file's directory by default. Pass `image_root` to use another image directory. The folder adapter discovers class names in sorted order. CSV single-label classes are also sorted; multi-label positions follow the CSV label-column order.

All adapters convert images to RGB and return `(image, target)`. Without a supplied transform, images are converted to `[C, H, W]` float tensors in the range `[0, 1]`.

```python
from uatf.data import CsvMultiLabelDataset, build_image_transform

dataset = CsvMultiLabelDataset(
    "dataset/labels.csv",
    transform=build_image_transform(224),
)
image, labels = dataset[0]
```

## Multi-class and multi-label

In multi-class classification, each image belongs to exactly one class. The target is one integer, such as `2`, and a model commonly produces one score per class for `CrossEntropyLoss`.

In multi-label classification, each image can have any combination of labels. For CSV columns `cat,dog,bird`, values `1,0,1` become the float vector `[1, 0, 1]`. A multi-label model typically produces one independent score per label and uses a loss such as `BCEWithLogitsLoss`; this differs from treating the classes as mutually exclusive.

## Transforms

`build_image_transform(image_size, mean, std)` composes only `Resize`, `ToTensor`, and `Normalize`. The defaults use 224 by 224 RGB images and ImageNet channel statistics. Pass a different size or channel statistics for a model or dataset with different requirements. Random augmentation is intentionally not included in this phase.

## Splits and leakage

`create_data_loaders` splits one dataset into disjoint subsets using a seeded `torch.Generator`. Reusing the same data and seed reproduces the same split. Training batches are shuffled; validation and test batches keep their order.

Validation guides choices during development, such as selecting a checkpoint. Test data must stay outside the training loop and model-selection decisions; it estimates performance on unseen examples after those choices are complete. Feeding test examples into optimization or repeatedly tuning against test results leaks information and makes the final score overly optimistic.

The helper prevents overlap by sample index, and the provided folder/CSV adapters reject repeated image paths. For datasets with related examples (for example, several scans from one patient), a future grouped split must keep the whole patient in one partition; a random image split alone cannot prevent that form of leakage.

```python
train_loader, validation_loader, test_loader = create_data_loaders(
    dataset,
    batch_size=32,
    validation_fraction=0.1,
    test_fraction=0.1,
    seed=42,
)
```

Set a fraction to `0` when that partition is already managed elsewhere. Small datasets must still have enough samples to allocate every requested partition.

## Validation

Adapters report `DatasetValidationError` with a direct message for missing directories or images, unreadable images, an empty dataset, missing or extra CSV columns, malformed row lengths, blank paths/labels, duplicate paths, and non-binary multi-label values. Set `verify_images=False` to skip the eager image decode check for a large dataset; missing files are still checked, and unreadable files can then fail when fetched.

## Extending the system

A new dataset adapter, such as `ChestXrayDataset`, should implement PyTorch's `Dataset` contract and return `(image_tensor, target_tensor)` with the target format expected by its task. The DataLoader and training loop consume that contract, not a folder or CSV layout. The task-specific loss and metrics remain separate from the data adapter.

Run the offline dataset tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_datasets.py
```
