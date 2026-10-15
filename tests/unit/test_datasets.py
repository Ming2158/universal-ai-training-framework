from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from uatf.data import (
    CsvImageClassificationDataset,
    CsvMultiLabelDataset,
    DatasetValidationError,
    ImageFolderDataset,
    build_image_transform,
    create_data_loaders,
    split_dataset,
)


def save_image(path: Path, color: tuple[int, int, int] = (20, 80, 150)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def test_image_folder_dataset_and_transform_return_tensor_sample(tmp_path: Path) -> None:
    save_image(tmp_path / "cats" / "cat.png")
    save_image(tmp_path / "dogs" / "dog.png")

    dataset = ImageFolderDataset(
        tmp_path,
        transform=build_image_transform((16, 10)),
    )

    assert len(dataset) == 2
    assert dataset.classes == ["cats", "dogs"]
    image, label = dataset[0]
    assert image.shape == (3, 16, 10)
    assert image.dtype == torch.float32
    assert label.shape == torch.Size([])
    assert label.dtype == torch.long


def test_single_label_csv_dataset_and_batch_shapes(tmp_path: Path) -> None:
    save_image(tmp_path / "one.png")
    save_image(tmp_path / "two.png")
    csv_path = tmp_path / "labels.csv"
    write_csv(csv_path, ["image_path", "label"], [["one.png", "cat"], ["two.png", "dog"]])

    dataset = CsvImageClassificationDataset(
        csv_path,
        transform=build_image_transform(14),
    )
    train_loader, validation_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=2,
        validation_fraction=0,
        seed=13,
    )
    images, labels = next(iter(train_loader))

    assert len(dataset) == 2
    assert images.shape == (2, 3, 14, 14)
    assert labels.shape == (2,)
    assert labels.dtype == torch.long
    assert validation_loader is None
    assert test_loader is None


def test_multilabel_csv_uses_column_order_and_float_vector(tmp_path: Path) -> None:
    save_image(tmp_path / "bird_cat.png")
    save_image(tmp_path / "dog.png")
    csv_path = tmp_path / "labels.csv"
    write_csv(
        csv_path,
        ["image_path", "cat", "dog", "bird"],
        [["bird_cat.png", "1", "0", "1"], ["dog.png", "0", "1", "0"]],
    )

    dataset = CsvMultiLabelDataset(csv_path, transform=build_image_transform(8))
    image, target = dataset[0]
    loader, _, _ = create_data_loaders(
        dataset,
        batch_size=2,
        validation_fraction=0,
    )
    batch_images, batch_targets = next(iter(loader))

    assert len(dataset) == 2
    assert dataset.classes == ["cat", "dog", "bird"]
    assert image.shape == (3, 8, 8)
    assert target.shape == (3,)
    assert target.dtype == torch.float32
    assert target.tolist() == [1.0, 0.0, 1.0]
    assert batch_images.shape == (2, 3, 8, 8)
    assert batch_targets.shape == (2, 3)


def test_split_is_reproducible_and_has_no_overlap(tmp_path: Path) -> None:
    rows = []
    for index in range(10):
        filename = f"image_{index}.png"
        save_image(tmp_path / filename)
        rows.append([filename, "cat" if index % 2 else "dog"])
    csv_path = tmp_path / "labels.csv"
    write_csv(csv_path, ["image_path", "label"], rows)
    dataset = CsvImageClassificationDataset(csv_path)

    first = split_dataset(dataset, validation_fraction=0.2, test_fraction=0.2, seed=77)
    second = split_dataset(dataset, validation_fraction=0.2, test_fraction=0.2, seed=77)
    first_indices = [set(subset.indices) for subset in first if subset is not None]
    second_indices = [set(subset.indices) for subset in second if subset is not None]

    assert first_indices == second_indices
    assert len(first_indices) == 3
    assert first_indices[0].isdisjoint(first_indices[1])
    assert first_indices[0].isdisjoint(first_indices[2])
    assert first_indices[1].isdisjoint(first_indices[2])
    assert set.union(*first_indices) == set(range(len(dataset)))


def test_training_augmentation_is_randomized_only_in_training_views(tmp_path: Path) -> None:
    image_path = tmp_path / "cats" / "asymmetric.png"
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (12, 8), "black")
    for x in range(6, 12):
        for y in range(8):
            image.putpixel((x, y), (255, 255, 255))
    image.save(image_path)
    for index in range(3):
        save_image(tmp_path / "dogs" / f"dog-{index}.png")

    evaluation_transform = build_image_transform(8, mean=(0, 0, 0), std=(1, 1, 1))
    training_transform = build_image_transform(
        8,
        mean=(0, 0, 0),
        std=(1, 1, 1),
        augmentation={"enabled": True, "horizontal_flip_probability": 1.0},
        training=True,
    )
    dataset = ImageFolderDataset(tmp_path, transform=evaluation_transform)
    train_loader, validation_loader, _ = create_data_loaders(
        dataset,
        batch_size=1,
        validation_fraction=0.5,
        seed=3,
        train_transform=training_transform,
        evaluation_transform=evaluation_transform,
    )

    training_view = train_loader.dataset.dataset
    validation_view = validation_loader.dataset.dataset
    assert isinstance(training_view.transform.transforms[0], transforms.RandomHorizontalFlip)
    assert not any(
        isinstance(operation, transforms.RandomHorizontalFlip)
        for operation in validation_view.transform.transforms
    )
    validation_image = validation_view[0][0]
    assert torch.equal(validation_image, validation_view[0][0])
    assert not torch.equal(training_view[0][0], evaluation_transform(image))


def test_image_transform_supports_rotation_and_color_jitter_only_for_training() -> None:
    augmentation = {
        "enabled": True,
        "rotation_degrees": 12,
        "brightness": 0.1,
        "contrast": 0.2,
        "saturation": 0.15,
    }

    train_operations = build_image_transform(augmentation=augmentation, training=True).transforms
    evaluation_operations = build_image_transform(augmentation=augmentation).transforms

    assert any(isinstance(operation, transforms.RandomRotation) for operation in train_operations)
    assert any(isinstance(operation, transforms.ColorJitter) for operation in train_operations)
    assert all(
        not isinstance(operation, (transforms.RandomRotation, transforms.ColorJitter))
        for operation in evaluation_operations
    )


@pytest.mark.parametrize(
    ("header", "rows", "message"),
    [
        (["image", "label"], [["img.png", "cat"]], "missing required column: image_path"),
        (["image_path", "label"], [["", "cat"]], "empty image_path"),
        (["image_path", "label"], [["missing.png", "cat"]], "does not exist"),
        (["image_path", "label"], [["img.png", ""]], "empty label"),
        (["image_path", "label"], [["img.png", "cat"], ["img.png", "dog"]], "repeats image_path"),
        (["image_path", "label"], [["img.png", "cat", "extra"]], "expected 2"),
    ],
)
def test_single_label_csv_rejects_invalid_rows(
    tmp_path: Path,
    header: list[str],
    rows: list[list[str]],
    message: str,
) -> None:
    save_image(tmp_path / "img.png")
    csv_path = tmp_path / "labels.csv"
    write_csv(csv_path, header, rows)

    with pytest.raises(DatasetValidationError, match=message):
        CsvImageClassificationDataset(csv_path)


def test_dataset_validation_rejects_damaged_image(tmp_path: Path) -> None:
    (tmp_path / "broken.png").write_bytes(b"not an image")
    csv_path = tmp_path / "labels.csv"
    write_csv(csv_path, ["image_path", "label"], [["broken.png", "cat"]])

    with pytest.raises(DatasetValidationError, match="damaged or unreadable"):
        CsvImageClassificationDataset(csv_path)


def test_dataset_validation_rejects_empty_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    write_csv(csv_path, ["image_path", "label"], [])

    with pytest.raises(DatasetValidationError, match="no image rows"):
        CsvImageClassificationDataset(csv_path)


def test_multilabel_dataset_rejects_non_binary_label(tmp_path: Path) -> None:
    save_image(tmp_path / "img.png")
    csv_path = tmp_path / "labels.csv"
    write_csv(csv_path, ["image_path", "cat", "dog"], [["img.png", "1", "2"]])

    with pytest.raises(DatasetValidationError, match="expected 0 or 1"):
        CsvMultiLabelDataset(csv_path)


def test_folder_dataset_rejects_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "cats").mkdir()

    with pytest.raises(DatasetValidationError, match="no images"):
        ImageFolderDataset(tmp_path)
