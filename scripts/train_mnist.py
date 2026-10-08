"""Train a small CNN on MNIST to demonstrate the complete PyTorch loop."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """A small CNN that maps 28x28 grayscale images to ten digit scores."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.hidden = nn.Linear(32 * 7 * 7, 64)
        self.classifier = nn.Linear(64, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.pool(torch.relu(self.conv1(images)))
        images = self.pool(torch.relu(self.conv2(images)))
        images = torch.flatten(images, start_dim=1)
        images = torch.relu(self.hidden(images))
        return self.classifier(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用公开 MNIST 手写数字数据集，演示从数据加载到最佳模型保存的完整训练流程。",
        add_help=False,
        usage="%(prog)s [选项]",
    )
    options = parser.add_argument_group("运行参数")
    options.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    options.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="完整遍历训练集的次数，默认 5",
    )
    options.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="每批处理的图片数量，默认 128",
    )
    options.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="每次参数更新的步幅，默认 0.001",
    )
    options.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于固定数据拆分，默认 42",
    )
    options.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备：auto 自动选择，或指定 cpu / cuda，默认 auto",
    )
    options.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/mnist"),
        help="MNIST 数据保存目录，默认 datasets/mnist",
    )
    options.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase3-mnist"),
        help="模型和训练输出目录，默认 outputs/phase3-mnist",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "你指定了 CUDA，但 PyTorch 当前无法使用 NVIDIA GPU。"
                "请检查 CUDA 版 PyTorch，或改用 --device cpu。"
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(
    data_dir: Path, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    try:
        full_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    except RuntimeError as error:
        if "Error downloading" not in str(error):
            raise
        raise RuntimeError(
            f"无法下载 MNIST 数据到 {data_dir}。请检查网络连接后重新运行。"
        ) from error

    validation_size = len(full_dataset) // 10
    training_size = len(full_dataset) - validation_size
    split_generator = torch.Generator().manual_seed(seed)
    training_dataset, validation_dataset = random_split(
        full_dataset,
        [training_size, validation_size],
        generator=split_generator,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return training_loader, validation_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    example_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        batch_count = images.size(0)
        total_loss += loss.item() * batch_count
        example_count += batch_count

    return total_loss / example_count


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    example_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = loss_function(predictions, labels)

            batch_count = images.size(0)
            total_loss += loss.item() * batch_count
            correct_predictions += (predictions.argmax(dim=1) == labels).sum().item()
            example_count += batch_count

    return total_loss / example_count, correct_predictions / example_count


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("训练轮数、批次大小和学习率都必须大于 0。")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    training_loader, validation_loader = make_loaders(
        args.data_dir, args.batch_size, args.seed
    )
    model = SimpleCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\n========== UATF 第一个图像分类训练 ==========")
    print(f"【运行设备】{device}")
    if device.type == "cuda":
        print(f"【显卡】{torch.cuda.get_device_name(device)}")
    print("【数据集】MNIST 手写数字")
    print(
        f"【数据拆分】训练集 {len(training_loader.dataset):,} 张，"
        f"验证集 {len(validation_loader.dataset):,} 张"
    )
    print(
        "【训练流程】读取批次 -> 前向预测 -> 计算损失 -> 反向传播 -> "
        "更新参数 -> 验证"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pth"
    best_validation_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        training_loss = train_one_epoch(
            model, training_loader, loss_function, optimizer, device
        )
        validation_loss, validation_accuracy = validate(
            model, validation_loader, loss_function, device
        )

        print(f"\n---------- 第 {epoch}/{args.epochs} 轮训练 ----------")
        print(f"训练损失：{training_loss:.4f}")
        print(f"验证损失：{validation_loss:.4f}")
        print(f"验证准确率：{validation_accuracy * 100:.2f}%")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": [str(digit) for digit in range(10)],
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                },
                checkpoint_path,
            )
            print(f"  -> 验证准确率刷新最佳，已保存模型：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loaded_validation_loss, loaded_validation_accuracy = validate(
        model, validation_loader, loss_function, device
    )

    print("\n========== 最佳模型检查 ==========")
    print("检查点重载：成功")
    print(f"最佳模型来自第 {checkpoint['epoch']}/{args.epochs} 轮")
    print(f"重载后验证损失：{loaded_validation_loss:.4f}")
    print(f"重载后验证准确率：{loaded_validation_accuracy * 100:.2f}%")
    print(f"模型文件位置：{checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
