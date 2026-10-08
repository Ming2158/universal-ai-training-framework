"""Check that this Python environment can run a small operation on an NVIDIA GPU."""

import sys

import torch
import torchvision


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Virtual environment: {'yes' if sys.prefix != sys.base_prefix else 'no'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda or 'not included'}")

    if not torch.cuda.is_available():
        print("CUDA available: no")
        print("GPU smoke test failed. Check that the CUDA-enabled PyTorch build is installed.")
        return 1

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print("CUDA available: yes")
    print(f"GPU: {properties.name}")
    print(f"GPU compute capability: {properties.major}.{properties.minor}")

    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    result = left @ right
    torch.cuda.synchronize(device)

    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    if result.device != device or not torch.equal(result.cpu(), expected):
        print("GPU tensor calculation: failed")
        print(f"Result: {result.cpu().tolist()}")
        return 1

    print("GPU tensor calculation: passed")
    print(f"Result: {result.cpu().tolist()}")
    print("GPU smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
