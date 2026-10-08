# Installation

UATF requires Python 3.10 or newer. PyTorch wheels must match the operating system and intended CPU/CUDA runtime. The commands below are for Windows PowerShell and the CUDA 12.8 build used on the development GPU.

## Create an Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

## Install PyTorch and UATF

For an NVIDIA GPU with a compatible driver, install the CUDA wheel first, then install UATF and its development tools:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

For CPU-only use, install the official CPU PyTorch/torchvision wheels for the platform, then run the UATF editable install command. The project declares its runtime packages in `pyproject.toml`; the `dev` extra adds pytest and Ruff.

## Verify

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe scripts/gpu_smoke_test.py
```

The GPU smoke test exits with a failure message when CUDA is unavailable. CPU users can skip it and choose `--device cpu` for training. See [Troubleshooting](troubleshooting.md) for stale virtual environments, CUDA errors, and data issues.
