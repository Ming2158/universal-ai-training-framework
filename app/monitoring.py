"""Optional hardware resource sampling for the Gradio interface."""

from __future__ import annotations

import importlib
import math
import shutil
import subprocess
import time
from typing import Any

import torch


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class HardwareResourceSampler:
    """Read best-effort system, training-process, and GPU statistics."""

    def __init__(self) -> None:
        try:
            self._psutil = importlib.import_module("psutil")
            self._psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None
        self._process: Any = None
        self._nvml: Any = None
        self._nvml_handle: Any = None
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._last_smi_sample = 0.0
        self._smi_values: dict[str, float | None] = {}
        try:
            self._nvml = importlib.import_module("pynvml")
            self._nvml.nvmlInit()
            self._nvml_handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._nvml_handle = None

    def attach_process(self, process_id: int) -> None:
        if self._psutil is None:
            return
        try:
            self._process = self._psutil.Process(process_id)
            self._process.cpu_percent(interval=None)
        except Exception:
            self._process = None

    def close(self) -> None:
        if self._nvml is None:
            return
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
        self._nvml = None
        self._nvml_handle = None

    def _gpu_utilization(self) -> dict[str, float | None]:
        values: dict[str, float | None] = {
            "utilization_percent": None,
            "temperature_c": None,
            "power_watts": None,
        }
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                values["utilization_percent"] = _finite_number(utilization.gpu)
            except Exception:
                pass
            try:
                values["temperature_c"] = _finite_number(
                    self._nvml.nvmlDeviceGetTemperature(
                        self._nvml_handle, self._nvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass
            try:
                values["power_watts"] = _finite_number(
                    self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000
                )
            except Exception:
                pass
            return values

        now = time.monotonic()
        if self._nvidia_smi and now - self._last_smi_sample >= 5:
            self._last_smi_sample = now
            try:
                result = subprocess.run(
                    [
                        self._nvidia_smi,
                        "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=1.5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                fields = result.stdout.splitlines()[0].split(",")
                if len(fields) == 3:
                    self._smi_values = {
                        "utilization_percent": _finite_number(fields[0]),
                        "temperature_c": _finite_number(fields[1]),
                        "power_watts": _finite_number(fields[2]),
                    }
            except Exception:
                self._smi_values = {}
        values.update(self._smi_values)
        return values

    def snapshot(self) -> dict[str, Any]:
        cpu: dict[str, float | None] = {
            "system_percent": None,
            "training_process_percent": None,
        }
        memory: dict[str, int | None] = {
            "used_bytes": None,
            "available_bytes": None,
            "total_bytes": None,
            "training_process_rss_bytes": None,
        }
        if self._psutil is not None:
            try:
                cpu["system_percent"] = _finite_number(self._psutil.cpu_percent(interval=None))
                virtual_memory = self._psutil.virtual_memory()
                memory.update(
                    used_bytes=int(virtual_memory.used),
                    available_bytes=int(virtual_memory.available),
                    total_bytes=int(virtual_memory.total),
                )
            except Exception:
                pass
            if self._process is not None:
                try:
                    cpu["training_process_percent"] = _finite_number(
                        self._process.cpu_percent(interval=None)
                    )
                    memory["training_process_rss_bytes"] = int(
                        self._process.memory_info().rss
                    )
                except Exception:
                    self._process = None

        gpu: dict[str, Any] = {
            "name": None,
            "utilization_percent": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
            "torch_allocated_bytes": None,
            "temperature_c": None,
            "power_watts": None,
        }
        try:
            if torch.cuda.is_available():
                device_index = 0
                gpu["name"] = torch.cuda.get_device_name(device_index)
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                gpu["memory_total_bytes"] = int(total_bytes)
                gpu["memory_used_bytes"] = int(total_bytes - free_bytes)
                gpu["torch_allocated_bytes"] = int(
                    torch.cuda.memory_allocated(device_index)
                )
                gpu.update(self._gpu_utilization())
        except Exception:
            pass
        return {"cpu": cpu, "memory": memory, "gpu": gpu}


def _percent(value: float | None) -> str:
    return "不可用" if value is None else f"{value:.0f}%"


def _bytes(value: int | None) -> str:
    if value is None:
        return "不可用"
    gib = value / (1024**3)
    return f"{gib:.2f} GiB"


def format_resource_snapshot(snapshot: dict[str, Any]) -> str:
    """Format resource fields without substituting values for unavailable data."""
    cpu = snapshot["cpu"]
    memory = snapshot["memory"]
    gpu = snapshot["gpu"]
    gpu_name = gpu["name"] or "不可用"
    gpu_memory = (
        f"{_bytes(gpu['memory_used_bytes'])} / {_bytes(gpu['memory_total_bytes'])}"
        if gpu["memory_total_bytes"] is not None
        else "不可用"
    )
    torch_memory = _bytes(gpu["torch_allocated_bytes"])
    temperature = (
        "不可用"
        if gpu["temperature_c"] is None
        else f"{gpu['temperature_c']:.0f} °C"
    )
    power = (
        "不可用"
        if gpu["power_watts"] is None
        else f"{gpu['power_watts']:.1f} W"
    )
    system_memory = (
        f"已用 {_bytes(memory['used_bytes'])}，可用 {_bytes(memory['available_bytes'])} / "
        f"{_bytes(memory['total_bytes'])}"
        if memory["total_bytes"] is not None
        else "不可用"
    )
    return "\n".join(
        (
            f"GPU: {gpu_name} | 利用率 {_percent(gpu['utilization_percent'])} | "
            f"显存 {gpu_memory} | PyTorch 分配 {torch_memory} | "
            f"温度 {temperature} | 功耗 {power}",
            f"CPU: 系统 {_percent(cpu['system_percent'])} | "
            f"训练进程 {_percent(cpu['training_process_percent'])}",
            f"系统内存: {system_memory} | "
            f"训练进程内存 {_bytes(memory['training_process_rss_bytes'])}",
        )
    )
