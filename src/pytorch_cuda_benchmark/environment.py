"""Collect hardware and software metadata for benchmark runs."""

import platform
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from typing import Protocol, cast

import psutil
import torch


class _CudaDeviceProperties(Protocol):
    @property
    def total_memory(self) -> int:
        ...


@dataclass(frozen=True)
class GpuMetadata:
    """Properties of a CUDA device used by the benchmark."""

    index: int
    name: str
    total_memory_bytes: int
    compute_capability: tuple[int, int]


@dataclass(frozen=True)
class EnvironmentMetadata:
    """Hardware and software details for a benchmark environment."""

    operating_system: str
    architecture: str
    python_version: str
    processor_identifier: str
    physical_cpu_cores: int | None
    logical_cpu_cores: int | None
    total_memory_bytes: int
    pytorch_version: str
    torchvision_version: str
    torch_thread_count: int
    torch_interop_thread_count: int
    cuda_available: bool
    cuda_build_version: str | None
    cudnn_version: int | None
    gpus: tuple[GpuMetadata, ...]


def _get_processor_identifier() -> str:
    processor = platform.processor().strip()
    architecture = platform.machine().strip()

    return processor or architecture or "unknown"


def _get_cpu_count(*, logical: bool) -> int | None:
    value: object = psutil.cpu_count(logical=logical)

    return value if isinstance(value, int) else None


def _get_total_memory() -> int:
    value: object = getattr(psutil.virtual_memory(), "total", None)

    if not isinstance(value, int):
        raise RuntimeError("Unable to determine total system memory.")

    return value


def _get_cuda_build_version() -> str | None:
    value: object = getattr(torch.version, "cuda", None)

    return value if isinstance(value, str) else None


def _get_cudnn_version(cuda_available: bool) -> int | None:
    if not cuda_available:
        return None

    version_reader = cast(
        Callable[[], object],
        getattr(torch.backends.cudnn, "version"),
    )
    value = version_reader()

    return value if isinstance(value, int) else None


def _collect_gpu_metadata(cuda_available: bool) -> tuple[GpuMetadata, ...]:
    if not cuda_available:
        return ()

    properties_reader = cast(
        Callable[[int], _CudaDeviceProperties],
        getattr(torch.cuda, "get_device_properties"),
    )

    gpus: list[GpuMetadata] = []

    for index in range(torch.cuda.device_count()):
        properties = properties_reader(index)

        gpus.append(
            GpuMetadata(
                index=index,
                name=torch.cuda.get_device_name(index),
                total_memory_bytes=properties.total_memory,
                compute_capability=torch.cuda.get_device_capability(index),
            )
        )

    return tuple(gpus)


def collect_environment_metadata() -> EnvironmentMetadata:
    """Return hardware and software metadata for the current environment."""

    cuda_available = torch.cuda.is_available()

    return EnvironmentMetadata(
        operating_system=platform.platform(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        processor_identifier=_get_processor_identifier(),
        physical_cpu_cores=_get_cpu_count(logical=False),
        logical_cpu_cores=_get_cpu_count(logical=True),
        total_memory_bytes=_get_total_memory(),
        pytorch_version=version("torch"),
        torchvision_version=version("torchvision"),
        torch_thread_count=torch.get_num_threads(),
        torch_interop_thread_count=torch.get_num_interop_threads(),
        cuda_available=cuda_available,
        cuda_build_version=_get_cuda_build_version(),
        cudnn_version=_get_cudnn_version(cuda_available),
        gpus=_collect_gpu_metadata(cuda_available),
    )
