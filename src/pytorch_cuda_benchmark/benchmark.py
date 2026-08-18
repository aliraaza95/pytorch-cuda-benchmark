"""Benchmark planning and execution utilities."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from .config import BenchmarkConfig
from .data import BenchmarkDevice, create_cifar10_training_loader
from .model import create_cifar10_resnet18
from .reproducibility import set_random_seed
from .training import TrainingResult, train_and_measure


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Configuration for one benchmark run."""

    device: BenchmarkDevice
    batch_size: int
    repetition: int
    seed: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measurements produced by one benchmark run."""

    device: BenchmarkDevice
    batch_size: int
    repetition: int
    seed: int
    mixed_precision: bool
    epochs: int
    processed_batches: int
    processed_samples: int
    mean_loss: float
    elapsed_seconds: float
    samples_per_second: float


def create_benchmark_plan(
    config: BenchmarkConfig,
) -> tuple[BenchmarkCase, ...]:
    """Create the device, batch-size, and repetition matrix."""

    cases: list[BenchmarkCase] = []

    for configured_device in config.devices:
        device = _resolve_device(configured_device)

        for batch_size in config.batch_sizes:
            for repetition_index in range(config.repetitions):
                cases.append(
                    BenchmarkCase(
                        device=device,
                        batch_size=batch_size,
                        repetition=repetition_index + 1,
                        seed=config.seed + repetition_index,
                    )
                )

    return tuple(cases)


def run_benchmark(
    config: BenchmarkConfig,
    *,
    data_directory: str | Path = "data",
    progress_callback: (
        Callable[[int, int, BenchmarkCase], None] | None
    ) = None,
) -> tuple[BenchmarkResult, ...]:
    """Execute every run in the configured benchmark matrix."""

    _validate_hardware(config)

    plan = create_benchmark_plan(config)
    total_runs = len(plan)
    results: list[BenchmarkResult] = []

    for run_number, case in enumerate(plan, start=1):
        if progress_callback is not None:
            progress_callback(run_number, total_runs, case)

        set_random_seed(case.seed)

        model = create_cifar10_resnet18()

        data_loader = cast(
            DataLoader[tuple[torch.Tensor, torch.Tensor]],
            create_cifar10_training_loader(
                data_directory,
                batch_size=case.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                device=case.device,
                seed=case.seed,
            ),
        )

        training_result = train_and_measure(
            model,
            data_loader,
            device=case.device,
            epochs=config.epochs,
            warmup_batches=config.warmup_batches,
            mixed_precision=config.mixed_precision,
        )

        results.append(
            _create_benchmark_result(
                case=case,
                training_result=training_result,
            )
        )

    return tuple(results)


def _resolve_device(device: str) -> BenchmarkDevice:
    """Convert a configured device string to a supported device type."""

    if device == "cpu":
        return "cpu"

    if device == "cuda":
        return "cuda"

    raise ValueError(f"Unsupported benchmark device: {device!r}.")


def _create_benchmark_result(
    *,
    case: BenchmarkCase,
    training_result: TrainingResult,
) -> BenchmarkResult:
    """Combine run configuration with its training measurements."""

    return BenchmarkResult(
        device=case.device,
        batch_size=case.batch_size,
        repetition=case.repetition,
        seed=case.seed,
        mixed_precision=training_result.mixed_precision,
        epochs=training_result.epochs,
        processed_batches=training_result.processed_batches,
        processed_samples=training_result.processed_samples,
        mean_loss=training_result.mean_loss,
        elapsed_seconds=training_result.elapsed_seconds,
        samples_per_second=training_result.samples_per_second,
    )


def _validate_hardware(config: BenchmarkConfig) -> None:
    """Confirm that every configured device is available."""

    if "cuda" in config.devices and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is configured but unavailable. Install a CUDA-enabled "
            "PyTorch build or remove 'cuda' from the benchmark configuration."
        )
