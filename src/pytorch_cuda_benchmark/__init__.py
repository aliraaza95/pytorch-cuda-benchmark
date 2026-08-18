"""CPU and CUDA training benchmark utilities."""

from .benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    create_benchmark_plan,
    run_benchmark,
)
from .config import BenchmarkConfig, load_config
from .data import create_cifar10_training_loader
from .environment import (
    EnvironmentMetadata,
    GpuMetadata,
    collect_environment_metadata,
)
from .model import create_cifar10_resnet18
from .reproducibility import (
    create_data_generator,
    seed_data_worker,
    set_random_seed,
)
from .results import SavedBenchmarkArtifacts, save_benchmark_artifacts
from .training import TrainingResult, train_and_measure

__all__ = [
    "BenchmarkCase",
    "BenchmarkConfig",
    "BenchmarkResult",
    "EnvironmentMetadata",
    "GpuMetadata",
    "SavedBenchmarkArtifacts",
    "TrainingResult",
    "collect_environment_metadata",
    "create_benchmark_plan",
    "create_cifar10_resnet18",
    "create_cifar10_training_loader",
    "create_data_generator",
    "load_config",
    "run_benchmark",
    "save_benchmark_artifacts",
    "seed_data_worker",
    "set_random_seed",
    "train_and_measure",
]
