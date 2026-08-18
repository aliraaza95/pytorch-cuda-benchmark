# pyright: reportMissingTypeStubs=false

"""Create the CIFAR-10 data pipeline used by benchmark runs."""

from pathlib import Path
from typing import Literal, cast

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from .reproducibility import create_data_generator, seed_data_worker

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STANDARD_DEVIATION = (0.2470, 0.2435, 0.2616)

Cifar10Sample = tuple[torch.Tensor, int]
BenchmarkDevice = Literal["cpu", "cuda"]


def _create_training_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=_CIFAR10_MEAN,
                std=_CIFAR10_STANDARD_DEVIATION,
            ),
        ]
    )


def create_cifar10_training_loader(
    data_directory: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: BenchmarkDevice,
    seed: int,
    download: bool = True,
) -> DataLoader[Cifar10Sample]:
    """Create a reproducible CIFAR-10 training DataLoader."""

    if batch_size <= 0:
        raise ValueError("'batch_size' must be a positive integer.")

    if num_workers < 0:
        raise ValueError(
            "'num_workers' must be greater than or equal to zero.")

    dataset = cast(
        Dataset[Cifar10Sample],
        CIFAR10(
            root=Path(data_directory),
            train=True,
            transform=_create_training_transform(),
            download=download,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device == "cuda",
        worker_init_fn=seed_data_worker,
        generator=create_data_generator(seed),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
