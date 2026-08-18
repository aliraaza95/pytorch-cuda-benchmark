"""Load and validate benchmark configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

SUPPORTED_DEVICES: frozenset[str] = frozenset({"cpu", "cuda"})


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated settings shared by CPU and CUDA benchmark runs."""

    seed: int
    devices: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    epochs: int
    repetitions: int
    warmup_batches: int
    num_workers: int
    pin_memory: bool
    mixed_precision: bool


def _read_integer(
    data: dict[str, object],
    key: str,
    *,
    minimum: int,
) -> int:
    value = data.get(key)

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"'{key}' must be an integer greater than or equal to {minimum}."
        )

    return value


def _read_boolean(data: dict[str, object], key: str) -> bool:
    value = data.get(key)

    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be a boolean.")

    return value


def _read_devices(data: dict[str, object]) -> tuple[str, ...]:
    value = data.get("devices")

    if not isinstance(value, list) or not value:
        raise ValueError("'devices' must be a non-empty list.")

    raw_devices = cast(list[object], value)
    devices: list[str] = []

    for device in raw_devices:
        if not isinstance(device, str):
            raise ValueError("Each device must be a string.")

        devices.append(device)

    unsupported = set(devices) - SUPPORTED_DEVICES

    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported devices: {names}.")

    if len(devices) != len(set(devices)):
        raise ValueError("'devices' must not contain duplicates.")

    return tuple(devices)


def _read_batch_sizes(data: dict[str, object]) -> tuple[int, ...]:
    value = data.get("batch_sizes")

    if not isinstance(value, list) or not value:
        raise ValueError("'batch_sizes' must be a non-empty list.")

    raw_batch_sizes = cast(list[object], value)
    batch_sizes: list[int] = []

    for batch_size in raw_batch_sizes:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("Each batch size must be a positive integer.")

        batch_sizes.append(batch_size)

    if len(batch_sizes) != len(set(batch_sizes)):
        raise ValueError("'batch_sizes' must not contain duplicates.")

    return tuple(batch_sizes)


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load a YAML file and return validated benchmark settings."""

    config_path = Path(path)

    with config_path.open(encoding="utf-8") as file:
        raw_data: object = yaml.safe_load(file)

    if not isinstance(raw_data, dict):
        raise ValueError("Benchmark configuration must be a YAML mapping.")

    raw_mapping = cast(dict[object, object], raw_data)
    data: dict[str, object] = {}

    for key, value in raw_mapping.items():
        if not isinstance(key, str):
            raise ValueError("Benchmark configuration keys must be strings.")

        data[key] = value

    return BenchmarkConfig(
        seed=_read_integer(data, "seed", minimum=0),
        devices=_read_devices(data),
        batch_sizes=_read_batch_sizes(data),
        epochs=_read_integer(data, "epochs", minimum=1),
        repetitions=_read_integer(data, "repetitions", minimum=1),
        warmup_batches=_read_integer(data, "warmup_batches", minimum=0),
        num_workers=_read_integer(data, "num_workers", minimum=0),
        pin_memory=_read_boolean(data, "pin_memory"),
        mixed_precision=_read_boolean(data, "mixed_precision"),
    )
