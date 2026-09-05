# pyright: reportUnknownMemberType=false

"""Create a LinkedIn-ready social card from saved benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum
from pathlib import Path
from typing import Final, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

RESULTS_FILE_NAME: Final = "benchmark_results.csv"
CONFIGURATION_FILE_NAME: Final = "configuration.json"
ENVIRONMENT_FILE_NAME: Final = "environment.json"
SOCIAL_CARD_FILE_NAME: Final = "benchmark_social_card.png"

IMAGE_WIDTH_PIXELS: Final = 1080
IMAGE_HEIGHT_PIXELS: Final = 1350
IMAGE_DPI: Final = 100

BACKGROUND_COLOR: Final = "#FFFFFF"
PRIMARY_TEXT_COLOR: Final = "#171717"
SECONDARY_TEXT_COLOR: Final = "#888888"
CPU_COLOR: Final = "#3F7FEF"
GPU_COLOR: Final = "#54B85A"
TRACK_COLOR: Final = "#ECECEC"
DIVIDER_COLOR: Final = "#DDDDDD"

GITHUB_USERNAME: Final = "aliraaza95"


@dataclass(frozen=True)
class SocialCardArtifact:
    """Location of the generated social-card image."""

    image_file: Path


@dataclass(frozen=True)
class _ResultRow:
    """Result values needed by the social-card generator."""

    device: str
    batch_size: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _BenchmarkMatrix:
    """Validated two-device benchmark matrix."""

    devices: tuple[str, str]
    batch_sizes: tuple[int, ...]
    repetitions: int

    @property
    def runs_per_device(self) -> int:
        return len(self.batch_sizes) * self.repetitions

    @property
    def total_runs(self) -> int:
        return len(self.devices) * self.runs_per_device


@dataclass(frozen=True)
class _AggregateComparison:
    """Aggregate measured times for equivalent device workloads."""

    slower_device: str
    faster_device: str
    slower_seconds: float
    faster_seconds: float
    speedup: float
    runs_per_device: int
    total_runs: int
    batch_size_count: int


def _require_text(
    row: Mapping[str, str | None],
    field_name: str,
) -> str:
    value = row.get(field_name)

    if value is None or not value.strip():
        raise ValueError(
            f"Missing '{field_name}' in benchmark results."
        )

    return value.strip()


def _read_result_rows(
    path: Path,
) -> tuple[_ResultRow, ...]:
    rows: list[_ResultRow] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for raw_row in reader:
            device = _require_text(
                raw_row,
                "device",
            )
            batch_size_text = _require_text(
                raw_row,
                "batch_size",
            )
            elapsed_text = _require_text(
                raw_row,
                "elapsed_seconds",
            )

            try:
                batch_size = int(batch_size_text)
                elapsed_seconds = float(elapsed_text)
            except ValueError as exc:
                raise ValueError(
                    "Benchmark results contain "
                    "an invalid numeric value."
                ) from exc

            if batch_size <= 0:
                raise ValueError(
                    "Batch sizes must be positive integers."
                )

            if elapsed_seconds <= 0.0:
                raise ValueError(
                    "Elapsed times must be positive numbers."
                )

            rows.append(
                _ResultRow(
                    device=device,
                    batch_size=batch_size,
                    elapsed_seconds=elapsed_seconds,
                )
            )

    if not rows:
        raise ValueError(
            "The benchmark results file contains no rows."
        )

    return tuple(rows)


def _read_json_object(
    path: Path,
) -> dict[str, object]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        value = cast(
            object,
            json.load(file),
        )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object in: {path}"
        )

    return cast(
        dict[str, object],
        value,
    )


def _require_positive_integer(
    data: Mapping[str, object],
    key: str,
) -> int:
    value = data.get(key)

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(
            f"Expected '{key}' to be a positive "
            "integer in configuration.json."
        )

    return value


def _require_devices(
    configuration: Mapping[str, object],
) -> tuple[str, str]:
    value = configuration.get("devices")

    if not isinstance(value, list):
        raise ValueError(
            "Expected 'devices' to be a list "
            "in configuration.json."
        )

    devices: list[str] = []

    for item in cast(
        list[object],
        value,
    ):
        if (
            not isinstance(item, str)
            or not item.strip()
        ):
            raise ValueError(
                "Every configured device must "
                "be a non-empty string."
            )

        devices.append(
            item.strip()
        )

    if (
        len(devices) != 2
        or len(set(devices)) != 2
    ):
        raise ValueError(
            "The aggregate social card requires "
            "exactly two configured devices."
        )

    return (
        devices[0],
        devices[1],
    )


def _require_batch_sizes(
    configuration: Mapping[str, object],
) -> tuple[int, ...]:
    value = configuration.get(
        "batch_sizes"
    )

    if not isinstance(value, list):
        raise ValueError(
            "Expected 'batch_sizes' to be a list "
            "in configuration.json."
        )

    batch_sizes: list[int] = []

    for item in cast(
        list[object],
        value,
    ):
        if (
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
        ):
            raise ValueError(
                "Every configured batch size must "
                "be a positive integer."
            )

        batch_sizes.append(item)

    if (
        not batch_sizes
        or len(set(batch_sizes))
        != len(batch_sizes)
    ):
        raise ValueError(
            "Configured batch sizes must be "
            "non-empty and unique."
        )

    return tuple(
        batch_sizes
    )


def _benchmark_matrix(
    configuration: Mapping[str, object],
) -> _BenchmarkMatrix:
    return _BenchmarkMatrix(
        devices=_require_devices(
            configuration
        ),
        batch_sizes=_require_batch_sizes(
            configuration
        ),
        repetitions=_require_positive_integer(
            configuration,
            "repetitions",
        ),
    )


def _validate_result_matrix(
    rows: Sequence[_ResultRow],
    matrix: _BenchmarkMatrix,
) -> None:
    expected_counts = Counter(
        (
            device,
            batch_size,
        )
        for device in matrix.devices
        for batch_size in matrix.batch_sizes
        for _ in range(
            matrix.repetitions
        )
    )

    measured_counts = Counter(
        (
            row.device,
            row.batch_size,
        )
        for row in rows
    )

    if measured_counts != expected_counts:
        raise ValueError(
            "benchmark_results.csv does not contain "
            "the complete device, batch-size, and "
            "repetition matrix from configuration.json."
        )


def _aggregate_comparison(
    rows: Sequence[_ResultRow],
    matrix: _BenchmarkMatrix,
) -> _AggregateComparison:
    _validate_result_matrix(
        rows,
        matrix,
    )

    first_device, second_device = (
        matrix.devices
    )

    first_seconds = fsum(
        row.elapsed_seconds
        for row in rows
        if row.device == first_device
    )

    second_seconds = fsum(
        row.elapsed_seconds
        for row in rows
        if row.device == second_device
    )

    if first_seconds >= second_seconds:
        slower_device = first_device
        faster_device = second_device
        slower_seconds = first_seconds
        faster_seconds = second_seconds
    else:
        slower_device = second_device
        faster_device = first_device
        slower_seconds = second_seconds
        faster_seconds = first_seconds

    return _AggregateComparison(
        slower_device=slower_device,
        faster_device=faster_device,
        slower_seconds=slower_seconds,
        faster_seconds=faster_seconds,
        speedup=(
            slower_seconds
            / faster_seconds
        ),
        runs_per_device=(
            matrix.runs_per_device
        ),
        total_runs=matrix.total_runs,
        batch_size_count=len(
            matrix.batch_sizes
        ),
    )


def _gpu_name(
    environment: Mapping[str, object],
) -> str | None:
    gpus_value = environment.get(
        "gpus"
    )

    if (
        not isinstance(gpus_value, list)
        or not gpus_value
    ):
        return None

    first_gpu = cast(
        list[object],
        gpus_value,
    )[0]

    if not isinstance(
        first_gpu,
        dict,
    ):
        return None

    name = cast(
        dict[str, object],
        first_gpu,
    ).get("name")

    if (
        not isinstance(name, str)
        or not name.strip()
    ):
        return None

    return " ".join(
        name.split()
    )


def _local_windows_cpu_name() -> str | None:
    if platform.system() != "Windows":
        return None

    try:
        import winreg

        key_path = (
            r"HARDWARE\DESCRIPTION\System"
            r"\CentralProcessor\0"
        )

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            key_path,
        ) as key:
            value = cast(
                object,
                winreg.QueryValueEx(
                    key,
                    "ProcessorNameString",
                )[0],
            )
    except OSError:
        return None

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        return None

    return " ".join(
        value.split()
    )


def _is_generic_processor_identifier(
    value: str,
) -> bool:
    normalized = value.casefold()

    return all(
        word in normalized
        for word in (
            "family",
            "model",
            "stepping",
        )
    )


def _cpu_name(
    environment: Mapping[str, object],
) -> str:
    for key in (
        "cpu_model",
        "processor_name",
    ):
        value = environment.get(key)

        if (
            isinstance(value, str)
            and value.strip()
            and value.strip().casefold()
            != "unknown"
        ):
            return " ".join(
                value.split()
            )

    identifier = environment.get(
        "processor_identifier"
    )

    if (
        isinstance(identifier, str)
        and identifier.strip()
        and identifier.strip().casefold()
        != "unknown"
        and not _is_generic_processor_identifier(
            identifier
        )
    ):
        return " ".join(
            identifier.split()
        )

    # Compatibility fallback for old environment.json
    # files created before cpu_model was recorded.
    local_name = (
        _local_windows_cpu_name()
    )

    if local_name:
        return local_name

    return "Unknown CPU"


def _display_device(
    device: str,
) -> str:
    normalized = device.casefold()

    if normalized == "cpu":
        return "CPU"

    if normalized in {
        "cuda",
        "gpu",
    }:
        return "GPU"

    return device.upper()


def _device_color(
    device: str,
) -> str:
    if device.casefold() in {
        "cuda",
        "gpu",
    }:
        return GPU_COLOR

    return CPU_COLOR


def _device_kind(
    device: str,
) -> str:
    normalized = device.casefold()

    if normalized == "cpu":
        return "The CPU"

    if normalized in {
        "cuda",
        "gpu",
    }:
        return "The GPU"

    return "The faster device"


def _format_duration(
    seconds: float,
) -> str:
    rounded_seconds = max(
        1,
        round(seconds),
    )

    hours, remainder = divmod(
        rounded_seconds,
        3600,
    )
    minutes, remaining_seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours} hr "
            f"{minutes:02d} min "
            f"{remaining_seconds:02d} sec"
        )

    if minutes:
        return (
            f"{minutes} min "
            f"{remaining_seconds:02d} sec"
        )

    return (
        f"{remaining_seconds} sec"
    )


def _draw_bar(
    figure: Figure,
    *,
    y: float,
    label: str,
    duration: str,
    fraction: float,
    color: str,
) -> None:
    left = 0.065
    width = 0.87
    height = 0.055

    figure.text(
        left,
        y + 0.080,
        label,
        color=PRIMARY_TEXT_COLOR,
        fontsize=15,
    )

    figure.text(
        left + width,
        y + 0.080,
        duration,
        color=PRIMARY_TEXT_COLOR,
        fontsize=15,
        horizontalalignment="right",
    )

    figure.add_artist(
        Rectangle(
            (
                left,
                y,
            ),
            width,
            height,
            transform=figure.transFigure,
            facecolor=TRACK_COLOR,
            edgecolor="none",
        )
    )

    figure.add_artist(
        Rectangle(
            (
                left,
                y,
            ),
            width * fraction,
            height,
            transform=figure.transFigure,
            facecolor=color,
            edgecolor="none",
        )
    )


def _build_figure(
    comparison: _AggregateComparison,
    *,
    cpu_name: str,
    gpu_name: str | None,
    epochs: int,
    repetitions: int,
) -> Figure:
    slower_label = (
        f"{_display_device(comparison.slower_device)} · "
        f"{comparison.runs_per_device} measured runs"
    )

    faster_label = (
        f"{_display_device(comparison.faster_device)} · "
        f"{comparison.runs_per_device} measured runs"
    )

    slower_duration = _format_duration(
        comparison.slower_seconds
    )
    faster_duration = _format_duration(
        comparison.faster_seconds
    )
    saved_duration = _format_duration(
        comparison.slower_seconds
        - comparison.faster_seconds
    )

    figure = plt.figure(
        figsize=(
            IMAGE_WIDTH_PIXELS
            / IMAGE_DPI,
            IMAGE_HEIGHT_PIXELS
            / IMAGE_DPI,
        ),
        dpi=IMAGE_DPI,
        facecolor=BACKGROUND_COLOR,
    )

    figure.text(
        0.065,
        0.930,
        "TOTAL MEASURED TRAINING TIME",
        color=SECONDARY_TEXT_COLOR,
        fontsize=15,
        fontweight="semibold",
    )

    figure.text(
        0.935,
        0.930,
        "01 / 03",
        color=SECONDARY_TEXT_COLOR,
        fontsize=14,
        fontweight="semibold",
        horizontalalignment="right",
    )

    figure.text(
        0.065,
        0.890,
        f"CPU: {cpu_name}",
        color=SECONDARY_TEXT_COLOR,
        fontsize=12.5,
    )

    figure.text(
        0.065,
        0.864,
        (
            "GPU: "
            f"{gpu_name or 'Unknown GPU'}"
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=12.5,
    )

    figure.text(
        0.065,
        0.790,
        (
            f"{slower_duration} vs "
            f"{faster_duration}"
        ),
        color=PRIMARY_TEXT_COLOR,
        fontsize=31,
        fontweight="semibold",
    )

    figure.text(
        0.065,
        0.730,
        (
            f"Complete {comparison.total_runs}-run matrix. "
            f"{_device_kind(comparison.faster_device)} "
            "completed\n"
            "the same measured workload "
            f"{comparison.speedup:.1f}x faster."
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=16,
        linespacing=1.35,
    )

    _draw_bar(
        figure,
        y=0.555,
        label=slower_label,
        duration=slower_duration,
        fraction=1.0,
        color=_device_color(
            comparison.slower_device
        ),
    )

    _draw_bar(
        figure,
        y=0.365,
        label=faster_label,
        duration=faster_duration,
        fraction=(
            comparison.faster_seconds
            / comparison.slower_seconds
        ),
        color=_device_color(
            comparison.faster_device
        ),
    )

    figure.add_artist(
        Rectangle(
            (
                0.065,
                0.260,
            ),
            0.004,
            0.035,
            transform=figure.transFigure,
            facecolor=_device_color(
                comparison.faster_device
            ),
            edgecolor="none",
        )
    )

    faster_kind = _device_kind(
        comparison.faster_device
    ).removeprefix("The ")

    figure.text(
        0.090,
        0.267,
        (
            f"{faster_kind} saved about "
            f"{saved_duration} across "
            "the complete matrix"
        ),
        color=PRIMARY_TEXT_COLOR,
        fontsize=15,
        fontweight="semibold",
    )

    figure.add_artist(
        Rectangle(
            (
                0.065,
                0.180,
            ),
            0.87,
            0.0012,
            transform=figure.transFigure,
            facecolor=DIVIDER_COLOR,
            edgecolor="none",
        )
    )

    figure.text(
        0.065,
        0.135,
        (
            "CIFAR-10 | ResNet-18 | "
            f"{comparison.batch_size_count} batch sizes | "
            f"{epochs} epochs"
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=11.5,
    )

    figure.text(
        0.935,
        0.135,
        (
            f"{repetitions} repetitions per device "
            "and batch size"
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=11.5,
        horizontalalignment="right",
    )

    figure.text(
        0.935,
        0.090,
        (
            "GitHub · "
            f"@{GITHUB_USERNAME.lstrip('@')}"
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=11.5,
        horizontalalignment="right",
    )

    return figure


def _resolve_run_directory(
    analysis: object,
    output_directory: str | Path | None,
) -> Path:
    if output_directory is not None:
        return Path(
            output_directory
        )

    if isinstance(
        analysis,
        (
            str,
            Path,
        ),
    ):
        return Path(
            analysis
        )

    raise TypeError(
        "Pass the run directory as the first argument "
        "or as 'output_directory'."
    )


def create_social_card(
    analysis: object,
    environment: object | None = None,
    output_directory: str | Path | None = None,
) -> SocialCardArtifact:
    """Create a 1080 by 1350 card from a saved run."""

    del environment

    run_directory = (
        _resolve_run_directory(
            analysis,
            output_directory,
        )
    )

    results_path = (
        run_directory
        / RESULTS_FILE_NAME
    )
    configuration_path = (
        run_directory
        / CONFIGURATION_FILE_NAME
    )
    environment_path = (
        run_directory
        / ENVIRONMENT_FILE_NAME
    )

    for required_path in (
        results_path,
        configuration_path,
        environment_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                "Required benchmark artifact "
                f"was not found: {required_path}"
            )

    rows = _read_result_rows(
        results_path
    )
    configuration = _read_json_object(
        configuration_path
    )
    environment_data = _read_json_object(
        environment_path
    )

    matrix = _benchmark_matrix(
        configuration
    )
    comparison = _aggregate_comparison(
        rows,
        matrix,
    )
    epochs = _require_positive_integer(
        configuration,
        "epochs",
    )

    figure = _build_figure(
        comparison,
        cpu_name=_cpu_name(
            environment_data
        ),
        gpu_name=_gpu_name(
            environment_data
        ),
        epochs=epochs,
        repetitions=matrix.repetitions,
    )

    output_path = (
        run_directory
        / SOCIAL_CARD_FILE_NAME
    )

    figure.savefig(
        output_path,
        dpi=IMAGE_DPI,
        facecolor=figure.get_facecolor(),
        bbox_inches=None,
    )

    plt.close(
        figure
    )

    return SocialCardArtifact(
        image_file=output_path
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a LinkedIn-ready social card "
            "from a saved benchmark run."
        )
    )

    parser.add_argument(
        "run_directory",
        type=Path,
        help=(
            "Directory containing benchmark_results.csv "
            "and metadata JSON."
        ),
    )

    return parser


def main() -> None:
    """Run the social-card generator as a module."""

    arguments = (
        _build_parser().parse_args()
    )

    artifact = create_social_card(
        arguments.run_directory
    )

    print(
        f"Social card: {artifact.image_file}"
    )


if __name__ == "__main__":
    main()
