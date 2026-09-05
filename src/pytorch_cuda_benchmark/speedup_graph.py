# pyright: reportUnknownMemberType=false

"""Create a GPU speedup graph from saved benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Final, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

RESULTS_FILE_NAME: Final = "benchmark_results.csv"
CONFIGURATION_FILE_NAME: Final = "configuration.json"
SPEEDUP_GRAPH_FILE_NAME: Final = "benchmark_speedup_graph.png"

IMAGE_WIDTH_PIXELS: Final = 1080
IMAGE_HEIGHT_PIXELS: Final = 1350
IMAGE_DPI: Final = 100

BACKGROUND_COLOR: Final = "#FFFFFF"
PRIMARY_TEXT_COLOR: Final = "#171717"
SECONDARY_TEXT_COLOR: Final = "#888888"
CPU_COLOR: Final = "#3F7FEF"
GPU_COLOR: Final = "#54B85A"
GRID_COLOR: Final = "#E2E2E2"
DIVIDER_COLOR: Final = "#DDDDDD"

GITHUB_USERNAME: Final = "aliraaza95"


@dataclass(frozen=True)
class SpeedupGraphArtifact:
    """Location of the generated speedup graph."""

    image_file: Path


@dataclass(frozen=True)
class _ResultRow:
    """Result values needed to calculate GPU speedup."""

    device: str
    batch_size: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _BenchmarkMatrix:
    """Expected CPU and GPU benchmark matrix."""

    devices: tuple[str, str]
    batch_sizes: tuple[int, ...]
    repetitions: int

    @property
    def total_runs(self) -> int:
        return (
            len(self.devices)
            * len(self.batch_sizes)
            * self.repetitions
        )


@dataclass(frozen=True)
class _SpeedupRow:
    """Mean CPU-to-GPU speedup for one batch size."""

    batch_size: int
    cpu_seconds: float
    gpu_seconds: float
    speedup: float


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


def _read_result_rows(path: Path) -> tuple[_ResultRow, ...]:
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
                    "Benchmark results contain an invalid "
                    "numeric value."
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


def _read_json_object(path: Path) -> dict[str, object]:
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


def _device_category(device: str) -> str | None:
    normalized = device.casefold()

    if normalized == "cpu":
        return "cpu"

    if normalized in {
        "cuda",
        "gpu",
    }:
        return "gpu"

    return None


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
            f"Expected '{key}' to be a positive integer "
            "in configuration.json."
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
                "Every configured device must be "
                "a non-empty string."
            )

        devices.append(
            item.strip()
        )

    categories = {
        _device_category(device)
        for device in devices
    }

    if (
        len(devices) != 2
        or len(set(devices)) != 2
        or categories != {"cpu", "gpu"}
    ):
        raise ValueError(
            "The speedup graph requires exactly one CPU "
            "device and one CUDA or GPU device."
        )

    return (
        devices[0],
        devices[1],
    )


def _require_batch_sizes(
    configuration: Mapping[str, object],
) -> tuple[int, ...]:
    value = configuration.get("batch_sizes")

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
                "Every configured batch size must be "
                "a positive integer."
            )

        batch_sizes.append(item)

    if (
        not batch_sizes
        or len(set(batch_sizes)) != len(batch_sizes)
    ):
        raise ValueError(
            "Configured batch sizes must be "
            "non-empty and unique."
        )

    return tuple(batch_sizes)


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
        (device, batch_size)
        for device in matrix.devices
        for batch_size in matrix.batch_sizes
        for _ in range(matrix.repetitions)
    )
    measured_counts = Counter(
        (row.device, row.batch_size)
        for row in rows
    )

    if measured_counts != expected_counts:
        raise ValueError(
            "benchmark_results.csv does not contain "
            "the complete device, batch-size, and "
            "repetition matrix from configuration.json."
        )


def _calculate_speedups(
    rows: Sequence[_ResultRow],
    matrix: _BenchmarkMatrix,
) -> tuple[_SpeedupRow, ...]:
    _validate_result_matrix(
        rows,
        matrix,
    )

    grouped: defaultdict[
        tuple[str, int],
        list[float],
    ] = defaultdict(list)

    for row in rows:
        category = _device_category(
            row.device
        )

        if category is None:
            raise ValueError(
                f"Unsupported benchmark device: {row.device}"
            )

        grouped[
            (
                category,
                row.batch_size,
            )
        ].append(
            row.elapsed_seconds
        )

    speedups: list[_SpeedupRow] = []

    for batch_size in sorted(
        matrix.batch_sizes
    ):
        cpu_seconds = fmean(
            grouped[
                (
                    "cpu",
                    batch_size,
                )
            ]
        )
        gpu_seconds = fmean(
            grouped[
                (
                    "gpu",
                    batch_size,
                )
            ]
        )

        speedups.append(
            _SpeedupRow(
                batch_size=batch_size,
                cpu_seconds=cpu_seconds,
                gpu_seconds=gpu_seconds,
                speedup=(
                    cpu_seconds
                    / gpu_seconds
                ),
            )
        )

    return tuple(speedups)


def _format_duration(seconds: float) -> str:
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

    return f"{remaining_seconds} sec"


def _build_figure(
    speedups: Sequence[_SpeedupRow],
    *,
    epochs: int,
    repetitions: int,
    total_runs: int,
) -> Figure:
    values = [
        row.speedup
        for row in speedups
    ]
    minimum_speedup = min(values)
    best_speedup = max(values)
    axis_maximum = max(
        2.0,
        best_speedup * 1.18,
    )

    figure = plt.figure(
        figsize=(
            IMAGE_WIDTH_PIXELS / IMAGE_DPI,
            IMAGE_HEIGHT_PIXELS / IMAGE_DPI,
        ),
        dpi=IMAGE_DPI,
        facecolor=BACKGROUND_COLOR,
    )

    figure.text(
        0.065,
        0.930,
        "GPU SPEEDUP BY BATCH SIZE",
        color=SECONDARY_TEXT_COLOR,
        fontsize=15,
        fontweight="semibold",
    )

    figure.text(
        0.935,
        0.930,
        "02 / 03",
        color=SECONDARY_TEXT_COLOR,
        fontsize=14,
        fontweight="semibold",
        horizontalalignment="right",
    )

    figure.text(
        0.065,
        0.850,
        (
            f"{minimum_speedup:.1f}x to "
            f"{best_speedup:.1f}x faster"
        ),
        color=PRIMARY_TEXT_COLOR,
        fontsize=31,
        fontweight="semibold",
    )

    figure.text(
        0.065,
        0.790,
        (
            "Each bar compares the GPU with the CPU "
            "at the same batch size.\n"
            "CPU is the 1.0x reference. "
            "Higher is better."
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=16,
        linespacing=1.35,
    )

    axes = figure.add_axes(
        (
            0.145,
            0.315,
            0.790,
            0.385,
        )
    )
    axes.set_facecolor(
        BACKGROUND_COLOR
    )
    axes.set_axisbelow(True)

    positions = [
        float(index) * 1.35
        for index in range(
            len(speedups)
        )
    ]

    bars = cast(
        Sequence[Rectangle],
        axes.barh(
            positions,
            values,
            height=0.44,
            color=GPU_COLOR,
            edgecolor="none",
            zorder=3,
        ).patches,
    )

    label_offset = (
        axis_maximum
        * 0.018
    )

    for bar, row in zip(
        bars,
        speedups,
        strict=True,
    ):
        center_y = (
            bar.get_y()
            + bar.get_height() / 2.0
        )

        axes.text(
            row.speedup + label_offset,
            center_y,
            f"{row.speedup:.2f}x",
            color=PRIMARY_TEXT_COLOR,
            fontsize=15,
            fontweight="semibold",
            verticalalignment="center",
        )

        axes.text(
            1.0 + label_offset,
            center_y + 0.41,
            (
                f"CPU {_format_duration(row.cpu_seconds)}"
                "  |  "
                f"GPU {_format_duration(row.gpu_seconds)}"
            ),
            color=SECONDARY_TEXT_COLOR,
            fontsize=11.5,
            verticalalignment="center",
        )

    axes.axvline(
        1.0,
        color=CPU_COLOR,
        linewidth=2.0,
        zorder=2,
    )

    axes.text(
        1.0 + label_offset,
        -0.52,
        "CPU baseline 1.0x",
        color=CPU_COLOR,
        fontsize=11.5,
        fontweight="semibold",
        verticalalignment="center",
    )

    axes.set_yticks(
        positions,
        labels=[
            f"Batch {row.batch_size}"
            for row in speedups
        ],
    )

    axes.set_xlim(
        0.0,
        axis_maximum,
    )
    axes.set_ylim(
        positions[-1] + 0.75,
        -0.75,
    )

    axes.xaxis.set_major_locator(
        MaxNLocator(
            nbins=5,
            integer=True,
        )
    )

    axes.set_xlabel(
        "GPU speedup over CPU (x)",
        color=SECONDARY_TEXT_COLOR,
        fontsize=12.5,
        labelpad=14,
    )

    axes.tick_params(
        axis="x",
        colors=SECONDARY_TEXT_COLOR,
        labelsize=11.5,
        length=0,
        pad=8,
    )

    axes.tick_params(
        axis="y",
        colors=PRIMARY_TEXT_COLOR,
        labelsize=14,
        length=0,
        pad=14,
    )

    axes.xaxis.grid(
        True,
        color=GRID_COLOR,
        linewidth=1.0,
        zorder=0,
    )
    axes.yaxis.grid(False)

    for spine in axes.spines.values():
        spine.set_visible(False)

    figure.text(
        0.065,
        0.235,
        (
            "Speedup = mean CPU training time / "
            "mean GPU training time"
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=11.5,
    )

    figure.text(
        0.935,
        0.235,
        f"Derived from {total_runs} measured runs",
        color=SECONDARY_TEXT_COLOR,
        fontsize=11.5,
        horizontalalignment="right",
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
            f"{len(speedups)} batch sizes | "
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
        f"GitHub · @{GITHUB_USERNAME.lstrip('@')}",
        color=SECONDARY_TEXT_COLOR,
        fontsize=11.5,
        horizontalalignment="right",
    )

    return figure


def create_speedup_graph(
    run_directory: str | Path,
) -> SpeedupGraphArtifact:
    """Create a 1080 by 1350 speedup graph from a saved run."""

    directory = Path(
        run_directory
    )
    results_path = (
        directory
        / RESULTS_FILE_NAME
    )
    configuration_path = (
        directory
        / CONFIGURATION_FILE_NAME
    )

    for required_path in (
        results_path,
        configuration_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                "Required benchmark artifact was "
                f"not found: {required_path}"
            )

    rows = _read_result_rows(
        results_path
    )
    configuration = _read_json_object(
        configuration_path
    )
    matrix = _benchmark_matrix(
        configuration
    )
    speedups = _calculate_speedups(
        rows,
        matrix,
    )
    epochs = _require_positive_integer(
        configuration,
        "epochs",
    )

    figure = _build_figure(
        speedups,
        epochs=epochs,
        repetitions=matrix.repetitions,
        total_runs=matrix.total_runs,
    )

    output_path = (
        directory
        / SPEEDUP_GRAPH_FILE_NAME
    )

    figure.savefig(
        output_path,
        dpi=IMAGE_DPI,
        facecolor=figure.get_facecolor(),
        bbox_inches=None,
    )
    plt.close(figure)

    return SpeedupGraphArtifact(
        image_file=output_path
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a GPU speedup graph "
            "from saved benchmark artifacts."
        )
    )

    parser.add_argument(
        "run_directory",
        type=Path,
        help=(
            "Directory containing saved "
            "benchmark artifacts."
        ),
    )

    return parser


def main() -> None:
    """Run the speedup-graph generator as a module."""

    arguments = _build_parser().parse_args()

    artifact = create_speedup_graph(
        arguments.run_directory
    )

    print(
        f"Speedup graph: {artifact.image_file}"
    )


if __name__ == "__main__":
    main()
