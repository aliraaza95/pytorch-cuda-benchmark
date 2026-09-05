# pyright: reportUnknownMemberType=false

"""Create a terminal-style receipt from saved benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum
from pathlib import Path
from typing import Final, Literal, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

RESULTS_FILE_NAME: Final = "benchmark_results.csv"
CONFIGURATION_FILE_NAME: Final = "configuration.json"
ENVIRONMENT_FILE_NAME: Final = "environment.json"
BENCHMARK_RECEIPT_FILE_NAME: Final = "benchmark_receipt.png"

IMAGE_WIDTH_PIXELS: Final = 1080
IMAGE_HEIGHT_PIXELS: Final = 1350
IMAGE_DPI: Final = 100

BACKGROUND_COLOR: Final = "#FFFFFF"
PRIMARY_TEXT_COLOR: Final = "#171717"
SECONDARY_TEXT_COLOR: Final = "#888888"
DIVIDER_COLOR: Final = "#DDDDDD"
CPU_COLOR: Final = "#3F7FEF"
GPU_COLOR: Final = "#54B85A"

TERMINAL_BACKGROUND_COLOR: Final = "#050B17"
TERMINAL_HEADER_COLOR: Final = "#111C32"
TERMINAL_BORDER_COLOR: Final = "#24344F"
TERMINAL_TEXT_COLOR: Final = "#E6EAF2"
TERMINAL_MUTED_COLOR: Final = "#9AA7BD"
TERMINAL_SUCCESS_COLOR: Final = "#4FD1AE"
TERMINAL_RED_COLOR: Final = "#F26D6D"
TERMINAL_YELLOW_COLOR: Final = "#F6C453"

GITHUB_USERNAME: Final = "aliraaza95"


@dataclass(frozen=True)
class BenchmarkReceiptArtifact:
    """Location of the generated benchmark-receipt image."""

    image_file: Path


@dataclass(frozen=True)
class _ResultRow:
    """Saved values required by the benchmark receipt."""

    device: str
    batch_size: int
    repetition: int
    seed: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _BenchmarkMatrix:
    """Validated benchmark configuration."""

    devices: tuple[str, str]
    batch_sizes: tuple[int, ...]
    repetitions: int
    epochs: int
    mixed_precision: bool

    @property
    def runs_per_device(self) -> int:
        return (
            len(self.batch_sizes)
            * self.repetitions
        )

    @property
    def total_runs(self) -> int:
        return (
            len(self.devices)
            * self.runs_per_device
        )


@dataclass(frozen=True)
class _ReceiptData:
    """Derived values displayed by the benchmark receipt."""

    matrix: _BenchmarkMatrix
    final_row: _ResultRow
    cpu_seconds: float
    gpu_seconds: float
    cpu_runs: int
    gpu_runs: int
    run_name: str


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
            repetition_text = _require_text(
                raw_row,
                "repetition",
            )
            seed_text = _require_text(
                raw_row,
                "seed",
            )
            elapsed_text = _require_text(
                raw_row,
                "elapsed_seconds",
            )

            try:
                batch_size = int(batch_size_text)
                repetition = int(repetition_text)
                seed = int(seed_text)
                elapsed_seconds = float(
                    elapsed_text
                )
            except ValueError as exc:
                raise ValueError(
                    "Benchmark results contain an invalid "
                    "numeric value."
                ) from exc

            if batch_size <= 0:
                raise ValueError(
                    "Batch sizes must be positive integers."
                )

            if repetition <= 0:
                raise ValueError(
                    "Repetition numbers must be "
                    "positive integers."
                )

            if elapsed_seconds <= 0.0:
                raise ValueError(
                    "Elapsed times must be positive numbers."
                )

            rows.append(
                _ResultRow(
                    device=device,
                    batch_size=batch_size,
                    repetition=repetition,
                    seed=seed,
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


def _device_category(
    device: str,
) -> str | None:
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


def _require_boolean(
    data: Mapping[str, object],
    key: str,
) -> bool:
    value = data.get(key)

    if not isinstance(value, bool):
        raise ValueError(
            f"Expected '{key}' to be a boolean "
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
            "The benchmark receipt requires exactly "
            "one CPU device and one CUDA or GPU device."
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
        epochs=_require_positive_integer(
            configuration,
            "epochs",
        ),
        mixed_precision=_require_boolean(
            configuration,
            "mixed_precision",
        ),
    )


def _validate_result_matrix(
    rows: Sequence[_ResultRow],
    matrix: _BenchmarkMatrix,
) -> None:
    expected_rows = Counter(
        (
            device,
            batch_size,
            repetition,
        )
        for device in matrix.devices
        for batch_size in matrix.batch_sizes
        for repetition in range(
            1,
            matrix.repetitions + 1,
        )
    )

    measured_rows = Counter(
        (
            row.device,
            row.batch_size,
            row.repetition,
        )
        for row in rows
    )

    if measured_rows != expected_rows:
        raise ValueError(
            "benchmark_results.csv does not contain "
            "the complete device, batch-size, and "
            "repetition matrix from configuration.json."
        )


def _receipt_data(
    rows: Sequence[_ResultRow],
    matrix: _BenchmarkMatrix,
    run_directory: Path,
) -> _ReceiptData:
    _validate_result_matrix(
        rows,
        matrix,
    )

    cpu_rows = [
        row
        for row in rows
        if _device_category(row.device) == "cpu"
    ]

    gpu_rows = [
        row
        for row in rows
        if _device_category(row.device) == "gpu"
    ]

    return _ReceiptData(
        matrix=matrix,
        final_row=rows[-1],
        cpu_seconds=fsum(
            row.elapsed_seconds
            for row in cpu_rows
        ),
        gpu_seconds=fsum(
            row.elapsed_seconds
            for row in gpu_rows
        ),
        cpu_runs=len(cpu_rows),
        gpu_runs=len(gpu_rows),
        run_name=run_directory.name,
    )


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

    return f"{remaining_seconds} sec"


def _precision_label(
    matrix: _BenchmarkMatrix,
) -> str:
    if matrix.mixed_precision:
        return "CUDA AMP"

    return "FP32"


def _gpu_device_label(
    matrix: _BenchmarkMatrix,
) -> str:
    for device in matrix.devices:
        if _device_category(device) == "gpu":
            if device.casefold() == "cuda":
                return "CUDA"

            return "GPU"

    raise ValueError(
        "A GPU device was not found in the matrix."
    )


def _terminal_text(
    figure: Figure,
    *,
    y: float,
    text: str,
    color: str = TERMINAL_TEXT_COLOR,
    weight: Literal[
        "normal",
        "semibold",
    ] = "normal",
) -> None:
    figure.text(
        0.090,
        y,
        text,
        color=color,
        fontsize=12.5,
        fontfamily="monospace",
        fontweight=weight,
    )


def _build_figure(
    data: _ReceiptData,
) -> Figure:
    matrix = data.matrix
    gpu_label = _gpu_device_label(
        matrix
    )
    final_row = data.final_row

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
        "BENCHMARK RECEIPT",
        color=SECONDARY_TEXT_COLOR,
        fontsize=15,
        fontweight="semibold",
    )

    figure.text(
        0.935,
        0.930,
        "03 / 03",
        color=SECONDARY_TEXT_COLOR,
        fontsize=14,
        fontweight="semibold",
        horizontalalignment="right",
    )

    figure.text(
        0.065,
        0.850,
        (
            f"{matrix.total_runs} of "
            f"{matrix.total_runs} runs completed."
        ),
        color=PRIMARY_TEXT_COLOR,
        fontsize=31,
        fontweight="semibold",
    )

    figure.text(
        0.065,
        0.795,
        (
            "Completion evidence and totals derived "
            "from the saved raw CSV."
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=16,
    )

    terminal_left = 0.065
    terminal_bottom = 0.210
    terminal_width = 0.870
    terminal_height = 0.505
    terminal_top = (
        terminal_bottom
        + terminal_height
    )
    header_height = 0.055

    figure.add_artist(
        FancyBboxPatch(
            (
                terminal_left,
                terminal_bottom,
            ),
            terminal_width,
            terminal_height,
            boxstyle=(
                "round,pad=0.0,"
                "rounding_size=0.015"
            ),
            transform=figure.transFigure,
            facecolor=TERMINAL_BACKGROUND_COLOR,
            edgecolor=TERMINAL_BORDER_COLOR,
            linewidth=1.2,
        )
    )

    figure.add_artist(
        Rectangle(
            (
                terminal_left,
                terminal_top - header_height,
            ),
            terminal_width,
            header_height,
            transform=figure.transFigure,
            facecolor=TERMINAL_HEADER_COLOR,
            edgecolor="none",
        )
    )

    dot_y = (
        terminal_top
        - header_height / 2.0
    )

    for x, color in (
        (
            0.095,
            TERMINAL_RED_COLOR,
        ),
        (
            0.122,
            TERMINAL_YELLOW_COLOR,
        ),
        (
            0.149,
            TERMINAL_SUCCESS_COLOR,
        ),
    ):
        figure.add_artist(
            Circle(
                (
                    x,
                    dot_y,
                ),
                0.008,
                transform=figure.transFigure,
                facecolor=color,
                edgecolor="none",
            )
        )

    figure.text(
        0.185,
        dot_y - 0.006,
        (
            "Windows PowerShell  ·  "
            "benchmark summary"
        ),
        color=TERMINAL_MUTED_COLOR,
        fontsize=12.5,
        fontfamily="monospace",
    )

    _terminal_text(
        figure,
        y=0.625,
        text=(
            f"[{matrix.total_runs}/"
            f"{matrix.total_runs}] "
            f"device={final_row.device} "
            f"batch_size={final_row.batch_size} "
            f"repetition={final_row.repetition} "
            f"seed={final_row.seed}"
        ),
        color=TERMINAL_SUCCESS_COLOR,
        weight="semibold",
    )

    _terminal_text(
        figure,
        y=0.585,
        text=(
            "Benchmark completed successfully."
        ),
        color=TERMINAL_SUCCESS_COLOR,
        weight="semibold",
    )

    _terminal_text(
        figure,
        y=0.540,
        text=(
            "Matrix       "
            f"{len(matrix.devices)} devices x "
            f"{len(matrix.batch_sizes)} batch sizes x "
            f"{matrix.repetitions} repetitions"
        ),
    )

    _terminal_text(
        figure,
        y=0.512,
        text=(
            f"Epochs       "
            f"{matrix.epochs} per run"
        ),
    )

    _terminal_text(
        figure,
        y=0.484,
        text=(
            f"Precision    "
            f"{_precision_label(matrix)}"
        ),
    )

    _terminal_text(
        figure,
        y=0.451,
        text="-" * 68,
        color=TERMINAL_MUTED_COLOR,
    )

    _terminal_text(
        figure,
        y=0.418,
        text=(
            f"CPU total    "
            f"{data.cpu_runs} measured runs | "
            f"{_format_duration(data.cpu_seconds)}"
        ),
        color=CPU_COLOR,
        weight="semibold",
    )

    _terminal_text(
        figure,
        y=0.390,
        text=(
            f"{gpu_label} total   "
            f"{data.gpu_runs} measured runs | "
            f"{_format_duration(data.gpu_seconds)}"
        ),
        color=GPU_COLOR,
        weight="semibold",
    )

    _terminal_text(
        figure,
        y=0.357,
        text="-" * 68,
        color=TERMINAL_MUTED_COLOR,
    )

    _terminal_text(
        figure,
        y=0.324,
        text=(
            f"Results      "
            f"{RESULTS_FILE_NAME}"
        ),
    )

    _terminal_text(
        figure,
        y=0.296,
        text=(
            f"Config       "
            f"{CONFIGURATION_FILE_NAME}"
        ),
    )

    _terminal_text(
        figure,
        y=0.268,
        text=(
            f"Environment  "
            f"{ENVIRONMENT_FILE_NAME}"
        ),
    )

    _terminal_text(
        figure,
        y=0.240,
        text=(
            f"Run          "
            f"{data.run_name}"
        ),
    )

    figure.text(
        0.065,
        0.160,
        (
            "Totals are sums of measured training "
            "intervals—not complete program "
            "wall-clock time."
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=12.5,
    )

    figure.add_artist(
        Rectangle(
            (
                0.065,
                0.115,
            ),
            0.870,
            0.0012,
            transform=figure.transFigure,
            facecolor=DIVIDER_COLOR,
            edgecolor="none",
        )
    )

    figure.text(
        0.065,
        0.075,
        (
            "Raw measurements remain "
            "auditable in CSV"
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=11.5,
    )

    figure.text(
        0.935,
        0.075,
        "Measured, not estimated",
        color=SECONDARY_TEXT_COLOR,
        fontsize=11.5,
        horizontalalignment="right",
    )

    figure.text(
        0.935,
        0.040,
        (
            f"GitHub · "
            f"@{GITHUB_USERNAME.lstrip('@')}"
        ),
        color=SECONDARY_TEXT_COLOR,
        fontsize=11.5,
        horizontalalignment="right",
    )

    return figure


def create_benchmark_receipt(
    run_directory: str | Path,
) -> BenchmarkReceiptArtifact:
    """Create a 1080 by 1350 receipt from a complete saved run."""

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
    environment_path = (
        directory
        / ENVIRONMENT_FILE_NAME
    )

    for required_path in (
        results_path,
        configuration_path,
        environment_path,
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
    _environment = _read_json_object(
        environment_path
    )
    matrix = _benchmark_matrix(
        configuration
    )
    data = _receipt_data(
        rows,
        matrix,
        directory,
    )

    figure = _build_figure(
        data
    )

    output_path = (
        directory
        / BENCHMARK_RECEIPT_FILE_NAME
    )

    figure.savefig(
        output_path,
        dpi=IMAGE_DPI,
        facecolor=figure.get_facecolor(),
        bbox_inches=None,
    )
    plt.close(figure)

    return BenchmarkReceiptArtifact(
        image_file=output_path
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a terminal-style receipt "
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
    """Run the benchmark-receipt generator as a module."""

    arguments = _build_parser().parse_args()

    artifact = create_benchmark_receipt(
        arguments.run_directory
    )

    print(
        f"Benchmark receipt: {artifact.image_file}"
    )


if __name__ == "__main__":
    main()
