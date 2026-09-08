# pyright: reportUnknownMemberType=false

"""Discover, load, and validate saved benchmark runs."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .benchmark import BenchmarkResult
from .config import BenchmarkConfig, load_config
from .data import BenchmarkDevice

RESULTS_FILE_NAME = "benchmark_results.csv"
CONFIGURATION_FILE_NAME = "configuration.json"
ENVIRONMENT_FILE_NAME = "environment.json"

SavedRunStatus = Literal[
    "complete",
    "incomplete",
    "invalid",
]


class IncompleteSavedRunError(ValueError):
    """Raised when a saved run does not contain its complete matrix."""


@dataclass(frozen=True, slots=True)
class SavedBenchmarkRun:
    """Validated artifacts and measurements for one saved run."""

    run_directory: Path
    results: tuple[BenchmarkResult, ...]
    config: BenchmarkConfig
    environment: dict[str, object]

    @property
    def expected_run_count(self) -> int:
        """Return the number of runs required by the saved configuration."""

        return (
            len(self.config.devices)
            * len(self.config.batch_sizes)
            * self.config.repetitions
        )


@dataclass(frozen=True, slots=True)
class SavedRunInspection:
    """Display-safe status information for one run directory."""

    run_directory: Path
    status: SavedRunStatus
    completed_runs: int | None
    expected_runs: int | None
    cpu_total_seconds: float | None
    cuda_total_seconds: float | None
    detail: str | None


def discover_saved_run_directories(
    output_root: str | Path = "results",
) -> tuple[Path, ...]:
    """Return timestamped run directories from newest to oldest."""

    root = Path(
        output_root
    ).expanduser().resolve()

    if not root.exists():
        return ()

    if not root.is_dir():
        raise NotADirectoryError(
            "Benchmark output path is not a directory: "
            f"{root}"
        )

    run_directories = [
        path
        for path in root.iterdir()
        if (
            path.is_dir()
            and path.name.startswith("run-")
        )
    ]

    return tuple(
        sorted(
            run_directories,
            key=lambda path: path.name,
            reverse=True,
        )
    )


def inspect_saved_runs(
    output_root: str | Path = "results",
) -> tuple[SavedRunInspection, ...]:
    """Inspect every discovered run without stopping on invalid entries."""

    return tuple(
        _inspect_saved_run(
            run_directory
        )
        for run_directory in discover_saved_run_directories(
            output_root
        )
    )


def load_latest_complete_run(
    output_root: str | Path = "results",
) -> SavedBenchmarkRun:
    """Load the newest saved run that passes complete-matrix validation."""

    run_directories = (
        discover_saved_run_directories(
            output_root
        )
    )

    if not run_directories:
        raise FileNotFoundError(
            "No saved benchmark runs were found. "
            "Run a benchmark first."
        )

    rejected: list[str] = []

    for run_directory in run_directories:
        try:
            return load_saved_benchmark_run(
                run_directory
            )
        except (
            IncompleteSavedRunError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            rejected.append(
                f"{run_directory.name}: {error}"
            )

    detail = "\n".join(
        f"  {item}"
        for item in rejected
    )

    raise FileNotFoundError(
        "No complete saved benchmark run was found.\n"
        f"Rejected runs:\n{detail}"
    )


def load_saved_benchmark_run(
    run_directory: str | Path,
) -> SavedBenchmarkRun:
    """Load and validate one saved benchmark run."""

    directory = Path(
        run_directory
    ).expanduser().resolve()

    if not directory.is_dir():
        raise FileNotFoundError(
            "Saved benchmark directory was not found: "
            f"{directory}"
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

    missing_files = [
        path.name
        for path in (
            results_path,
            configuration_path,
            environment_path,
        )
        if not path.is_file()
    ]

    if missing_files:
        names = ", ".join(
            missing_files
        )

        raise FileNotFoundError(
            "Saved run is missing required files: "
            f"{names}."
        )

    # JSON is valid YAML, so the existing validated configuration
    # loader can also load the saved configuration.json file.
    config = load_config(
        configuration_path
    )
    environment = _read_json_object(
        environment_path
    )
    results = _read_results(
        results_path
    )

    _validate_complete_matrix(
        results,
        config=config,
    )

    return SavedBenchmarkRun(
        run_directory=directory,
        results=results,
        config=config,
        environment=environment,
    )


def calculate_device_totals(
    results: Sequence[BenchmarkResult],
) -> dict[BenchmarkDevice, float]:
    """Sum saved measured training intervals for each device."""

    return {
        device: math.fsum(
            result.elapsed_seconds
            for result in results
            if result.device == device
        )
        for device in (
            "cpu",
            "cuda",
        )
    }


def results_for_device(
    results: Sequence[BenchmarkResult],
    device: BenchmarkDevice,
) -> tuple[BenchmarkResult, ...]:
    """Return measurements for one device in their saved order."""

    return tuple(
        result
        for result in results
        if result.device == device
    )


def _inspect_saved_run(
    run_directory: Path,
) -> SavedRunInspection:
    try:
        saved_run = (
            load_saved_benchmark_run(
                run_directory
            )
        )
    except (
        FileNotFoundError,
        IncompleteSavedRunError,
    ) as error:
        return SavedRunInspection(
            run_directory=run_directory,
            status="incomplete",
            completed_runs=None,
            expected_runs=None,
            cpu_total_seconds=None,
            cuda_total_seconds=None,
            detail=str(error),
        )
    except (
        OSError,
        TypeError,
        ValueError,
    ) as error:
        return SavedRunInspection(
            run_directory=run_directory,
            status="invalid",
            completed_runs=None,
            expected_runs=None,
            cpu_total_seconds=None,
            cuda_total_seconds=None,
            detail=str(error),
        )

    totals = calculate_device_totals(
        saved_run.results
    )

    return SavedRunInspection(
        run_directory=run_directory,
        status="complete",
        completed_runs=len(
            saved_run.results
        ),
        expected_runs=(
            saved_run.expected_run_count
        ),
        cpu_total_seconds=totals["cpu"],
        cuda_total_seconds=totals["cuda"],
        detail=None,
    )


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

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"Expected a JSON object in: {path}"
        )

    raw_mapping = cast(
        dict[object, object],
        value,
    )
    result: dict[str, object] = {}

    for key, item in raw_mapping.items():
        if not isinstance(
            key,
            str,
        ):
            raise ValueError(
                "JSON keys must be strings in: "
                f"{path}"
            )

        result[key] = item

    return result


def _read_results(
    path: Path,
) -> tuple[BenchmarkResult, ...]:
    results: list[
        BenchmarkResult
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row_number, raw_row in enumerate(
            reader,
            start=2,
        ):
            try:
                results.append(
                    _parse_result_row(
                        raw_row
                    )
                )
            except (
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "Invalid benchmark result "
                    f"on CSV row {row_number}: "
                    f"{error}"
                ) from error

    if not results:
        raise IncompleteSavedRunError(
            "benchmark_results.csv "
            "contains no result rows."
        )

    return tuple(
        results
    )


def _parse_result_row(
    row: Mapping[
        str,
        str | None,
    ],
) -> BenchmarkResult:
    device_text = _required_text(
        row,
        "device",
    )

    if device_text not in {
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "Unsupported device: "
            f"{device_text!r}."
        )

    device = cast(
        BenchmarkDevice,
        device_text,
    )

    return BenchmarkResult(
        device=device,
        batch_size=_positive_integer(
            row,
            "batch_size",
        ),
        repetition=_positive_integer(
            row,
            "repetition",
        ),
        seed=_nonnegative_integer(
            row,
            "seed",
        ),
        mixed_precision=_boolean(
            row,
            "mixed_precision",
        ),
        epochs=_positive_integer(
            row,
            "epochs",
        ),
        processed_batches=_positive_integer(
            row,
            "processed_batches",
        ),
        processed_samples=_positive_integer(
            row,
            "processed_samples",
        ),
        mean_loss=_finite_float(
            row,
            "mean_loss",
            minimum=0.0,
        ),
        elapsed_seconds=_finite_float(
            row,
            "elapsed_seconds",
            minimum=0.0,
            minimum_is_exclusive=True,
        ),
        samples_per_second=_finite_float(
            row,
            "samples_per_second",
            minimum=0.0,
            minimum_is_exclusive=True,
        ),
    )


def _required_text(
    row: Mapping[
        str,
        str | None,
    ],
    field_name: str,
) -> str:
    value = row.get(
        field_name
    )

    if (
        value is None
        or not value.strip()
    ):
        raise ValueError(
            f"Missing '{field_name}'."
        )

    return value.strip()


def _positive_integer(
    row: Mapping[
        str,
        str | None,
    ],
    field_name: str,
) -> int:
    value = _integer(
        row,
        field_name,
    )

    if value <= 0:
        raise ValueError(
            f"'{field_name}' must be "
            "greater than zero."
        )

    return value


def _nonnegative_integer(
    row: Mapping[
        str,
        str | None,
    ],
    field_name: str,
) -> int:
    value = _integer(
        row,
        field_name,
    )

    if value < 0:
        raise ValueError(
            f"'{field_name}' must not "
            "be negative."
        )

    return value


def _integer(
    row: Mapping[
        str,
        str | None,
    ],
    field_name: str,
) -> int:
    text = _required_text(
        row,
        field_name,
    )

    try:
        return int(
            text
        )
    except ValueError as error:
        raise ValueError(
            f"'{field_name}' must be "
            "an integer."
        ) from error


def _boolean(
    row: Mapping[
        str,
        str | None,
    ],
    field_name: str,
) -> bool:
    text = _required_text(
        row,
        field_name,
    ).casefold()

    if text == "true":
        return True

    if text == "false":
        return False

    raise ValueError(
        f"'{field_name}' must be "
        "true or false."
    )


def _finite_float(
    row: Mapping[
        str,
        str | None,
    ],
    field_name: str,
    *,
    minimum: float,
    minimum_is_exclusive: bool = False,
) -> float:
    text = _required_text(
        row,
        field_name,
    )

    try:
        value = float(
            text
        )
    except ValueError as error:
        raise ValueError(
            f"'{field_name}' must be numeric."
        ) from error

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"'{field_name}' must be finite."
        )

    below_minimum = (
        value <= minimum
        if minimum_is_exclusive
        else value < minimum
    )

    if below_minimum:
        comparison = (
            "greater than"
            if minimum_is_exclusive
            else "at least"
        )

        raise ValueError(
            f"'{field_name}' must be "
            f"{comparison} {minimum}."
        )

    return value


def _validate_complete_matrix(
    results: Sequence[
        BenchmarkResult
    ],
    *,
    config: BenchmarkConfig,
) -> None:
    expected_cases = {
        (
            device,
            batch_size,
            repetition,
        )
        for device in config.devices
        for batch_size in config.batch_sizes
        for repetition in range(
            1,
            config.repetitions + 1,
        )
    }

    actual_counts = Counter(
        (
            result.device,
            result.batch_size,
            result.repetition,
        )
        for result in results
    )

    unexpected_cases = (
        set(actual_counts)
        - expected_cases
    )
    duplicate_cases = {
        key
        for key, count
        in actual_counts.items()
        if count > 1
    }

    if unexpected_cases:
        raise ValueError(
            "Saved results contain cases "
            "outside the configured matrix."
        )

    if duplicate_cases:
        raise ValueError(
            "Saved results contain duplicate "
            "device, batch-size, and repetition "
            "combinations."
        )

    missing_cases = (
        expected_cases
        - set(actual_counts)
    )

    if missing_cases:
        raise IncompleteSavedRunError(
            "Saved result matrix is incomplete: "
            f"{len(results)} of "
            f"{len(expected_cases)} "
            "cases are present."
        )

    for result in results:
        expected_seed = (
            config.seed
            + result.repetition
            - 1
        )
        expected_mixed_precision = (
            config.mixed_precision
            and result.device == "cuda"
        )

        if result.seed != expected_seed:
            raise ValueError(
                "Saved result seed does not "
                "match the configured "
                "repetition seed."
            )

        if result.epochs != config.epochs:
            raise ValueError(
                "Saved result epochs do not "
                "match the saved configuration."
            )

        if (
            result.mixed_precision
            != expected_mixed_precision
        ):
            raise ValueError(
                "Saved mixed-precision state "
                "does not match the device and "
                "saved configuration."
            )
