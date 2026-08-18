"""Benchmark result persistence utilities."""

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import BenchmarkResult
from .config import BenchmarkConfig
from .environment import EnvironmentMetadata

_RESULT_FIELD_NAMES: tuple[str, ...] = (
    "device",
    "batch_size",
    "repetition",
    "seed",
    "mixed_precision",
    "epochs",
    "processed_batches",
    "processed_samples",
    "mean_loss",
    "elapsed_seconds",
    "samples_per_second",
)


@dataclass(frozen=True, slots=True)
class SavedBenchmarkArtifacts:
    """Paths to the files produced by a benchmark execution."""

    run_directory: Path
    results_file: Path
    configuration_file: Path
    environment_file: Path


def save_benchmark_artifacts(
    results: Sequence[BenchmarkResult],
    *,
    config: BenchmarkConfig,
    environment: EnvironmentMetadata,
    output_root: str | Path = "results",
) -> SavedBenchmarkArtifacts:
    """Save results and reproducibility metadata for one benchmark."""

    if not results:
        raise ValueError("At least one benchmark result is required.")

    run_directory = _create_run_directory(output_root)

    results_file = run_directory / "benchmark_results.csv"
    configuration_file = run_directory / "configuration.json"
    environment_file = run_directory / "environment.json"

    _write_results_csv(results_file, results)
    _write_json(configuration_file, asdict(config))
    _write_json(environment_file, asdict(environment))

    return SavedBenchmarkArtifacts(
        run_directory=run_directory,
        results_file=results_file,
        configuration_file=configuration_file,
        environment_file=environment_file,
    )


def _create_run_directory(output_root: str | Path) -> Path:
    """Create a unique timestamped directory for one benchmark."""

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    run_directory = (
        Path(output_root).expanduser().resolve() / f"run-{timestamp}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    return run_directory


def _write_results_csv(
    path: Path,
    results: Sequence[BenchmarkResult],
) -> None:
    """Write individual benchmark measurements to CSV."""

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(_RESULT_FIELD_NAMES),
        )
        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))


def _write_json(path: Path, value: object) -> None:
    """Write a JSON document with stable human-readable formatting."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        file.write("\n")
