"""Command-line interface for executing the benchmark workflow."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .analysis import analyze_benchmark_results
from .benchmark import (
    BenchmarkCase,
    create_benchmark_plan,
    run_benchmark,
)
from .config import BenchmarkConfig, load_config
from .environment import collect_environment_metadata
from .plotting import create_benchmark_plots
from .results import save_benchmark_artifacts


@dataclass(frozen=True, slots=True)
class CommandArguments:
    """Validated command-line path and execution options."""

    config_path: Path
    data_directory: Path
    output_directory: Path
    plan_only: bool


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute the command-line benchmark workflow."""

    try:
        command_arguments = _parse_arguments(arguments)
        config = load_config(command_arguments.config_path)
        plan = create_benchmark_plan(config)

        _print_plan_summary(config, plan)

        if command_arguments.plan_only:
            _print_plan(plan)
            return 0

        print("Collecting environment metadata...", flush=True)
        environment = collect_environment_metadata()

        print("Starting benchmark execution...", flush=True)
        results = run_benchmark(
            config,
            data_directory=command_arguments.data_directory,
            progress_callback=_report_progress,
        )

        print("Analyzing benchmark results...", flush=True)
        analysis = analyze_benchmark_results(results)

        print("Saving benchmark artifacts...", flush=True)
        artifacts = save_benchmark_artifacts(
            results,
            config=config,
            environment=environment,
            output_root=command_arguments.output_directory,
        )

        plot_artifact = create_benchmark_plots(
            analysis,
            output_directory=artifacts.run_directory,
        )

        print()
        print("Benchmark completed successfully.")
        print(f"Run directory: {artifacts.run_directory}")
        print(f"Results: {artifacts.results_file}")
        print(f"Configuration: {artifacts.configuration_file}")
        print(f"Environment: {artifacts.environment_file}")
        print(f"Plot: {plot_artifact.comparison_plot}")

        return 0

    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _parse_arguments(
    arguments: Sequence[str] | None,
) -> CommandArguments:
    """Parse command-line options into concrete types."""

    parser = argparse.ArgumentParser(
        prog="pytorch-cuda-benchmark",
        description=(
            "Compare CIFAR-10 ResNet-18 training performance "
            "between CPU and CUDA devices."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark.yaml"),
        help="Path to the benchmark YAML configuration.",
    )
    parser.add_argument(
        "--data-directory",
        "--data-dir",
        dest="data_directory",
        type=Path,
        default=Path("data"),
        help="Directory used to store or load CIFAR-10.",
    )
    parser.add_argument(
        "--output-directory",
        "--output-dir",
        dest="output_directory",
        type=Path,
        default=Path("results"),
        help="Root directory for generated benchmark artifacts.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Display the benchmark matrix without training.",
    )

    namespace = parser.parse_args(arguments)

    return CommandArguments(
        config_path=cast(Path, namespace.config),
        data_directory=cast(Path, namespace.data_directory),
        output_directory=cast(Path, namespace.output_directory),
        plan_only=cast(bool, namespace.plan_only),
    )


def _print_plan_summary(
    config: BenchmarkConfig,
    plan: Sequence[BenchmarkCase],
) -> None:
    """Display the configured benchmark workload."""

    devices = ", ".join(config.devices)
    batch_sizes = ", ".join(
        str(batch_size) for batch_size in config.batch_sizes
    )

    print("Benchmark plan")
    print(f"  Devices: {devices}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Epochs per run: {config.epochs}")
    print(f"  Repetitions: {config.repetitions}")
    print(f"  Warm-up batches: {config.warmup_batches}")
    print(f"  Total runs: {len(plan)}")
    print()


def _print_plan(plan: Sequence[BenchmarkCase]) -> None:
    """Display every planned benchmark case."""

    total_runs = len(plan)

    for run_number, case in enumerate(plan, start=1):
        _report_progress(run_number, total_runs, case)


def _report_progress(
    run_number: int,
    total_runs: int,
    case: BenchmarkCase,
) -> None:
    """Display the benchmark case currently being executed."""

    print(
        f"[{run_number}/{total_runs}] "
        f"device={case.device} "
        f"batch_size={case.batch_size} "
        f"repetition={case.repetition} "
        f"seed={case.seed}",
        flush=True,
    )
