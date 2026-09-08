"""Command-line interface for benchmark execution and saved results."""

from __future__ import annotations

import argparse
import math
import sys
import webbrowser
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TypeAlias, cast

from .analysis import BenchmarkAnalysis, analyze_benchmark_results
from .benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    create_benchmark_plan,
    run_benchmark,
)
from .config import BenchmarkConfig, load_config
from .data import BenchmarkDevice
from .environment import EnvironmentMetadata, collect_environment_metadata
from .plotting import create_benchmark_plots
from .results import save_benchmark_artifacts
from .saved_runs import (
    SavedBenchmarkRun,
    calculate_device_totals,
    inspect_saved_runs,
    load_latest_complete_run,
    load_saved_benchmark_run,
    results_for_device,
)
from .social_assets import SocialAssetBundle, create_social_assets


@dataclass(frozen=True, slots=True)
class PlanCommand:
    """Options for displaying the configured run matrix."""

    config_path: Path


@dataclass(frozen=True, slots=True)
class RunCommand:
    """Options for executing and saving a new benchmark."""

    config_path: Path
    data_directory: Path
    output_directory: Path


@dataclass(frozen=True, slots=True)
class ListCommand:
    """Options for listing saved benchmark runs."""

    output_directory: Path


@dataclass(frozen=True, slots=True)
class ShowCommand:
    """Options for inspecting one complete saved run."""

    output_directory: Path
    run_directory: Path | None
    open_images: bool


CommandArguments: TypeAlias = (
    PlanCommand | RunCommand | ListCommand | ShowCommand
)


@dataclass(frozen=True, slots=True)
class GeneratedOutputs:
    """Technical and social images generated for one saved run."""

    technical_plot: Path
    social_assets: SocialAssetBundle


class _RunReporter:
    """Print progress outside measured training intervals."""

    def __init__(
        self,
        *,
        config: BenchmarkConfig,
        plan: Sequence[BenchmarkCase],
        environment: Mapping[str, object],
    ) -> None:
        self._config = config
        self._environment = environment
        self._expected_by_device: Counter[BenchmarkDevice] = Counter(
            case.device for case in plan
        )
        self._completed_by_device: defaultdict[
            BenchmarkDevice,
            list[BenchmarkResult],
        ] = defaultdict(list)
        self._active_device: BenchmarkDevice | None = None

    def report_start(
        self,
        run_number: int,
        total_runs: int,
        case: BenchmarkCase,
    ) -> None:
        """Print the case that is about to start."""

        if case.device != self._active_device:
            self._active_device = case.device
            _print_device_heading(case.device, self._environment)

        completed_count = len(self._completed_by_device[case.device])
        device_run_number = completed_count + 1
        device_total = self._expected_by_device[case.device]

        print(
            f"[{run_number:02d}/{total_runs:02d} | "
            f"{_display_device(case.device)} "
            f"{device_run_number:02d}/{device_total:02d}]"
        )
        print(
            f"  Batch size: {case.batch_size} | "
            f"Repetition: {case.repetition}/{self._config.repetitions} | "
            f"Seed: {case.seed}"
        )
        print("  Status: running", flush=True)

    def report_result(
        self,
        _run_number: int,
        _total_runs: int,
        case: BenchmarkCase,
        result: BenchmarkResult,
    ) -> None:
        """Print a result after its training timer has stopped."""

        completed_results = self._completed_by_device[case.device]
        completed_results.append(result)

        print("  Status: completed")
        _print_measurement(result, indentation="  ")
        print()

        if len(completed_results) == self._expected_by_device[case.device]:
            _print_device_summary(case.device, completed_results)


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute the requested command."""

    try:
        command = _parse_arguments(arguments)

        if isinstance(command, PlanCommand):
            return _run_plan_command(command)

        if isinstance(command, RunCommand):
            return _run_benchmark_command(command)

        if isinstance(command, ListCommand):
            return _run_list_command(command)

        return _run_show_command(command)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _parse_arguments(
    arguments: Sequence[str] | None,
) -> CommandArguments:
    parser = argparse.ArgumentParser(
        prog="pytorch-cuda-benchmark",
        description=(
            "Run, inspect, and present reproducible CIFAR-10 "
            "ResNet-18 CPU and CUDA benchmarks."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Display the configured benchmark matrix without training.",
    )
    _add_config_argument(plan_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute and save a new benchmark run.",
    )
    _add_config_argument(run_parser)
    run_parser.add_argument(
        "--data-directory",
        "--data-dir",
        dest="data_directory",
        type=Path,
        default=Path("data"),
        help="Directory used to store or load CIFAR-10.",
    )
    _add_output_argument(run_parser)

    list_parser = subparsers.add_parser(
        "list",
        help="List saved benchmark runs from newest to oldest.",
    )
    _add_output_argument(list_parser)

    show_parser = subparsers.add_parser(
        "show",
        help="Inspect a complete saved run and regenerate its images.",
    )
    show_parser.add_argument(
        "run_directory",
        nargs="?",
        type=Path,
        help=(
            "Specific run directory. If omitted, the newest complete "
            "run is selected."
        ),
    )
    _add_output_argument(show_parser)
    show_parser.add_argument(
        "--open",
        dest="open_images",
        action="store_true",
        help="Open the three generated social images.",
    )

    namespace = parser.parse_args(arguments)
    command_name = cast(str, namespace.command)

    if command_name == "plan":
        return PlanCommand(
            config_path=cast(Path, namespace.config_path),
        )

    if command_name == "run":
        return RunCommand(
            config_path=cast(Path, namespace.config_path),
            data_directory=cast(Path, namespace.data_directory),
            output_directory=cast(Path, namespace.output_directory),
        )

    if command_name == "list":
        return ListCommand(
            output_directory=cast(Path, namespace.output_directory),
        )

    return ShowCommand(
        output_directory=cast(Path, namespace.output_directory),
        run_directory=cast(Path | None, namespace.run_directory),
        open_images=cast(bool, namespace.open_images),
    )


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=Path("configs/benchmark.yaml"),
        help="Path to the benchmark YAML configuration.",
    )


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-directory",
        "--output-dir",
        dest="output_directory",
        type=Path,
        default=Path("results"),
        help="Root directory containing timestamped benchmark runs.",
    )


def _run_plan_command(command: PlanCommand) -> int:
    config = load_config(command.config_path)
    plan = create_benchmark_plan(config)

    _print_heading("BENCHMARK PLAN")
    _print_plan_summary(config, total_runs=len(plan))
    _print_plan_cases(plan, config=config)

    return 0


def _run_benchmark_command(command: RunCommand) -> int:
    config = load_config(command.config_path)
    _require_cpu_cuda_comparison(config)
    plan = create_benchmark_plan(config)

    _print_heading("NEW BENCHMARK RUN")
    _print_plan_summary(config, total_runs=len(plan))

    print("Collecting environment metadata...", flush=True)
    environment = collect_environment_metadata()
    reporter = _RunReporter(
        config=config,
        plan=plan,
        environment=_environment_mapping(environment),
    )

    print("\nStarting benchmark execution.")
    execution_started = perf_counter()

    results = run_benchmark(
        config,
        data_directory=command.data_directory,
        progress_callback=reporter.report_start,
        result_callback=reporter.report_result,
    )

    execution_wall_seconds = perf_counter() - execution_started

    print("Analyzing benchmark results...", flush=True)
    analysis = analyze_benchmark_results(results)

    print("Saving benchmark artifacts...", flush=True)
    artifacts = save_benchmark_artifacts(
        results,
        config=config,
        environment=environment,
        output_root=command.output_directory,
    )

    print("Generating benchmark images...", flush=True)
    outputs = _generate_outputs(
        analysis,
        run_directory=artifacts.run_directory,
    )

    totals = calculate_device_totals(results)
    combined_seconds = math.fsum(totals.values())

    _print_heading("BENCHMARK COMPLETED")
    print(
        "CPU measured total:  "
        f"{_format_total_duration(totals['cpu'])}"
    )
    print(
        "CUDA measured total: "
        f"{_format_total_duration(totals['cuda'])}"
    )
    print(
        "Combined measured intervals: "
        f"{_format_total_duration(combined_seconds)}"
    )
    print(
        "Benchmark execution wall-clock: "
        f"{_format_total_duration(execution_wall_seconds)}"
    )
    print(
        "  Includes case setup and warm-up; measured totals contain only "
        "timed training intervals."
    )
    print()
    print(f"Run directory: {artifacts.run_directory}")
    print(f"Results: {artifacts.results_file}")
    print(f"Configuration: {artifacts.configuration_file}")
    print(f"Environment: {artifacts.environment_file}")
    _print_generated_outputs(outputs)

    return 0


def _run_list_command(command: ListCommand) -> int:
    inspections = inspect_saved_runs(command.output_directory)

    _print_heading("SAVED BENCHMARK RUNS")

    if not inspections:
        print("No saved benchmark runs were found.")
        print("Create one with: python -m pytorch_cuda_benchmark run")
        return 0

    for index, inspection in enumerate(inspections, start=1):
        print(
            f"{index}. {inspection.run_directory.name} "
            f"[{inspection.status.upper()}]"
        )

        if inspection.status == "complete":
            print(
                f"   Runs: {inspection.completed_runs}/"
                f"{inspection.expected_runs}"
            )
            print(
                "   CPU measured total: "
                f"{_format_optional_total(inspection.cpu_total_seconds)}"
            )
            print(
                "   CUDA measured total: "
                f"{_format_optional_total(inspection.cuda_total_seconds)}"
            )
        elif inspection.detail is not None:
            print(f"   Reason: {inspection.detail}")

        print(f"   Directory: {inspection.run_directory}")
        print()

    return 0


def _run_show_command(command: ShowCommand) -> int:
    if command.run_directory is None:
        saved_run = load_latest_complete_run(command.output_directory)
    else:
        saved_run = load_saved_benchmark_run(command.run_directory)

    _require_cpu_cuda_comparison(saved_run.config)
    _print_saved_run(saved_run)

    print("Regenerating benchmark images...", flush=True)
    analysis = analyze_benchmark_results(saved_run.results)
    outputs = _generate_outputs(
        analysis,
        run_directory=saved_run.run_directory,
    )

    _print_generated_outputs(outputs)

    if command.open_images:
        _open_social_images(outputs.social_assets)

    return 0


def _generate_outputs(
    analysis: BenchmarkAnalysis,
    *,
    run_directory: Path,
) -> GeneratedOutputs:
    technical_plot = create_benchmark_plots(
        analysis,
        output_directory=run_directory,
    )
    social_assets = create_social_assets(run_directory)

    return GeneratedOutputs(
        technical_plot=technical_plot.comparison_plot,
        social_assets=social_assets,
    )


def _print_saved_run(saved_run: SavedBenchmarkRun) -> None:
    _print_heading("SAVED BENCHMARK RESULTS")
    print(f"Run: {saved_run.run_directory.name}")
    print(f"Directory: {saved_run.run_directory}")
    print("Status: complete and matrix-validated")
    print()

    _print_plan_summary(
        saved_run.config,
        total_runs=saved_run.expected_run_count,
    )

    total_results = len(saved_run.results)
    global_run_number = 0

    for device_name in saved_run.config.devices:
        device = cast(BenchmarkDevice, device_name)
        device_results = results_for_device(
            saved_run.results,
            device,
        )

        _print_device_heading(
            device,
            saved_run.environment,
        )

        for device_run_number, result in enumerate(
            device_results,
            start=1,
        ):
            global_run_number += 1

            print(
                f"[{global_run_number:02d}/{total_results:02d} | "
                f"{_display_device(device)} "
                f"{device_run_number:02d}/{len(device_results):02d}]"
            )
            print(
                f"  Batch size: {result.batch_size} | "
                f"Repetition: {result.repetition}/"
                f"{saved_run.config.repetitions} | "
                f"Seed: {result.seed}"
            )
            _print_measurement(
                result,
                indentation="  ",
            )
            print()

        _print_device_summary(
            device,
            device_results,
        )

    totals = calculate_device_totals(saved_run.results)
    combined_seconds = math.fsum(totals.values())

    print("Complete measured matrix")
    print(
        f"  Runs: {len(saved_run.results)}/"
        f"{saved_run.expected_run_count}"
    )
    print(
        "  CPU total: "
        f"{_format_total_duration(totals['cpu'])}"
    )
    print(
        "  CUDA total: "
        f"{_format_total_duration(totals['cuda'])}"
    )
    print(
        "  Combined measured intervals: "
        f"{_format_total_duration(combined_seconds)}"
    )
    print()


def _print_plan_summary(
    config: BenchmarkConfig,
    *,
    total_runs: int,
) -> None:
    devices = ", ".join(
        _display_device(
            cast(BenchmarkDevice, device)
        )
        for device in config.devices
    )
    batch_sizes = ", ".join(
        map(str, config.batch_sizes)
    )
    mixed_precision = (
        "enabled"
        if config.mixed_precision
        else "disabled"
    )

    print(f"Devices: {devices}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Epochs per run: {config.epochs}")
    print(
        "Repetitions per device and batch size: "
        f"{config.repetitions}"
    )
    print(
        "Warm-up batches per run: "
        f"{config.warmup_batches}"
    )
    print(
        "CUDA automatic mixed precision: "
        f"{mixed_precision}"
    )
    print(f"Total planned runs: {total_runs}")
    print()


def _print_plan_cases(
    plan: Sequence[BenchmarkCase],
    *,
    config: BenchmarkConfig,
) -> None:
    total_runs = len(plan)

    for run_number, case in enumerate(
        plan,
        start=1,
    ):
        print(
            f"[{run_number:02d}/{total_runs:02d}] "
            f"{_display_device(case.device)} | "
            f"batch {case.batch_size} | "
            f"repetition {case.repetition}/"
            f"{config.repetitions} | "
            f"seed {case.seed}"
        )


def _print_measurement(
    result: BenchmarkResult,
    *,
    indentation: str,
) -> None:
    mean_batch_seconds = (
        result.elapsed_seconds
        / result.processed_batches
    )

    print(
        f"{indentation}Measured training time: "
        f"{_format_run_duration(result.elapsed_seconds)}"
    )
    print(
        f"{indentation}Mean timed batch duration: "
        f"{_format_batch_duration(mean_batch_seconds)}"
    )
    print(
        f"{indentation}Throughput: "
        f"{result.samples_per_second:,.2f} samples/sec"
    )
    print(
        f"{indentation}Mean loss: "
        f"{result.mean_loss:.4f}"
    )
    print(
        f"{indentation}Timed batches: "
        f"{result.processed_batches:,}"
    )
    print(
        f"{indentation}Processed samples: "
        f"{result.processed_samples:,}"
    )


def _print_device_summary(
    device: BenchmarkDevice,
    results: Sequence[BenchmarkResult],
) -> None:
    total_seconds = math.fsum(
        result.elapsed_seconds
        for result in results
    )
    total_samples = sum(
        result.processed_samples
        for result in results
    )

    print(f"{_display_device(device)} summary")
    print(f"  Completed runs: {len(results)}")
    print(
        "  Total measured training time: "
        f"{_format_total_duration(total_seconds)}"
    )
    print(
        "  Total processed samples: "
        f"{total_samples:,}"
    )
    print()


def _print_device_heading(
    device: BenchmarkDevice,
    environment: Mapping[str, object],
) -> None:
    name = (
        _cpu_name(environment)
        if device == "cpu"
        else _gpu_name(environment)
    )

    print()
    print("=" * 72)
    print(_display_device(device))
    print(name)
    print("=" * 72)
    print()


def _cpu_name(
    environment: Mapping[str, object],
) -> str:
    for key in (
        "cpu_model",
        "processor_name",
        "processor_identifier",
    ):
        value = environment.get(key)

        if isinstance(value, str) and value.strip():
            return " ".join(value.split())

    return "Unknown CPU"


def _gpu_name(
    environment: Mapping[str, object],
) -> str:
    value = environment.get("gpus")

    if not isinstance(value, (list, tuple)) or not value:
        return "Unknown GPU"

    gpu_items = cast(Sequence[object], value)
    first_gpu = gpu_items[0]

    if not isinstance(first_gpu, Mapping):
        return "Unknown GPU"

    gpu = cast(Mapping[str, object], first_gpu)
    name = gpu.get("name")

    if isinstance(name, str) and name.strip():
        return " ".join(name.split())

    return "Unknown GPU"


def _environment_mapping(
    environment: EnvironmentMetadata,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        asdict(environment),
    )


def _require_cpu_cuda_comparison(
    config: BenchmarkConfig,
) -> None:
    if set(config.devices) != {"cpu", "cuda"}:
        raise ValueError(
            "The reporting workflow requires exactly the 'cpu' and "
            "'cuda' devices so the comparison images can be generated."
        )


def _display_device(
    device: BenchmarkDevice,
) -> str:
    return (
        "CPU"
        if device == "cpu"
        else "CUDA"
    )


def _format_run_duration(
    seconds: float,
) -> str:
    if seconds >= 3600.0:
        hours = int(
            seconds // 3600.0
        )
        remainder = (
            seconds
            - hours * 3600.0
        )
        minutes = int(
            remainder // 60.0
        )
        remaining_seconds = (
            remainder
            - minutes * 60.0
        )

        return (
            f"{hours} hr "
            f"{minutes:02d} min "
            f"{remaining_seconds:05.2f} sec"
        )

    if seconds >= 60.0:
        minutes = int(
            seconds // 60.0
        )
        remaining_seconds = (
            seconds
            - minutes * 60.0
        )

        return (
            f"{minutes} min "
            f"{remaining_seconds:05.2f} sec"
        )

    return f"{seconds:.2f} sec"


def _format_total_duration(
    seconds: float,
) -> str:
    rounded_seconds = max(
        0,
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


def _format_batch_duration(
    seconds: float,
) -> str:
    if seconds < 1.0:
        return (
            f"{seconds * 1000.0:.2f} ms"
        )

    return f"{seconds:.3f} sec"


def _format_optional_total(
    seconds: float | None,
) -> str:
    if seconds is None:
        return "unavailable"

    return _format_total_duration(
        seconds
    )


def _print_generated_outputs(
    outputs: GeneratedOutputs,
) -> None:
    print(
        "Technical plot: "
        f"{outputs.technical_plot}"
    )
    print(
        "Social card: "
        f"{outputs.social_assets.social_card}"
    )
    print(
        "Speedup graph: "
        f"{outputs.social_assets.speedup_graph}"
    )
    print(
        "Benchmark receipt: "
        f"{outputs.social_assets.benchmark_receipt}"
    )


def _open_social_images(
    assets: SocialAssetBundle,
) -> None:
    print("Opening social images...")
    failed: list[Path] = []

    for image_path in assets.image_files:
        opened = webbrowser.open(
            image_path.resolve().as_uri(),
            new=2,
        )

        if not opened:
            failed.append(image_path)

    if failed:
        names = ", ".join(
            path.name
            for path in failed
        )
        print(
            "Warning: the operating system "
            f"did not open: {names}",
            file=sys.stderr,
        )


def _print_heading(
    title: str,
) -> None:
    print()
    print(title)
    print("=" * len(title))


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
