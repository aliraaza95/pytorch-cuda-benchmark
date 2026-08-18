# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

"""Benchmark performance visualization utilities."""

from dataclasses import dataclass
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .analysis import BenchmarkAnalysis, DevicePerformanceSummary

_CPU_COLOR = "#4C78A8"
_CUDA_COLOR = "#F58518"
_SPEEDUP_COLOR = "#54A24B"


@dataclass(frozen=True, slots=True)
class BenchmarkPlotArtifact:
    """Path to a generated benchmark visualization."""

    comparison_plot: Path


def create_benchmark_plots(
    analysis: BenchmarkAnalysis,
    *,
    output_directory: str | Path,
) -> BenchmarkPlotArtifact:
    """Create timing, throughput, and speedup comparison plots."""

    cpu_summaries = _summaries_for_device(analysis, "cpu")
    cuda_summaries = _summaries_for_device(analysis, "cuda")
    speedups = tuple(
        sorted(
            analysis.speedups,
            key=lambda speedup: speedup.batch_size,
        )
    )

    _validate_analysis(
        cpu_summaries=cpu_summaries,
        cuda_summaries=cuda_summaries,
        speedup_batch_sizes=tuple(
            speedup.batch_size for speedup in speedups
        ),
    )

    output_path = Path(output_directory).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_plot = output_path / "benchmark_comparison.png"

    figure = Figure(figsize=(15.0, 4.8))
    FigureCanvasAgg(figure)

    time_axis = figure.add_subplot(1, 3, 1)
    throughput_axis = figure.add_subplot(1, 3, 2)
    speedup_axis = figure.add_subplot(1, 3, 3)

    _plot_training_time(
        time_axis,
        cpu_summaries=cpu_summaries,
        cuda_summaries=cuda_summaries,
    )
    _plot_throughput(
        throughput_axis,
        cpu_summaries=cpu_summaries,
        cuda_summaries=cuda_summaries,
    )
    _plot_speedup(speedup_axis, analysis)

    figure.suptitle(
        "PyTorch CIFAR-10 Training Performance",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure.savefig(
        comparison_plot,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    return BenchmarkPlotArtifact(
        comparison_plot=comparison_plot,
    )


def _plot_training_time(
    axis: Axes,
    *,
    cpu_summaries: tuple[DevicePerformanceSummary, ...],
    cuda_summaries: tuple[DevicePerformanceSummary, ...],
) -> None:
    """Plot mean training time with repetition variability."""

    batch_sizes = [
        summary.batch_size for summary in cpu_summaries
    ]

    axis.errorbar(
        batch_sizes,
        [
            summary.mean_elapsed_seconds
            for summary in cpu_summaries
        ],
        yerr=[
            summary.elapsed_seconds_standard_deviation
            for summary in cpu_summaries
        ],
        color=_CPU_COLOR,
        marker="o",
        linewidth=2,
        capsize=4,
        label="CPU",
    )
    axis.errorbar(
        batch_sizes,
        [
            summary.mean_elapsed_seconds
            for summary in cuda_summaries
        ],
        yerr=[
            summary.elapsed_seconds_standard_deviation
            for summary in cuda_summaries
        ],
        color=_CUDA_COLOR,
        marker="o",
        linewidth=2,
        capsize=4,
        label="CUDA",
    )

    axis.set_title("Training Time")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Mean elapsed time (seconds)")
    axis.set_xticks(batch_sizes)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)


def _plot_throughput(
    axis: Axes,
    *,
    cpu_summaries: tuple[DevicePerformanceSummary, ...],
    cuda_summaries: tuple[DevicePerformanceSummary, ...],
) -> None:
    """Plot mean samples processed per second."""

    batch_sizes = [
        summary.batch_size for summary in cpu_summaries
    ]

    axis.errorbar(
        batch_sizes,
        [
            summary.mean_samples_per_second
            for summary in cpu_summaries
        ],
        yerr=[
            summary.samples_per_second_standard_deviation
            for summary in cpu_summaries
        ],
        color=_CPU_COLOR,
        marker="o",
        linewidth=2,
        capsize=4,
        label="CPU",
    )
    axis.errorbar(
        batch_sizes,
        [
            summary.mean_samples_per_second
            for summary in cuda_summaries
        ],
        yerr=[
            summary.samples_per_second_standard_deviation
            for summary in cuda_summaries
        ],
        color=_CUDA_COLOR,
        marker="o",
        linewidth=2,
        capsize=4,
        label="CUDA",
    )

    axis.set_title("Training Throughput")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Mean samples per second")
    axis.set_xticks(batch_sizes)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)


def _plot_speedup(
    axis: Axes,
    analysis: BenchmarkAnalysis,
) -> None:
    """Plot CUDA elapsed-time speedup relative to CPU."""

    speedups = tuple(
        sorted(
            analysis.speedups,
            key=lambda speedup: speedup.batch_size,
        )
    )
    batch_sizes = [
        speedup.batch_size for speedup in speedups
    ]
    elapsed_time_speedups = [
        speedup.elapsed_time_speedup for speedup in speedups
    ]

    axis.plot(
        batch_sizes,
        elapsed_time_speedups,
        color=_SPEEDUP_COLOR,
        marker="o",
        linewidth=2,
    )
    axis.axhline(
        1.0,
        color="#777777",
        linestyle="--",
        linewidth=1,
    )

    for batch_size, speedup in zip(
        batch_sizes,
        elapsed_time_speedups,
        strict=True,
    ):
        axis.annotate(
            f"{speedup:.2f}×",
            xy=(batch_size, speedup),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    axis.set_title("CUDA Speedup")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("CPU time ÷ CUDA time")
    axis.set_xticks(batch_sizes)
    axis.grid(alpha=0.25)


def _summaries_for_device(
    analysis: BenchmarkAnalysis,
    device: str,
) -> tuple[DevicePerformanceSummary, ...]:
    """Return summaries for one device ordered by batch size."""

    return tuple(
        sorted(
            (
                summary
                for summary in analysis.device_summaries
                if summary.device == device
            ),
            key=lambda summary: summary.batch_size,
        )
    )


def _validate_analysis(
    *,
    cpu_summaries: tuple[DevicePerformanceSummary, ...],
    cuda_summaries: tuple[DevicePerformanceSummary, ...],
    speedup_batch_sizes: tuple[int, ...],
) -> None:
    """Validate that plotting inputs contain matching comparisons."""

    if not cpu_summaries:
        raise ValueError("CPU performance summaries are required.")

    if not cuda_summaries:
        raise ValueError("CUDA performance summaries are required.")

    cpu_batch_sizes = tuple(
        summary.batch_size for summary in cpu_summaries
    )
    cuda_batch_sizes = tuple(
        summary.batch_size for summary in cuda_summaries
    )

    if cpu_batch_sizes != cuda_batch_sizes:
        raise ValueError(
            "CPU and CUDA summaries must use identical batch sizes."
        )

    if cpu_batch_sizes != speedup_batch_sizes:
        raise ValueError(
            "Speedup results must use the summarized batch sizes."
        )
