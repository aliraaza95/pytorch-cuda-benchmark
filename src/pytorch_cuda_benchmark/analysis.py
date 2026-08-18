"""Benchmark result aggregation and comparison utilities."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, stdev

from .benchmark import BenchmarkResult
from .data import BenchmarkDevice


@dataclass(frozen=True, slots=True)
class DevicePerformanceSummary:
    """Aggregated measurements for one device and batch size."""

    device: BenchmarkDevice
    batch_size: int
    mixed_precision: bool
    epochs: int
    repetitions: int
    mean_loss: float
    mean_elapsed_seconds: float
    elapsed_seconds_standard_deviation: float
    mean_samples_per_second: float
    samples_per_second_standard_deviation: float


@dataclass(frozen=True, slots=True)
class BatchSizeSpeedup:
    """CUDA performance relative to CPU for one batch size."""

    batch_size: int
    cuda_mixed_precision: bool
    cpu_mean_elapsed_seconds: float
    cuda_mean_elapsed_seconds: float
    elapsed_time_speedup: float
    cpu_mean_samples_per_second: float
    cuda_mean_samples_per_second: float
    throughput_speedup: float


@dataclass(frozen=True, slots=True)
class BenchmarkAnalysis:
    """Aggregated device measurements and CUDA speedups."""

    device_summaries: tuple[DevicePerformanceSummary, ...]
    speedups: tuple[BatchSizeSpeedup, ...]


def analyze_benchmark_results(
    results: Sequence[BenchmarkResult],
) -> BenchmarkAnalysis:
    """Aggregate repeated measurements and calculate CUDA speedups."""

    if not results:
        raise ValueError("At least one benchmark result is required.")

    device_summaries = _create_device_summaries(results)
    speedups = _calculate_speedups(device_summaries)

    return BenchmarkAnalysis(
        device_summaries=device_summaries,
        speedups=speedups,
    )


def _create_device_summaries(
    results: Sequence[BenchmarkResult],
) -> tuple[DevicePerformanceSummary, ...]:
    """Group measurements by device, batch size, and precision mode."""

    grouped_results: dict[
        tuple[BenchmarkDevice, int, bool, int],
        list[BenchmarkResult],
    ] = {}

    for result in results:
        key = (
            result.device,
            result.batch_size,
            result.mixed_precision,
            result.epochs,
        )
        grouped_results.setdefault(key, []).append(result)

    summaries: list[DevicePerformanceSummary] = []

    sorted_groups = sorted(
        grouped_results.items(),
        key=lambda item: (item[0][1], item[0][0]),
    )

    for key, grouped_measurements in sorted_groups:
        device, batch_size, mixed_precision, epochs = key

        losses = [
            measurement.mean_loss
            for measurement in grouped_measurements
        ]
        elapsed_times = [
            measurement.elapsed_seconds
            for measurement in grouped_measurements
        ]
        throughputs = [
            measurement.samples_per_second
            for measurement in grouped_measurements
        ]

        summaries.append(
            DevicePerformanceSummary(
                device=device,
                batch_size=batch_size,
                mixed_precision=mixed_precision,
                epochs=epochs,
                repetitions=len(grouped_measurements),
                mean_loss=fmean(losses),
                mean_elapsed_seconds=fmean(elapsed_times),
                elapsed_seconds_standard_deviation=(
                    _sample_standard_deviation(elapsed_times)
                ),
                mean_samples_per_second=fmean(throughputs),
                samples_per_second_standard_deviation=(
                    _sample_standard_deviation(throughputs)
                ),
            )
        )

    return tuple(summaries)


def _calculate_speedups(
    summaries: Sequence[DevicePerformanceSummary],
) -> tuple[BatchSizeSpeedup, ...]:
    """Calculate CUDA-to-CPU performance ratios by batch size."""

    cpu_summaries: dict[int, DevicePerformanceSummary] = {}
    cuda_summaries: dict[int, DevicePerformanceSummary] = {}

    for summary in summaries:
        target = (
            cpu_summaries
            if summary.device == "cpu"
            else cuda_summaries
        )

        if summary.batch_size in target:
            raise ValueError(
                "Each device must have one summary per batch size."
            )

        target[summary.batch_size] = summary

    if set(cpu_summaries) != set(cuda_summaries):
        raise ValueError(
            "CPU and CUDA summaries must contain identical batch sizes."
        )

    speedups: list[BatchSizeSpeedup] = []

    for batch_size in sorted(cpu_summaries):
        cpu_summary = cpu_summaries[batch_size]
        cuda_summary = cuda_summaries[batch_size]

        if cpu_summary.mean_elapsed_seconds <= 0:
            raise ValueError("CPU elapsed time must be greater than zero.")

        if cuda_summary.mean_elapsed_seconds <= 0:
            raise ValueError("CUDA elapsed time must be greater than zero.")

        if cpu_summary.mean_samples_per_second <= 0:
            raise ValueError("CPU throughput must be greater than zero.")

        speedups.append(
            BatchSizeSpeedup(
                batch_size=batch_size,
                cuda_mixed_precision=cuda_summary.mixed_precision,
                cpu_mean_elapsed_seconds=(
                    cpu_summary.mean_elapsed_seconds
                ),
                cuda_mean_elapsed_seconds=(
                    cuda_summary.mean_elapsed_seconds
                ),
                elapsed_time_speedup=(
                    cpu_summary.mean_elapsed_seconds
                    / cuda_summary.mean_elapsed_seconds
                ),
                cpu_mean_samples_per_second=(
                    cpu_summary.mean_samples_per_second
                ),
                cuda_mean_samples_per_second=(
                    cuda_summary.mean_samples_per_second
                ),
                throughput_speedup=(
                    cuda_summary.mean_samples_per_second
                    / cpu_summary.mean_samples_per_second
                ),
            )
        )

    return tuple(speedups)


def _sample_standard_deviation(
    values: Sequence[float],
) -> float:
    """Return sample deviation or zero for a single measurement."""

    if len(values) < 2:
        return 0.0

    return stdev(values)
