"""Simulated dataset generator for CPU/GPU performance modelling.

The generated samples emulate three workload archetypes running on a
Windows 11 mobile workstation equipped with an Intel Core i9-13900HX CPU
and an NVIDIA GeForce RTX 4060 (8 GB) GPU. Each archetype captures the
static characteristics that mirror the kernel feature extractor logic
(Control-flow, loop structure, floating-point intensity, etc.) and their
impact on dynamic execution times.

Running this module will produce 500 simulated observations split across
three performance personas:

* GPU absolute advantage (``deep_learning_ids``)
* CPU absolute advantage (``stateful_protocol_parser``)
* Crossover advantage (``parallel_pattern_match``)

Two CSV files are written to the script directory:
``cpu.csv`` and ``gpu.csv``. Both contain the four static features, one
input size feature, and the respective execution time (milliseconds) for
the target processor type.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

RANDOM_SEED = 13900  # deterministic but derived from the CPU model number
SAMPLE_COUNT = 500


@dataclass(frozen=True)
class WorkloadSample:
    """Container describing one simulated workload instance."""

    feat_float_ops: float
    feat_control_statements: float
    feat_loops: float
    feat_function_calls: float
    input_data_size_bytes: float
    cpu_time_ms: float
    gpu_time_ms: float

    def cpu_row(self) -> Sequence[float]:
        return (
            self.feat_float_ops,
            self.feat_control_statements,
            self.feat_loops,
            self.feat_function_calls,
            self.input_data_size_bytes,
            self.cpu_time_ms,
        )

    def gpu_row(self) -> Sequence[float]:
        return (
            self.feat_float_ops,
            self.feat_control_statements,
            self.feat_loops,
            self.feat_function_calls,
            self.input_data_size_bytes,
            self.gpu_time_ms,
        )


def _gpu_absolute_advantage(count: int) -> Iterator[WorkloadSample]:
    """Simulate compute bound workloads where the GPU dominates."""

    for _ in range(count):
        feat_float_ops = random.uniform(8.0e7, 2.0e8)
        feat_control_statements = random.uniform(20, 90)
        feat_loops = random.uniform(140, 260)
        feat_function_calls = random.uniform(45, 110)
        input_data_size_bytes = random.uniform(5.0e6, 6.0e7)

        gpu_startup_ms = 9.5 + random.uniform(0, 6)
        gpu_compute_ms = (input_data_size_bytes / 1.5e6) * 0.8
        gpu_float_penalty_ms = (feat_float_ops / 1.0e8) * 6.5
        gpu_time_ms = gpu_startup_ms + gpu_compute_ms + gpu_float_penalty_ms

        cpu_superlinear_component = math.pow(input_data_size_bytes / 1.0e6, 1.25) * 8.5
        cpu_control_penalty = feat_control_statements * 0.12
        cpu_float_penalty = feat_float_ops / 2.8e6
        cpu_time_ms = 18.0 + cpu_superlinear_component + cpu_control_penalty + cpu_float_penalty

        yield WorkloadSample(
            feat_float_ops=feat_float_ops,
            feat_control_statements=feat_control_statements,
            feat_loops=feat_loops,
            feat_function_calls=feat_function_calls,
            input_data_size_bytes=input_data_size_bytes,
            cpu_time_ms=cpu_time_ms,
            gpu_time_ms=gpu_time_ms,
        )


def _cpu_absolute_advantage(count: int) -> Iterator[WorkloadSample]:
    """Simulate branch-heavy workloads where the CPU dominates."""

    for _ in range(count):
        feat_float_ops = random.uniform(5.0e3, 6.0e4)
        feat_control_statements = random.uniform(5.0e3, 2.4e4)
        feat_loops = random.uniform(40, 180)
        feat_function_calls = random.uniform(2.2e3, 9.5e3)
        input_data_size_bytes = random.uniform(8.0e4, 6.5e6)

        cpu_startup_ms = 1.2 + random.uniform(0, 1.2)
        cpu_logic_cost = (feat_control_statements + feat_function_calls * 0.65) * 0.0011
        cpu_linear_cost = (input_data_size_bytes / 6.0e5) * 3.2
        cpu_time_ms = cpu_startup_ms + cpu_logic_cost + cpu_linear_cost

        gpu_startup_ms = 42.0 + random.uniform(0, 18.0)
        gpu_branch_penalty = (feat_control_statements + feat_function_calls) * 0.0065
        gpu_transfer_penalty = (input_data_size_bytes / 2.5e5) * 6.5
        gpu_time_ms = gpu_startup_ms + gpu_branch_penalty + gpu_transfer_penalty

        yield WorkloadSample(
            feat_float_ops=feat_float_ops,
            feat_control_statements=feat_control_statements,
            feat_loops=feat_loops,
            feat_function_calls=feat_function_calls,
            input_data_size_bytes=input_data_size_bytes,
            cpu_time_ms=cpu_time_ms,
            gpu_time_ms=gpu_time_ms,
        )


def _crossover_advantage(count: int) -> Iterator[WorkloadSample]:
    """Simulate workloads where the faster device depends on the input size."""

    for _ in range(count):
        feat_float_ops = random.uniform(4.0e6, 2.2e7)
        feat_control_statements = random.uniform(400, 2.2e3)
        feat_loops = random.uniform(210, 520)
        feat_function_calls = random.uniform(480, 1.6e3)
        input_data_size_bytes = random.uniform(2.0e5, 5.5e7)

        gpu_startup_ms = 12.0 + random.uniform(0, 8.0)
        gpu_parallel_gain = (input_data_size_bytes / 1.2e6) * 1.05
        gpu_float_cost = (feat_float_ops / 1.2e7) * 4.0
        gpu_loop_cost = feat_loops * 0.03
        gpu_time_ms = gpu_startup_ms + gpu_parallel_gain + gpu_float_cost + gpu_loop_cost

        if input_data_size_bytes < 5.0e6:
            cpu_base = 4.5
            cpu_size_factor = (input_data_size_bytes / 1.0e6) * 3.4
        else:
            cpu_base = 4.5
            cpu_size_factor = (input_data_size_bytes / 1.0e6) * 5.6

        cpu_logic_penalty = (feat_control_statements * 0.004) + (feat_function_calls * 0.0025)
        cpu_loop_penalty = feat_loops * 0.04
        cpu_time_ms = cpu_base + cpu_size_factor + cpu_logic_penalty + cpu_loop_penalty

        yield WorkloadSample(
            feat_float_ops=feat_float_ops,
            feat_control_statements=feat_control_statements,
            feat_loops=feat_loops,
            feat_function_calls=feat_function_calls,
            input_data_size_bytes=input_data_size_bytes,
            cpu_time_ms=cpu_time_ms,
            gpu_time_ms=gpu_time_ms,
        )


def generate_samples(sample_count: int = SAMPLE_COUNT) -> list[WorkloadSample]:
    """Generate workload samples spanning the three archetypes."""

    random.seed(RANDOM_SEED)

    # Split the total sample count across the three archetypes.
    gpu_count = sample_count // 3
    cpu_count = sample_count // 3
    crossover_count = sample_count - gpu_count - cpu_count

    samples = list(_gpu_absolute_advantage(gpu_count))
    samples.extend(_cpu_absolute_advantage(cpu_count))
    samples.extend(_crossover_advantage(crossover_count))

    random.shuffle(samples)
    return samples


def write_csvs(samples: Sequence[WorkloadSample], directory: Path) -> None:
    """Persist the simulated dataset into cpu.csv and gpu.csv."""

    directory.mkdir(parents=True, exist_ok=True)
    cpu_path = directory / "cpu.csv"
    gpu_path = directory / "gpu.csv"

    with cpu_path.open("w", newline="", encoding="utf-8") as cpu_file:
        writer = csv.writer(cpu_file)
        writer.writerow(
            [
                "feat_float_ops",
                "feat_control_statements",
                "feat_loops",
                "feat_function_calls",
                "input_data_size_bytes",
                "cpu_time_ms",
            ]
        )
        for sample in samples:
            writer.writerow(sample.cpu_row())

    with gpu_path.open("w", newline="", encoding="utf-8") as gpu_file:
        writer = csv.writer(gpu_file)
        writer.writerow(
            [
                "feat_float_ops",
                "feat_control_statements",
                "feat_loops",
                "feat_function_calls",
                "input_data_size_bytes",
                "gpu_time_ms",
            ]
        )
        for sample in samples:
            writer.writerow(sample.gpu_row())


def main() -> None:
    samples = generate_samples(SAMPLE_COUNT)
    write_csvs(samples, Path(__file__).parent)


if __name__ == "__main__":
    main()
