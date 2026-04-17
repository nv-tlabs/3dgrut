# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataloader profiler — measures batch generation throughput in isolation.

Works with any dataset type (colmap, ncore, nerf, scannetpp). For ncore
datasets, additionally compares JPEG decode backends and image pre-loading
modes.

Usage examples:

    # NCore: runs all decode-backend / preload variants and prints comparison
    python benchmark/profile_dataloader.py \\
        --config-name=apps/ncore_3dgut.yaml \\
        path='/tmp/colmap-ppisp/huerstholz/huerstholz.json'

    # COLMAP: baseline throughput
    python benchmark/profile_dataloader.py \\
        --config-name=apps/colmap_3dgut.yaml \\
        path='/data/mipnerf360/garden'

    # Override benchmark parameters
    python benchmark/profile_dataloader.py \\
        --config-name=apps/ncore_3dgut.yaml \\
        path='/tmp/colmap-ppisp/huerstholz/huerstholz.json' \\
        +benchmark.warmup_iters=20 \\
        +benchmark.measure_iters=200

    # Only run specific ncore configs
    python benchmark/profile_dataloader.py \\
        --config-name=apps/ncore_3dgut.yaml \\
        path='/tmp/colmap-ppisp/huerstholz/huerstholz.json' \\
        '+benchmark.configs=[simplejpeg]'

    # With cProfile output
    python benchmark/profile_dataloader.py \\
        --config-name=apps/ncore_3dgut.yaml \\
        path='/tmp/colmap-ppisp/huerstholz/huerstholz.json' \\
        +benchmark.cprofiler_output=dataloader.prof
"""

import cProfile
import gc
import sys
import time
from dataclasses import dataclass, field

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])


# ---------------------------------------------------------------------------
# NCore benchmark configuration variants
# ---------------------------------------------------------------------------

# Each entry: (display_name, dataset config overrides)
NCORE_CONFIGS: dict[str, tuple[str, dict]] = {
    "PIL": (
        "PIL baseline",
        {"jpeg_backend_cpu": "PIL"},
    ),
    "simplejpeg": (
        "simplejpeg",
        {"jpeg_backend_cpu": "simplejpeg"},
    ),
}

ALL_NCORE_CONFIG_KEYS = list(NCORE_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    name: str
    samples: list[float] = field(default_factory=list)
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)

    @property
    def median_its(self) -> float:
        return float(np.median(self.samples)) if self.samples else 0.0


def _make_dataloader(conf: DictConfig):
    """Construct dataset + dataloader from a fully-composed Hydra config (same path as Trainer)."""
    import threedgrut.datasets as datasets
    from threedgrut.datasets.utils import (
        MultiEpochsDataLoader,
        configure_dataloader_for_platform,
    )

    train_dataset, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)

    dataloader_kwargs = configure_dataloader_for_platform(
        {
            "num_workers": conf.num_workers,
            "batch_size": 1,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True if conf.num_workers > 0 else False,
        }
    )

    dataloader = MultiEpochsDataLoader(train_dataset, **dataloader_kwargs)
    return train_dataset, dataloader


def _apply_dataset_overrides(conf: DictConfig, overrides: dict) -> DictConfig:
    """Return a deep copy of *conf* with dataset-level keys overridden."""
    import copy

    conf = copy.deepcopy(conf)
    for key, value in overrides.items():
        OmegaConf.update(conf, f"dataset.{key}", value)
    return conf


def _next_batch(it, dataloader):
    """Get next batch, re-creating the iterator on epoch boundary."""
    try:
        # get the item from the iterator
        cpu_batch = next(it)
        gpu_batch = dataloader.dataset.get_gpu_batch_with_intrinsics(cpu_batch)
        return cpu_batch, it
    except StopIteration:
        it = iter(dataloader)
        return next(it), it


def reccord_time():
    cpu_time = time.perf_counter()
    cuda_event = torch.cuda.Event(enable_timing=True)
    cuda_event.record()
    return cpu_time, cuda_event


def _run_benchmark(
    dataloader,
    warmup_iters: int,
    measure_iters: int,
    sample_n: int,
    profiler: cProfile.Profile | None = None,
) -> BenchmarkResult:
    """Pull batches from *dataloader* and return timing samples."""
    result = BenchmarkResult(name="")
    it = iter(dataloader)

    # Warmup
    for _ in range(warmup_iters):
        _, it = _next_batch(it, dataloader)
    torch.cuda.synchronize()

    # Measure
    if profiler is not None:
        profiler.enable()

    start, start_cuda = reccord_time()
    for i in range(1, measure_iters + 1):
        _, it = _next_batch(it, dataloader)

        if i % sample_n == 0:
            end, end_cuda = reccord_time()

            its = sample_n / (end - start)
            result.samples.append(its)
            result.event_pairs.append((start_cuda, end_cuda))

            print(f"  {sample_n}-iter avg: {its:>8.1f} it/s\r", end="")
            sys.stdout.flush()

            start = time.perf_counter()
            start_cuda = torch.cuda.Event(enable_timing=True)
            start_cuda.record()

    if profiler is not None:
        profiler.disable()

    torch.cuda.synchronize()
    for i, (start_cuda, end_cuda) in enumerate(result.event_pairs):
        start_cuda.synchronize()
        end_cuda.synchronize()
        elapsed = start_cuda.elapsed_time(end_cuda) / 1000.0
        old_its = result.samples[i]
        new_its = sample_n / elapsed
        result.samples[i] = new_its

    return result


def _print_summary(results: list[BenchmarkResult]) -> None:
    """Print a comparison table of all benchmark results."""
    baseline_its = results[0].median_its if results else 1.0

    print()
    print("=" * 64)
    print("Summary")
    print("=" * 64)

    has_speedup = len(results) > 1
    if has_speedup:
        print(f"{'Config':<40s} {'Median it/s':>12s} {'Speedup':>9s}")
        print("-" * 64)
        for r in results:
            speedup = r.median_its / baseline_its if baseline_its > 0 else 0.0
            print(f"{r.name:<40s} {r.median_its:>12.1f} {speedup:>8.2f}x")
    else:
        print(f"{'Config':<40s} {'Median it/s':>12s}")
        print("-" * 64)
        for r in results:
            print(f"{r.name:<40s} {r.median_its:>12.1f}")

    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", version_base=None)
def main(conf: DictConfig) -> None:
    # Extract benchmark parameters (passed via +benchmark.* overrides)
    bench_conf = OmegaConf.to_container(OmegaConf.create(conf.get("benchmark", {})), resolve=True)
    warmup_iters: int = bench_conf.get("warmup_iters", 1000)
    measure_iters: int = bench_conf.get("measure_iters", 10000)
    sample_n: int = bench_conf.get("sample_n", 1000)
    cprofiler_output: str | None = bench_conf.get("cprofiler_output", None)
    requested_configs: list[str] | None = bench_conf.get("configs", None)

    dataset_type = conf.dataset.type
    dataset_path = conf.path

    print()
    print("=" * 64)
    print("Dataloader Benchmark")
    print("=" * 64)
    print(f"Dataset type   : {dataset_type}")
    print(f"Dataset path   : {dataset_path}")
    print(f"num_workers    : {conf.num_workers}")
    print(f"Warmup iters   : {warmup_iters}")
    print(f"Measure iters  : {measure_iters}")
    print(f"Sample group   : {sample_n}")
    print()

    # Determine which configurations to benchmark
    if dataset_type == "ncore":
        config_keys = requested_configs if requested_configs is not None else ALL_NCORE_CONFIG_KEYS
        # Validate requested config keys
        for key in config_keys:
            if key not in NCORE_CONFIGS:
                raise ValueError(f"Unknown ncore benchmark config: '{key}'. " f"Available: {ALL_NCORE_CONFIG_KEYS}")
        configs_to_run: list[tuple[str, DictConfig]] = []
        for key in config_keys:
            display_name, overrides = NCORE_CONFIGS[key]
            configs_to_run.append((display_name, _apply_dataset_overrides(conf, overrides)))
    else:
        # Non-ncore datasets: single baseline run with the config as-is
        configs_to_run = [("baseline", conf)]

    # Optional cProfile (enabled only for the last config to avoid noise)
    profiler = cProfile.Profile() if cprofiler_output else None

    results: list[BenchmarkResult] = []

    for i, (name, run_conf) in enumerate(configs_to_run):
        print(f"--- {name} ---")

        _, dataloader = _make_dataloader(run_conf)

        # Enable profiler only on last config
        active_profiler = profiler if (profiler and i == len(configs_to_run) - 1) else None

        result = _run_benchmark(dataloader, warmup_iters, measure_iters, sample_n, active_profiler)
        result.name = name

        print(f"  Median: {result.median_its:.1f} it/s")
        print()

        results.append(result)

        # Tear down to free resources before next config
        del dataloader
        gc.collect()

    # cProfile output
    if profiler is not None and cprofiler_output:
        profiler.dump_stats(cprofiler_output)
        print(f"cProfile output written to {cprofiler_output}")

    _print_summary(results)


if __name__ == "__main__":
    main()
