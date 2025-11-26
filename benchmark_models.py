import argparse
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from config import load_onnx, load_pytorch


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency and throughput for Bert model variants."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of benchmark samples to run (default: 100)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="Number of warmup runs before benchmarking (default: 10)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for input text (default: 32)",
    )
    return parser.parse_args()


def measure_latency(model: Any, name: str, inputs: dict, runs: int) -> list[float]:
    """Measure inference latency over multiple runs."""
    latencies = []
    with torch.no_grad():
        for _ in tqdm(range(runs), desc=f"Measuring {name}"):
            start = time.perf_counter()
            model(**inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    return latencies


def run_benchmark(
    name: str, loader_fn: Any, num_samples: int, warmup_runs: int, seq_len: int
) -> dict | None:
    """Run benchmark for a model."""
    print(f"\n[{'=' * 10} {name.upper()} {'=' * 10}]")

    loaded = loader_fn()
    if not loaded:
        print("   Skipping: Model files not found.")
        return None

    model, tokenizer, size_mb = loaded

    # Generate inputs
    text = "Functional programming is " * (seq_len // 4)
    inputs = tokenizer(text, return_tensors="pt")

    # Warmup
    for _ in tqdm(range(warmup_runs), desc=f"Warming up {name}"):
        model(**inputs)

    # Measure
    latencies = measure_latency(model, name, inputs, num_samples)
    avg_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))

    return {
        "Framework": name,
        "Size (MB)": size_mb,
        "Latency (ms)": avg_lat,
        "Std (ms)": std_lat,
        "IPS": 1000 / avg_lat,
    }


def print_results(results: list[dict]) -> None:
    """Print benchmark results table."""
    if not results:
        return

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    header = f"{'Framework':<25} | {'Size (MB)':<10} | {'Latency (ms)':<12} | {'Std (ms)':<10} | {'IPS':<10}"
    print(header)
    print("-" * 80)

    for r in results:
        print(
            f"{r['Framework']:<25} | {r['Size (MB)']:<10.2f} | "
            f"{r['Latency (ms)']:<12.2f} | {r['Std (ms)']:<10.2f} | {r['IPS']:<10.2f}"
        )

    print("=" * 80)


def main() -> None:
    """Run all benchmarks."""
    args = parse_args()

    tasks = {
        "PyTorch Original": lambda: load_pytorch(is_quantized=False),
        "PyTorch Quantized": lambda: load_pytorch(is_quantized=True),
        "ONNX Runtime": lambda: load_onnx(is_quantized=False),
        "ONNX Runtime Quantized": lambda: load_onnx(is_quantized=True),
    }

    results = [
        run_benchmark(name, loader, args.num_samples, args.warmup_runs, args.seq_len)
        for name, loader in tasks.items()
    ]
    print_results([r for r in results if r])


if __name__ == "__main__":
    main()
