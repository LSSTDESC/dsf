"""Benchmark numerical accuracy, runtime, and memory of chunked Hankel covariance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from dsf.hankel.hankel_transform_matrix_zeros import HankelTransformMatrixZeros

CHUNK_SIZES = (None, 256, 1024, 4096, 8192)


def make_transform(n_r: int, n_k: int) -> HankelTransformMatrixZeros:
    """Create a deterministic synthetic Hankel transform."""
    rng = np.random.default_rng(42)

    transform = HankelTransformMatrixZeros.__new__(HankelTransformMatrixZeros)

    transform.orders = (0,)
    transform.k = {0: np.linspace(0.1, 10.0, n_k)}
    transform.r = {0: np.linspace(1.0, 100.0, n_r)}
    transform.j = {0: rng.normal(size=(n_r, n_k))}
    transform.inverse_j_next_squared = {0: np.ones(n_k)}
    transform.normalization = {0: 0.5}

    return transform


def peak_memory_mb() -> float:
    """Return peak resident memory in megabytes."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if sys.platform == "darwin":
        return peak / 1024**2

    return peak / 1024


def run_case(
    n_r: int,
    n_k: int,
    chunk_size: int | None,
    reference_path: Path,
    repeats: int,
) -> dict:
    """Benchmark one projection and compare it with the direct reference."""
    transform = make_transform(n_r, n_k)

    k = transform.k[0]
    spectra = [
        np.exp(-k),
        1.0 / (1.0 + k**2),
    ]

    times = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()

        _, result = transform._project_spectra_to_radial(
            spectra,
            order=0,
            chunk_size=chunk_size,
        )

        times.append(time.perf_counter() - start)

    if chunk_size is None:
        np.save(reference_path, result)

    reference = np.load(reference_path)

    difference = np.abs(result - reference)
    scale = float(np.max(np.abs(reference)))

    max_abs_diff = float(np.max(difference))

    significant = np.abs(reference) > scale * 1e-10

    if np.any(significant):
        max_rel_diff = float(
            np.max(
                difference[significant]
                / np.abs(reference[significant])
            )
        )
    else:
        max_rel_diff = 0.0

    scaled_max_diff = (
        max_abs_diff / scale if scale > 0 else max_abs_diff
    )

    numerical_pass = bool(
        np.allclose(result, reference, rtol=1e-10, atol=1e-12)
    )

    symmetry_error = float(
        np.max(np.abs(result - result.T))
    )

    return {
        "n_r": n_r,
        "n_k": n_k,
        "chunk_size": "direct" if chunk_size is None else chunk_size,
        "runtime_s": float(np.median(times)),
        "peak_rss_mb": peak_memory_mb(),
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "scaled_max_diff": scaled_max_diff,
        "symmetry_error": symmetry_error,
        "numerical_pass": numerical_pass,
    }


def main() -> None:
    """Run isolated benchmark cases and save their results."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-r", type=int, default=256)
    parser.add_argument("--n-k", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--single", action="store_true")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/hankel/results_chunked.csv"),
    )

    args = parser.parse_args()

    if args.n_r <= 0 or args.n_k <= 0 or args.repeats <= 0:
        parser.error("n-r, n-k, and repeats must be positive.")

    if args.single:
        result = run_case(
            n_r=args.n_r,
            n_k=args.n_k,
            chunk_size=args.chunk_size,
            reference_path=args.reference,
            repeats=args.repeats,
        )

        print(json.dumps(result))
        return

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        reference_path = Path(tmpdir) / "reference.npy"

        for chunk_size in CHUNK_SIZES:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single",
                "--n-r",
                str(args.n_r),
                "--n-k",
                str(args.n_k),
                "--repeats",
                str(args.repeats),
                "--reference",
                str(reference_path),
            ]

            if chunk_size is not None:
                command.extend(
                    ["--chunk-size", str(chunk_size)]
                )

            env = os.environ.copy()

            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            env["VECLIB_MAXIMUM_THREADS"] = "1"
            env["BLIS_NUM_THREADS"] = "1"

            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            result = json.loads(completed.stdout)
            results.append(result)

            print(
                f"{str(result['chunk_size']):>8} | "
                f"{result['runtime_s']:>9.4f} s | "
                f"{result['peak_rss_mb']:>9.1f} MB | "
                f"scaled diff={result['scaled_max_diff']:.3e} | "
                f"symmetry={result['symmetry_error']:.3e} | "
                f"pass={result['numerical_pass']}"
            )

    direct_runtime = results[0]["runtime_s"]
    direct_memory = results[0]["peak_rss_mb"]

    for result in results:
        result["speedup"] = (
            direct_runtime / result["runtime_s"]
        )

        result["memory_ratio"] = (
            result["peak_rss_mb"] / direct_memory
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
