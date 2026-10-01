# CUTLASS tile-offset regression

Issue: https://github.com/NVIDIA/cutlass/issues/3487
Pinned upstream revision: `0b55a2f691d69981583568fd9eb69687b1f0de8a`.
Executed notebook: https://colab.research.google.com/drive/1NV-dyDY0NdJEVayH2iCcuuTriBA7EH6y

## Executed evidence

On Colab Tesla T4 with nvcc 12.8.93, the standalone CUDA addressing probe compiled and executed against unchanged upstream: 24 of 64 cases failed. With a three-expression fix to the plain pitch-linear specialization, all 64 passed. Compute Sanitizer memcheck reported zero errors on the fixed probe. See captured `colab-environment.txt`, `colab-baseline.txt`, and `colab-fixed-memcheck.txt`.

The final raw logs are `cutlass3487-environment.log`, `cutlass3487-baseline-final.log`, `cutlass3487-fixed-final.log`, `cutlass3487-memcheck-final.log`, `cutlass3487-suite-baseline.log`, and `cutlass3487-suite-fixed.log`. The native transform/threadblock target passes its 18 existing tests on unchanged upstream and fails all four added regressions; with the patch all 22 tests pass. Final standalone runs repeat the 24/64 failing baseline and 64/64 passing fixed result, with zero memcheck errors. The final probe starts at an interior tensor offset so backward residual transitions remain within the allocation.

The probe compares every lane's observed byte displacement against an independently calculated pitch-linear address. Cases cover fp16/fp32, both advancement ranks, residual/steady-state paths, aligned/partial extents, and zero/single-axis/diagonal moves. It observes addresses without dereferencing tensor data. Memcheck therefore verifies the executed probe's allocations and result stores, not correctness of payload loads or a complete GEMM.

## Findings and patch scope

For advancement along the strided rank, the steady-state cross-axis displacement uses element counts as bytes. For advancement along the contiguous rank, the cross-axis displacement also omits the pitch-linear layout stride, in both residual and steady-state paths. The patch uses the existing element-offset helper and includes the layout stride where required.

The patch intentionally covers only the tested non-gather, non-permuted pitch-linear specialization. AffineRankN, ELL, triangular and 2D-thread-tile variants remain separate work. No performance improvement is claimed.

`probe.cu` is the standalone regression. `predicated_tile_access_iterator.cu` adapts the regression into four Google Tests for CUTLASS's transform/threadblock target. The patch includes that file and its CMake registration. The target was configured for SM75 with examples/library disabled and utility tools enabled, using CUDA 12.8.93 and GNU 13.3.0. The complete CUTLASS suite and other architectures were not run.

`CUTLASS3487-Colab.ipynb` reproduces the pinned baseline and fixed standalone probe. It stops on CUDA or compilation failures, preserves baseline failures as diagnostic output, and requires the fixed run and sanitizer to succeed. The earlier interactive session contains compile/edit retries; only the corrected execution supplies the reported addressing results.

## Attribution and status

The bug was reported by the upstream issue author. The regression, patch preparation, and documentation were produced with Codex assistance and executed in the user's Colab session. The upstream library remains NVIDIA CUTLASS code. Upstream PR: https://github.com/NVIDIA/cutlass/pull/3704 . Open for review as of September 30, 2026; not merged.
