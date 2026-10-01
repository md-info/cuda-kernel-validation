# CUDA kernel validation

Reproducible, independent correctness validation of CUDA kernels. First case: NVIDIA/cuda-samples [PR #456](https://github.com/NVIDIA/cuda-samples/pull/456), which adds support for partial matrix-multiplication tiles.

## Verified result

Tested commit: `fd6f3f85e6ec1d35359c1a47139edacc0e328cf7`.

| Check | Result |
|---|---|
| GPU | Tesla T4, compute capability 7.5 |
| Compiler / driver | nvcc 12.8.93 / NVIDIA 580.82.07 |
| Correctness | 22/22 cases passed; zero observed absolute error |
| Compute Sanitizer memcheck | 0 errors |
| Compute Sanitizer synccheck | 0 errors |

Run on September 30, 2026 (Toronto); environment log timestamps use UTC. Raw evidence: [environment](colab-environment.txt), [validation and sanitizer output](colab-validation.txt). Logs were captured from the Colab cell outputs.

## Reproduce

Upload [PR456-Colab.ipynb](PR456-Colab.ipynb) into Colab, select a GPU runtime and run all cells. It fetches the upstream PR and checks its exact commit before compiling. If the PR head changes, it stops so that the new source can be reviewed. It writes the included harness, compiles for the allocated GPU, runs correctness checks and runs sanitizers when available.

For a local Linux CUDA environment, check out the tested upstream commit, then run:

```sh
nvcc -std=c++17 -O2 -arch=sm_75 -I /path/to/cuda-samples -I /path/to/cuda-samples/Common validate.cu -o validate
./validate
compute-sanitizer --tool memcheck --error-exitcode 99 ./validate
compute-sanitizer --tool synccheck --error-exitcode 99 ./validate
```

Adjust the architecture for your GPU. A supported CUDA host compiler is required.

## Test design

[validate.cu](validate.cu) includes the unchanged upstream sample, renames its entry point, and calls its actual `MatrixMulCUDA<16>` and `<32>` kernels. It does not substitute a rewritten implementation.

Eleven `(M,N,K)` shapes per tile size cover sub-tile matrices, aligned controls, rectangular outputs and partial reduction tiles: `(1,1,1)`, `(7,5,3)`, `(15,17,31)`, `(16,16,16)`, `(17,15,17)`, `(31,33,32)`, `(32,32,32)`, `(33,31,33)`, `(63,65,67)`, `(65,63,64)`, `(97,101,127)`.

Inputs vary by element and contain signed binary fractions. Every output is compared with an independent double-precision CPU multiplication, using absolute-plus-relative tolerance. Output buffers start poisoned with NaNs, so skipped stores cannot accidentally pass. Nonfinite results are rejected explicitly. Binary fractions make these small cases exactly representable; zero error here is not a claim about arbitrary floating-point inputs.

Partial tile lanes load zero but still participate in both block barriers. An early return in those lanes would risk invalid synchronization. The final output store is guarded separately. The cases exercise partial K tiles as well as output edges.

## Scope and provenance

The upstream kernel and boundary fix are authored by the contributors to NVIDIA/cuda-samples, including PR author efegokdemir. This repository contributes an external validation harness and reproducible evidence, not that kernel's authorship. The harness, notebook and documentation were prepared with Codex assistance and executed through a signed-in Colab session.

This is targeted validation on one GPU, not a full repository test suite, a performance study, or a merged code contribution. Larger cases, randomized floating-point inputs, more architectures and racecheck remain future coverage.

The existing sample's constant-input checker can accept NaN because comparison with a NaN error term is false. Our harness rejects that case. This observation does not demonstrate a failure in the patched kernel.

Public validation comment: https://github.com/NVIDIA/cuda-samples/pull/456#issuecomment-5923407662
