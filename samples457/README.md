# CUDA host compiler review: cuda-samples PR457

Target: https://github.com/NVIDIA/cuda-samples/pull/457
Base: 5443602d89ed99aede2e4b7bf329daddeadb320e
Head: 70c867aaf00bcca8729c6a422c00c2d07ffb8500

The unconditional option causes a default-build regression: when CMAKE_CUDA_HOST_COMPILER is unset, the generated command contains -ccbin= and nvcc fails to preprocess host compiler properties.

Executed the actual vectorAddDrv generate_fatbin_vectorAdd target in Colab: CUDA13.0.88, CMake3.31.10, GNU13.3.0, architecture75, T4 runtime. These are compilation checks, not GPU execution or performance measurements.

| Revision | Default compiler | Explicit /usr/bin/g++ |
| --- | --- | --- |
| Base | Pass | Pass, custom command ignores selection |
| PR head | Fail: empty -ccbin | Pass, option forwarded |
| Head plus conditional patch | Pass | Pass, option forwarded |

The conditional patch was tested for one sample. The same unconditional expression occurs in all eight changed files, but the other seven targets, Windows/QNX, paths containing spaces and the original Fedora/GCC16 environment were not executed. No full-suite or merged-fix claim.

Run build-probe.py in a fresh Linux CUDA environment, then fix-probe.py in the same Python namespace. The scripts use /content/samples457 and pin both revisions. Fresh build directories are required for repeats because cached fatbins may skip nvcc. PR457-Colab.ipynb embeds the scripts. JSON files retain commands, exit codes and raw stdout/stderr. nvcc-probe.py is a supplementary direct compiler check.

conditional-host-compiler.patch is against the pinned PR head. This is independent review evidence and a proposed adjustment to another contributor's build fix.
