# Native Windows review of cuda-samples PR462

Target: https://github.com/NVIDIA/cuda-samples/pull/462

Pinned base5443602d89ed99aede2e4b7bf329daddeadb320e and head6b6173174d92a435c5d5a0a167c3961318a5fda3. Executed actual `find_executables` and `run_single_test_instance` functions on Windows11 build26200, Python3.12.14. No mocks of subprocess, platform or discovery; no CUDA sample execution claimed.

The probe copies Windows' existing `where.exe` into a local directory containing spaces, invokes it with `/Q cmd.exe`, and tests artifact discovery. It does not compile or download a binary. The copied executable is not published.

- Base discovers seven non-executable artifacts alongside the two executables; head discovers only `.exe` and uppercase `.EXE` files and excludes the directory named `.exe`.
- Base fails absolute and nested relative launches with WinError2; head passes both, including argument forwarding and the path with spaces.
- Discovery from `.` returns a bare `probe.exe` path. Both versions fail that launch with WinError123 because `os.path.dirname(str(executable))` is empty.
- A subprocess control resolving both the command path and its parent working directory passes.

The bare-name case is pre-existing, not a new regression, but remains relevant to the PR's Windows path-resolution goal. Suggested adjustment: resolve the executable once and use its resolved parent as `cwd`. The changed default discovery directory was inspected but the complete CLI, GPU detection, all CUDA samples and POSIX behavior were not executed.

Run `python probe.py` on Windows with base.py and head.py alongside it. It creates a reusable fixture directory, checks expected before/after outcomes and writes results.json. Expected baseline/edge-case failures are evidence; the probe itself exits zero after confirming them. Raw output is windows-probe.log. Upstream runner snapshots retain their NVIDIA copyright and license header; this is an independent review probe, not authorship of the PR fix.
