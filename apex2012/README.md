# Apex PR #2012: autograd callback regression review

Review target: https://github.com/NVIDIA/apex/pull/2012

The proposed inference-leak guard returns early from `setup_context` when `torch.is_grad_enabled()` is false. In PyTorch 2.11.0, that callback runs with grad mode disabled even when the caller enables gradients and the input requires them. The guard therefore skips the tensor saves needed for training backward.

## Executed evidence

Google Colab Tesla T4 (SM75), PyTorch 2.11.0+cu128, CUDA build 12.8. Tested CPU and CUDA tensors in the same process.

- Base: `becbb77cea4cb54f2929f7c938a0a6f7dd1fdc39`.
- PR head: `87584037722e3f3e8753b29c5159bbe8bbec5dbc`.
- Exact source path: `apex/normalization/fused_layer_norm.py`.
- Four setup callbacks: layer norm and RMSNorm, affine and non-affine.
- Both memory-efficient settings, FP32/FP16, contiguous/strided inputs, CPU/CUDA.
- Base backward: all 64 cases passed the tensor-save contract and identity gradients.
- PR backward: all 64 cases reached backward with zero saved tensors, failing the contract.
- All 128 `no_grad` controls bypassed setup_context entirely.

## Scope of the probe

`probe.py` extracts the exact callback AST from the pinned files and registers it with real `torch.library.custom_op` operators. The forward and backward implementations are identity probes, not Apex normalization kernels. Backward checks the number of saved tensors expected by the corresponding Apex backward before verifying identity gradients. No fake context or mocked grad-mode result is used.

This proves a callback contract regression in the tested framework version. It does **not** execute the compiled Apex CUDA extension, reproduce the B200/bfloat16 memory leak from issue #1999, establish a fix for that leak, validate normalization numerics, or cover all PyTorch versions. The explicit `missing saved tensors` error is emitted by the probe; Apex's own backward unpacks the same saved tuple and would fail at that point if the patched callback is reached.

## Reproduce

Run the portable `Apex2012-Review.ipynb` with a CUDA runtime, or place `probe.py`, `base.py`, and `head.py` in one directory and run:

```bash
python probe.py
```

The script asserts the observed base/head distinction and writes `probe-results.json`. Expected regression failures are recorded as evidence, so a successful probe exits zero after establishing them. Raw output is `apex2012-probe.log`.

## Suggested review action

Remove the four grad-mode guards and add actual Apex training-backward controls alongside the original inference-leak reproducer. In this tested framework version, setup_context is already skipped on the outer no-grad dispatch; the callback's local grad mode is not a reliable test of whether backward will run.

The source snapshots are upstream Apex code, including the PR author's changes; they are retained under `LICENSE.apex`. The identity probe and review evidence are this repository's independent contribution. No competing fix PR is being opened and no upstream merge is claimed.
