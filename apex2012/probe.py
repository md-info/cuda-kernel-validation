"""Exercise exact Apex setup_context callbacks through real PyTorch custom ops.

Forward/backward bodies are identity probes, not Apex normalization kernels.
This isolates tensor saving and grad-mode dispatch without compiling extensions.
"""
import ast
import itertools
import json
from pathlib import Path

import torch


CALLBACKS = {
    "layer_affine": ("_fused_layer_norm_affine_setup_context", 5, 3),
    "rms_affine": ("_fused_rms_norm_affine_setup_context", 3, 2),
    "layer": ("_fused_layer_norm_setup_context", 3, 3),
    "rms": ("_fused_rms_norm_setup_context", 2, 2),
}


def extract(path, name):
    tree = ast.parse(Path(path).read_text())
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1
    # Execute only the exact callback AST; no imported Apex modules or stand-in ctx.
    module = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def register(version, kind, path):
    callback_name, expected_saved, outputs = CALLBACKS[kind]
    callback = extract(path, callback_name)
    events = []
    inputs = "Tensor input, "
    if "affine" in kind:
        inputs += "Tensor weight, "
        if kind == "layer_affine":
            inputs += "Tensor bias, "
    inputs += "SymInt[] normalized_shape, float eps, bool memory_efficient"
    schema = f"({inputs}) -> ({', '.join(['Tensor'] * outputs)})"

    def forward(*args):
        x = args[0]
        auxiliary = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        return tuple([x.clone()] + [auxiliary.clone() for _ in range(outputs - 1)])

    op = torch.library.custom_op(f"apex2012probe::{version}_{kind}", forward,
                                 mutates_args=(), schema=schema)

    def setup(ctx, inputs, output):
        events.append({"grad_enabled": torch.is_grad_enabled(), "requires_grad": inputs[0].requires_grad})
        callback(ctx, inputs, output)

    def backward(ctx, grad_output, *grad_auxiliary):
        saved = ctx.saved_tensors
        if len(saved) != expected_saved:
            raise RuntimeError(f"missing saved tensors: expected {expected_saved}, got {len(saved)}")
        result = [grad_output]
        if "affine" in kind:
            result.append(torch.zeros_like(saved[1]))
            if kind == "layer_affine":
                result.append(torch.zeros_like(saved[2]))
        return tuple(result + [None, None, None])

    op.register_autograd(backward, setup_context=setup)
    return op, events


def main():
    assert torch.cuda.is_available(), "This evidence run requires CUDA"
    print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}))
    records = []
    for version, path in (("base", "base.py"), ("head", "head.py")):
        for kind in CALLBACKS:
            op, events = register(version, kind, path)
            for device, dtype, memory_efficient, strided in itertools.product(
                    ("cpu", "cuda"), (torch.float32, torch.float16), (False, True), (False, True)):
                x = torch.randn(4, 8, device=device, dtype=dtype)
                if strided:
                    x = x.T
                x.requires_grad_()
                args = [x]
                if "affine" in kind:
                    args.append(torch.ones(x.shape[-1], device=device, dtype=dtype, requires_grad=True))
                    if kind == "layer_affine":
                        args.append(torch.zeros(x.shape[-1], device=device, dtype=dtype, requires_grad=True))
                args += [[x.shape[-1]], 1e-5, memory_efficient]
                events.clear()
                out = op(*args)[0]
                assert out.requires_grad
                assert len(events) == 1 and events[0]["requires_grad"]
                callback_grad = events[0]["grad_enabled"]
                error = None
                try:
                    out.sum().backward()
                except RuntimeError as exc:
                    error = str(exc)
                if version == "base":
                    assert error is None, error
                    torch.testing.assert_close(x.grad, torch.ones_like(x))
                    for parameter in args[1:-3]:
                        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
                else:
                    assert error == f"missing saved tensors: expected {CALLBACKS[kind][1]}, got 0", error
                events.clear()
                with torch.no_grad():
                    inference_out = op(*args)[0]
                assert not inference_out.requires_grad
                assert len(events) == 0, "setup_context unexpectedly ran under no_grad"
                records.append({"version": version, "kind": kind, "device": device,
                                "dtype": str(dtype), "memory_efficient": memory_efficient,
                                "strided": strided, "callback_grad_enabled": callback_grad,
                                "backward_error": error, "no_grad_setup_calls": len(events)})
    torch.cuda.synchronize()
    result = {"records": records, "base_backward_pass": 64, "head_backward_missing_saves": 64,
              "no_grad_controls": 128, "status": "PASS"}
    Path("probe-results.json").write_text(json.dumps(result, indent=2))
    for record in records:
        print(json.dumps(record))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
