"""Native Windows checks of the actual PR runner functions, without GPU claims."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assert sys.platform == "win32"
root = Path(__file__).parent.resolve()
fixture = root / "fixture with spaces"
fixture.mkdir(exist_ok=True)
binary = Path(os.environ["SystemRoot"]) / "System32" / "where.exe"
shutil.copy2(binary, fixture / "probe.exe")
shutil.copy2(binary, fixture / "upper.EXE")
for suffix in (".dll", ".lib", ".pdb", ".obj", ".txt", ".json", ".cu", ".py"):
    (fixture / ("artifact" + suffix)).write_text("not an executable")
(fixture / "directory.exe").mkdir(exist_ok=True)
print(json.dumps({"python": sys.version, "platform": platform.platform(), "binary": str(binary)}))
records = []
for version in ("base", "head"):
    runner = load(version)
    found = sorted(p.name for p in runner.find_executables(fixture))
    print(json.dumps({"version": version, "discovered": found}))
    if version == "head":
        assert found == ["probe.exe", "upper.EXE"]
    for case in ("absolute", "relative_nested", "bare_name"):
        previous = Path.cwd()
        try:
            if case == "absolute":
                exe = fixture / "probe.exe"
            elif case == "relative_nested":
                os.chdir(root)
                exe = Path(fixture.name) / "probe.exe"
            else:
                os.chdir(fixture)
                exe = next(p for p in runner.find_executables(".") if p.name == "probe.exe")
                assert str(exe) == "probe.exe"
            output = root / f"{version}-{case}.log"
            result = runner.run_single_test_instance(exe, ["/Q"], str(output), ["cmd.exe"], case)
            records.append({"version": version, "case": case, **result})
            if version == "head" and case != "bare_name":
                assert result["return_code"] == 0, result
            if version == "head" and case == "bare_name":
                assert result["return_code"] == -1 and "Error:" in result["status"], result
                assert os.path.dirname(str(exe)) == ""
                # Control: normalize both executable and working directory.
                import subprocess
                fixed = subprocess.run([str(exe.resolve()), "/Q", "cmd.exe"], cwd=str(exe.resolve().parent))
                assert fixed.returncode == 0
        finally:
            os.chdir(previous)
print(json.dumps({"records": records, "status": "PASS", "bare_name_control": "normalized cwd passes"}, indent=2))
(root / "results.json").write_text(json.dumps(records, indent=2))
