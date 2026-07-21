import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_importing_homopt_optim_does_not_eagerly_load_optional_stacks():
    payload = _run_python(
        "import sys, json; import homopt.optim; "
        "print(json.dumps({"
        "'cvxpy': any(name.startswith('cvxpy') for name in sys.modules), "
        "'pypower': any(name.startswith('pypower') for name in sys.modules), "
        "'scipy': any(name.startswith('scipy') for name in sys.modules), "
        "'skopt': any(name.startswith('skopt') for name in sys.modules)}))"
    )
    assert payload == {
        "cvxpy": False,
        "pypower": False,
        "scipy": False,
        "skopt": False,
    }


def test_importing_gd_optimizer_does_not_eagerly_load_optional_stacks():
    payload = _run_python(
        "import sys, json; from homopt.optim import GDOptimizer; "
        "print(json.dumps({"
        "'cvxpy': any(name.startswith('cvxpy') for name in sys.modules), "
        "'pypower': any(name.startswith('pypower') for name in sys.modules), "
        "'scipy': any(name.startswith('scipy') for name in sys.modules), "
        "'skopt': any(name.startswith('skopt') for name in sys.modules), "
        "'name': GDOptimizer.__name__}))"
    )
    assert payload == {
        "cvxpy": False,
        "pypower": False,
        "scipy": False,
        "skopt": False,
        "name": "GDOptimizer",
    }
