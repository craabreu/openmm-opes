import importlib
import subprocess
import sys


def test_package_imports_and_has_version():
    mod = importlib.import_module("openmm_opes")
    assert isinstance(mod.__version__, str)
    assert mod.__version__ == "0.1.0"


def test_kde_and_io_do_not_import_openmm():
    """kde.py and io.py must stay OpenMM-free so most tests run anywhere.

    Runs in a subprocess because openmm may already be imported in-process.
    """
    code = (
        "import sys;"
        "import openmm_opes.kde, openmm_opes.io;"
        "assert 'openmm' not in sys.modules, sorted("
        "m for m in sys.modules if m.startswith('openmm') "
        "and not m.startswith('openmm_opes'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
