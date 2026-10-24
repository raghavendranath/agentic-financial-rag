"""Load the preserved v2 policy without modifying it or sharing its Python globals."""
import importlib
import importlib.util
import sys
from pathlib import Path
from .core import digest

DIRECTORY = Path(__file__).resolve().parents[2] / "baseline/src/finrag"


def baseline_hash():
    return digest({p.name: p.read_text() for p in sorted(DIRECTORY.glob("*.py"))})


def policy():
    name = "_finrag_frozen_v2"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, DIRECTORY / "__init__.py",
                                                     submodule_search_locations=[str(DIRECTORY)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name + ".experiment")
