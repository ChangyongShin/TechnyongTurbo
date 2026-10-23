"""Legacy alias; implementation lives in the core TurboVLA modules."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("turbovla.data.dataset")
