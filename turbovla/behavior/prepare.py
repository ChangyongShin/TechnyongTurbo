"""Legacy alias; implementation lives in the core TurboVLA modules."""
import importlib
import sys

if __name__ == "__main__":
    from turbovla.data.prepare import main
    main()
else:
    sys.modules[__name__] = importlib.import_module("turbovla.data.prepare")
