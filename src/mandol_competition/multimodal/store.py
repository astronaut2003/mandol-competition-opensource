"""Legacy compatibility only; formal persistence lives in mandol_substrate."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("mandol_competition.multimodal.compat.store")
