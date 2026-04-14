from typing import TYPE_CHECKING

from .cfg import Config

_engine = Config().load().engine

match _engine:
  case "GPU":
    from . import GPU as _backend
  case _:
    if _engine != "CPU": print(f"Unknown engine {_engine} switching to CPU")
    from . import CPU as _backend

# just to calm down the lsp
if TYPE_CHECKING:
    from . import CPU as _cpu_backend

    _backend = _cpu_backend

Config = Config
Migration = _backend.Migration 
Model = _backend.Model
Geometry = _backend.Geometry
Seismogram = _backend.Seismogram
Propagation = _backend.Propagation 
Wavelet = _backend.Wavelet 

from .plots import Plotting
from .utils import measure_runtime

__all__ = [
  "Config",
  "Migration",
  "Model",
  "Geometry",
  "Seismogram",
  "Propagation",
  "Wavelet",
  "Plotting",
  "measure_runtime",
]
