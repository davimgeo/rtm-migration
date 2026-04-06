from .cfg import Config

_engine = Config().load().engine

if _engine == "GPU":
  from . import GPU as _backend
elif _engine == "CPU":
  from . import CPU as _backend
else:
  raise ValueError(f"Unknown engine: {_engine}")

Config = Config
Migration = _backend.Migration 
Model = _backend.Model
Geometry = _backend.Geometry
Seismogram = _backend.Seismogram
Propagation = _backend.Propagation 
Wavelet = _backend.Wavelet 

from .utils import measure_runtime

__all__ = [
  "Config",
  "Migration",
  "Model",
  "Geometry",
  "Seismogram",
  "Propagation",
  "Wavelet",
  "measure_runtime",
]
