from .cfg import Config

from .migration import Migration

from .modeling import Modeling

from .model import Model

from .wavelet import Wavelet

from .seismogram import Seismogram

from .geometry import Geometry

from .plots import Plotting

def measure_runtime(func):
  import time
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Function {func.__name__} took: {round(end - start, 4)} seconds")
    return result

  return wrapper

__all__ = [
    "Config", "Migration",
    "Model", "Geometry",
    "Seismogram", "measure_runtime",
    "Modeling", "Wavelet", "Plotting"
]