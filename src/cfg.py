import tomllib
from dataclasses import dataclass

@dataclass
class Parameters:
  debug = False

  receivers = ""
  sources = ""
  offset = 0.0

  nx = 0
  nz = 0

  nb = 0
  factor = 0.015

  dh = 0

  interfaces = []

  velocity_interfaces = []

  tlag = 0.0

  save_seismogram = False

  perc = 99

  seismogram_input_path = ""
  seismogram_output_path = ""
  nt   = 0
  dt   = 0.0

  fmax = 0
  
  snap_path = ""
  snap_num  = 0
  snap_bool = False

class Config(Parameters):
  def __init__(self, toml_path: str):
    super().__init__()

    self.toml_path = toml_path 

  def load(self):
    with open(self.toml_path, "rb") as f:
      data = tomllib.load(f)

      if self.debug: print(data)

    for key, value in data.items():
      setattr(self, key, value)

    return self



