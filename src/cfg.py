import tomllib

from dataclasses import dataclass, field, fields
from typing import List

DEFAULT_PATH = "config/parameters.toml"

@dataclass
class Parameters:

  # General 
  debug: bool = field(default=False, metadata={"toml": "General.debug"})
  engine: str = field(default="", metadata={"toml": "General.engine"})

  # RTM 
  save_image: bool = field(default=False, metadata={"toml": "RTM.save"})
  is_laplacian: bool = field(default=False, metadata={"toml": "RTM.laplacian"})

  # Modeling
  dh: float = field(default=0.0, metadata={"toml": "Modeling.dh"})
  nb: int = field(default=0, metadata={"toml": "Modeling.Cerjan.nb"})
  factor: float = field(default=0.0, metadata={"toml": "Modeling.Cerjan.factor"})

  # Seismogram 
  seismogram_mode: str = field(default="generate", metadata={"toml": "Seismogram.mode"})
  load_seis_path: str = field(default="", metadata={"toml": "Seismogram.load.path"})
  nt: int = field(default=0, metadata={"toml": "Seismogram.parameters.nt"})
  dt: float = field(default=0.0, metadata={"toml": "Seismogram.parameters.dt"})
  perc: int = field(default=99, metadata={"toml": "Seismogram.plot.perc"})

  # Model 
  model_mode: str = field(default="create", metadata={"toml": "Model.mode"})
  model_path: str = field(default="", metadata={"toml": "Model.load.path"})
  nx_load: int = field(default=0, metadata={"toml": "Model.load.nx"})
  nz_load: int = field(default=0, metadata={"toml": "Model.load.nz"})
  nx: int = field(default=0, metadata={"toml": "Model.create.nx"})
  nz: int = field(default=0, metadata={"toml": "Model.create.nz"})
  interfaces: List[int] = field(default_factory=list, metadata={"toml": "Model.create.interfaces"})
  value_interfaces: List[float] = field(default_factory=list, metadata={"toml": "Model.create.layerValues"})

  # Geometry
  geometry_mode: str = field(default="load", metadata={"toml": "Geometry.mode"})
  receivers: str = field(default="", metadata={"toml": "Geometry.load.receivers"})
  sources: str = field(default="", metadata={"toml": "Geometry.load.sources"})
  offset: float = field(default=0.0, metadata={"toml": "Geometry.create.offset"})
  save_create: bool = field(default=False, metadata={"toml": "Geometry.load.save"})
  rec_depth: float = field(default=0.0, metadata={"toml": "Geometry.create.receiversDepth"})
  src_depth: float = field(default=0.0, metadata={"toml": "Geometry.create.sourcesDepth"})
  sources_create: List[int] = field(default_factory=list, metadata={"toml": "Geometry.create.sources"})

  # Wavelet
  fmax: float = field(default=0.0, metadata={"toml": "Wavelet.fmax"})
  tlag: float = field(default=0.0, metadata={"toml": "Wavelet.tlag"})

  # Snapshots
  snap_num_nyquist: bool = field(default=False, metadata={"toml": "Snapshots.numNyquist"})
  snap_num: int = field(default=0, metadata={"toml": "Snapshots.number"})

class Config(Parameters):
  def __init__(self, toml_path: str | None = None):
    super().__init__()

    self.toml_path = toml_path or DEFAULT_PATH

  def load(self):
    with open(self.toml_path, "rb") as f:
      data = tomllib.load(f)

    for f in fields(self):
      metadata = f.metadata.get("toml")

      try:
        value = self.get_nested(data, metadata)
        setattr(self, f.name, value)
      except KeyError:
        pass

    self.manage_modes_attributes()

    return self
  
  def get_nested(self, root: dict, path: str) -> dict:
    """
    Loop through a nested dictionary and returns the value of the node
    """
    node = root
    for key in path.split("."):
      node = node[key]
    return node

  def manage_modes_attributes(self):
    if self.model_mode.upper() == "LOAD":
      self.nx = self.nx_load
      self.nz = self.nz_load
