from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from src import Config

import numpy as np

class Geometry:
  def __init__(self, c: Config) -> None:
    self.c = c

    self.recx, self.recz     = np.array([]), np.array([])
    self.srcxId, self.srczId = np.array([]), np.array([])

    self.nrec = 0

  def get(self):
    mode = self.c.geometry_mode.upper()
    if mode == "LOAD":
      self.load()
    elif mode == "CREATE":
      self.create()
      self.load()
    else:
      raise KeyError("Choose a valid mode. (create, load)")

  def load(self) -> None:
    receivers = np.loadtxt(self.c.receivers, delimiter=',', skiprows=1)

    if receivers.ndim == 1:
      self.recx = np.array([receivers[1]]) / self.c.dh
      self.recz = np.array([receivers[2]]) / self.c.dh
    else:
      self.recx = receivers[:, 1] / self.c.dh
      self.recz = receivers[:, 2] / self.c.dh

    sources = np.loadtxt(self.c.sources, delimiter=',', skiprows=1)

    if sources.ndim == 1:
      self.srcxId = np.array([sources[1]]) / self.c.dh
      self.srczId = np.array([sources[2]]) / self.c.dh
    else:
      self.srcxId = sources[:, 1] / self.c.dh
      self.srczId = sources[:, 2] / self.c.dh
 
    self.nrec = len(self.recx)

  def create(self) -> None:
      self.nrec = int(self.c.nx / self.c.offset)

      recId = np.arange(1, self.nrec + 1)
      self.recx = np.arange(0, self.nrec) * self.c.offset
      self.recz = np.full(self.nrec, self.c.rec_depth)

      data = np.column_stack((recId, self.recx * self.c.dh, self.recz))

      np.savetxt(
          "data/input/geometry/receivers.txt",
          data,
          fmt="%.0f",
          delimiter=",",
          header="recId, recx, recz",
      )