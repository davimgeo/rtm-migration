from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from . import Config

import numpy as np

class Geometry:
  def __init__(self, c: Config) -> None:
    self.c = c

    self.recx, self.recz     = np.array([]), np.array([])
    self.srcxId, self.srczId = np.array([]), np.array([])

    self.nrec, self.nsrc = 0, 0

    self.direct_wave = np.array([])

  def get(self):
    mode = self.c.geometry_mode.upper()
    if mode == "LOAD":
      self.load()
    elif mode == "CREATE":
      self.create()
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
    self.createReceivers()
    self.createSources()
     
    self.save() 

  def createReceivers(self):
    self.nrec = int(self.c.nx_geom / self.c.offset)

    self.recx = np.arange(0, self.nrec) * self.c.offset
    self.recz = np.full(self.nrec, self.c.rec_depth)

  def createSources(self):
    self.nsrc = len(self.c.sources_create) 

    self.srcxId = [src / self.c.dh for src in self.c.sources_create]
    self.srczId = np.full(self.nsrc, self.c.src_depth)

  def save(self):
    print(self.c.save_create)
    if self.c.save_create:

      recId = np.arange(1, self.nrec + 1)
      srcId = np.arange(1, self.nsrc + 1)

      recCompensatedbyGrid = [rec * self.c.dh for rec in self.recx]
      srcCompensatedbyGrid = [src * self.c.dh for src in self.srcxId]

      receivers = np.column_stack((recId, recCompensatedbyGrid, self.recz))
      sources = np.column_stack((srcId, srcCompensatedbyGrid, self.srczId))

      files = [
        ("data/input/geometry/receivers_new.txt", receivers, "recId, recx, recz"),
        ("data/input/geometry/sources_new.txt",  sources,   "srcId, srcxId, srczId"),
      ]

      for path, data, header in files:
        np.savetxt(
          path,
          data,
          fmt="%.0f",
          delimiter=",",
          header=header,
        )
