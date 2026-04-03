from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from src import Config, Geometry

import matplotlib.pyplot as plt
import numpy as np

class Model:
  def __init__(self, c: Config, geom: Geometry) -> None:
    self.c = c
    self.geom = geom

    self.nxx = 2*self.c.nb + self.c.nx
    self.nzz = 2*self.c.nb + self.c.nz

    self.model = np.zeros((self.c.nz, self.c.nx))
    self.model_smooth = np.zeros((self.c.nz, self.c.nx))

  def get(self) -> None:
    mode = self.c.model_mode.upper()
    if mode == "LOAD":
      self.load()

    elif mode == "CREATE":
      self.create()
    else:
      raise KeyError("Choose a valid mode. (create, load)")

  def load(self) -> None:
    self.model = np.fromfile(
      self.c.model_path, dtype=np.float32, count=self.c.nx*self.c.nz
    ).reshape([self.c.nz, self.c.nx], order='F')

  def create(self) -> None:
    if not len(self.c.interfaces): 
      self.model[:, :] = self.c.value_interfaces[0] 

    else: 
      self.model[:self.c.interfaces[0], :] = self.c.value_interfaces[0]

    for layer, velocity in enumerate(self.c.value_interfaces[1:]):
      self.model[self.c.interfaces[layer]:, :] = velocity

  def set_boundary(self) -> None:
    model_ext = np.zeros((self.nzz, self.nxx))

    for j in range(self.c.nx):
      for i in range(self.c.nz):
        model_ext[i+self.c.nb, j+self.c.nb] = self.model[i, j]

    for j in range(self.c.nb, self.c.nx+self.c.nb):
      for i in range(self.c.nb):
        model_ext[i, j] = model_ext[self.c.nb, j]
        model_ext[self.c.nz+self.c.nb+i, j] = model_ext[self.c.nz+self.c.nb-1, j]

    for i in range(self.nzz):
      for j in range(self.c.nb):
        model_ext[i, j] = model_ext[i, self.c.nb]
        model_ext[i, self.c.nx+self.c.nb+j] = model_ext[i, self.c.nx+self.c.nb-1]

    self.model = model_ext

  def gaussian_smooth(self, sigma: float, truncate: float = 4.0):
    self.model_smooth = self.__gaussian_filter(
      self.model.copy(), sigma=sigma, truncate=truncate
    )

  def __gaussian_filter(
      self, arr: np.ndarray, 
      sigma: float, 
      truncate: float
    ) -> np.ndarray:
      
      radius = int(truncate * sigma + 0.5)

      x = np.arange(-radius, radius + 1)
      gaussian = np.exp(-0.5 * (x**2) / (sigma**2))

      gaussian /= gaussian.sum()

      for col in range(arr.shape[1]):
        data = arr[:, col]

        padded = np.pad(data, radius, mode="reflect")

        smoothed = np.convolve(padded, gaussian, mode="same")

        arr[:, col] = smoothed[radius:-radius]
      
      return arr

  def plot(self, model=None):
      if model is None:
        model = self.model

      xloc = np.linspace(0, self.c.nx - 1, 11, dtype=int)
      xlab = np.array(xloc * self.c.dh, dtype=int)

      zloc = np.linspace(0, self.c.nz - 1, 7, dtype=int)
      zlab = np.array(zloc * self.c.dh, dtype=int)

      fig, ax = plt.subplots(figsize=(12, 5))

      img = ax.imshow(
          model[
              self.c.nb:self.c.nb + self.c.nz,
              self.c.nb:self.c.nb + self.c.nx
          ],
          aspect="auto",
          cmap="jet",
      )

      ax.plot(self.geom.recx, self.geom.recz, 'bv', label="Receivers")
      ax.plot(self.geom.srcxId, self.geom.srczId, 'r*', markersize=12, label="Source")

      ax.set_xticks(xloc)
      ax.set_xticklabels(xlab)
      ax.set_yticks(zloc)
      ax.set_yticklabels(zlab)

      ax.set_xlabel("Distance [m]")
      ax.set_ylabel("Depth [m]")
      ax.set_title("Velocity Model")

      plt.colorbar(img, ax=ax, label="VP [m/s]")
      ax.legend()

      plt.show()