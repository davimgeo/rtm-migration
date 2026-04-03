from __future__ import annotations
from os import system

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from src import (
    Config, Model, Seismogram, 
    Wavelet, Geometry
  )

import numpy as np
from matplotlib import animation
from numba import njit, prange

class Modeling:

  def __init__(
      self, c: Config, mdl: Model, geom: Geometry,
      seis: Seismogram, wl: Wavelet
    ) -> None:

    self.c = c
    self.seis = seis
    self.mdl = mdl
    self.geom = geom
    self.wl = wl

    shape = (self.mdl.nzz, self.mdl.nxx)

    self.upas = np.zeros(shape)
    self.upre = np.zeros(shape)
    self.ufut = np.zeros(shape)

    self.laplacian = np.zeros(shape)

    self.dh2 = self.c.dh * self.c.dh
    self.inv_dh2 = 1.0 / (5040.0 * self.dh2)
    self.arg = self.c.dt **2 * self.mdl.model**2

    self.upas_homo = np.zeros(shape)
    self.upre_homo = np.zeros(shape)
    self.ufut_homo = np.zeros(shape)

    self.laplacian_homo = np.zeros(shape)

    model_homo = np.full(shape, 1500)
    self.arg2 = self.c.dt **2 * model_homo**2

    self.damp_x = np.zeros((self.mdl.nxx))
    self.damp_z = np.zeros((self.mdl.nzz))

    self.nsnaps = 101
    self.snap_ratio = int((self.c.nt - 1) / self.nsnaps) + 1

    self.snapshots = np.zeros((self.nsnaps, self.mdl.nzz, self.mdl.nxx))

    self.ix, self.iz = 0, 0
    self.rx = self.geom.recx.astype(int) + self.c.nb
    self.rz = self.geom.recz.astype(int) + self.c.nb
    self.snap_id_src = 0
    self.current = 1

  def fdm_propagation(self, ix: int, iz: int, isSnap=False) -> None:
      self.ix, self.iz = ix, iz

      for t in range(1, self.c.nt - 1):

        _forward_kernel(
          self.upas, self.upre, self.ufut, self.laplacian,
          self.damp_x, self.damp_z, self.inv_dh2,
          self.mdl.nzz, self.mdl.nxx, self.wl.wavelet,
          self.ix, self.iz, self.dh2, self.arg, t
        )

        self.__get_seismogram(self.seis.seismogram, self.upre, t)

        self.get_snapshots(t, isSnap)

      self.upas.fill(0.0)
      self.upre.fill(0.0)
      self.ufut.fill(0.0)

  def get_snapshots(self, t: int, isSnap: bool):
    if isSnap and not t % self.snap_ratio:
      self.snapshots[self.snap_id_src] = self.upre.copy()
      self.snap_id_src += 1   

  def remove_direct_wave_offset(self, ix: int, iz: int) -> None:
      self.ix, self.iz = ix, iz

      for t in range(1, self.c.nt - 1):

        _forward_kernel(
          self.upas, self.upre, self.ufut, self.laplacian,
          self.damp_x, self.damp_z, self.inv_dh2,
          self.mdl.nzz, self.mdl.nxx, self.wl.wavelet,
          self.ix, self.iz, self.dh2, self.arg, t
        )

        self.__get_seismogram(self.seis.seismogram, self.upre, t)

      self.seis.remove_direct_wave(self.ix, self.iz)

  def remove_direct_wave_model(self, ix: int, iz: int) -> None:
      self.ix, self.iz = ix, iz
      
      self.zero_out_matrices()

      for t in range(1, self.c.nt - 1):

        _forward_kernel(
          self.upas, self.upre, self.ufut, self.laplacian,
          self.damp_x, self.damp_z, self.inv_dh2,
          self.mdl.nzz, self.mdl.nxx, self.wl.wavelet,
          self.ix, self.iz, self.dh2, self.arg, t
        )

        self.__get_seismogram(self.seis.seismogram, self.upre, t)

        _forward_kernel(
          self.upas_homo, self.upre_homo, self.ufut_homo, 
          self.laplacian_homo, self.damp_x, self.damp_z, 
          self.inv_dh2, self.mdl.nzz, self.mdl.nxx, 
          self.wl.wavelet, self.ix, self.iz, self.dh2, self.arg2, t
        )

        self.__get_seismogram(self.seis.seismogram_homo, self.upre_homo, t)

      self.seis.seismogram -= self.seis.seismogram_homo

  def zero_out_matrices(self):
    self.seis.seismogram.fill(0.0)
    self.seis.seismogram_homo.fill(0.0)

    self.upas.fill(0.0)
    self.upre.fill(0.0)
    self.ufut.fill(0.0)
    self.upas_homo.fill(0.0)
    self.upre_homo.fill(0.0)
    self.ufut_homo.fill(0.0)

  def __get_seismogram(self, seismogram: np.ndarray, upre: np.ndarray, t: int) -> None:
    for irec in range(len(self.geom.recx)):
      rx = int(self.geom.recx[irec]) + self.c.nb
      rz = int(self.geom.recz[irec]) + self.c.nb
      seismogram[t, irec] = upre[rz, rx]

  def get_damp(self):
    for i in range(self.mdl.nzz):

      if self.c.nb <= i < self.c.nb + self.c.nz:
          self.damp_z[i] = 1.0

      elif i < self.c.nb:
          d = self.c.nb - i
          self.damp_z[i] = np.exp(-(self.c.factor * d) * (self.c.factor * d))

      else:
          d = i - (self.c.nb + self.c.nz - 1)
          self.damp_z[i] = np.exp(-(self.c.factor * d) * (self.c.factor * d))

    for j in range(self.mdl.nxx):

      if self.c.nb <= j < self.c.nb + self.c.nx:
          self.damp_x[j] = 1.0

      elif j < self.c.nb:
          d = self.c.nb - j
          self.damp_x[j] = np.exp(-(self.c.factor * d) * (self.c.factor * d))

      else:
          d = j - (self.c.nb + self.c.nx - 1)
          self.damp_x[j] = np.exp(-(self.c.factor * d) * (self.c.factor * d))

  def show_modeling_status(self):
    system("clear")
    progress = self.current/len(self.geom.srcxId)
    bar = 10 * "██"
    print(f"\n Shots: {100 * progress}% | {bar[:int((10.0 * progress))]} |")
    self.current += 1

  def plot_snapshots(self) -> None:
    xloc = np.linspace(0, self.c.nx-1, 11, dtype=int)
    xlab = np.array(xloc * self.c.dh, dtype=int)

    zloc = np.linspace(0, self.c.nz-1, 7, dtype=int)
    zlab = np.array(zloc * self.c.dh, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 5))

    ims = []
    for snap in self.snapshots:
      scale = 2.0 * np.std(snap)

      model_frame = ax.imshow(
        self.mdl.model[self.c.nb:self.c.nb+self.c.nz,
                             self.c.nb:self.c.nb+self.c.nx],
        aspect="auto", cmap="jet", alpha=0.5
      )

      snap_frame = ax.imshow(
        snap[self.c.nb:self.c.nb+self.c.nz,
             self.c.nb:self.c.nb+self.c.nx],
        aspect="auto", cmap="Greys",
        vmin=-scale, vmax=scale, alpha=0.7
      )

      ax.plot(self.geom.recx, self.geom.recz, 'bv')
      ax.plot(self.geom.srcxId, self.geom.srczId, 'r*')

      ims.append([model_frame, snap_frame])

    ani = animation.ArtistAnimation(
      fig, ims,
      interval=(self.c.nt / len(self.snapshots) + 1) * self.c.dt * 1e3,
      blit=False,
      repeat_delay=0
    )

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)

    ax.set_yticks(zloc)
    ax.set_yticklabels(zlab)

    plt.show()
    return ani

@njit(parallel=True, fastmath=True)
def _forward_kernel(
  upas: np.ndarray,
  upre: np.ndarray,
  ufut: np.ndarray,
  laplacian: np.ndarray,
  damp_x: np.ndarray,
  damp_z: np.ndarray,
  inv_dh2: float,
  nzz: int,
  nxx: int,
  ricker: np.ndarray,
  ix: int,
  iz: int,
  dh2: float,
  arg: np.ndarray,
  t: int,
) -> None:

  upre[iz, ix] += ricker[t] / dh2

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      d2u_dx2 = (
        -9.0   * upre[i-4, j] + 128.0   * upre[i-3, j] - 1008.0 * upre[i-2, j] +
        8064.0 * upre[i-1, j] - 14350.0 * upre[i,   j] + 8064.0 * upre[i+1, j] -
        1008.0 * upre[i+2, j] + 128.0   * upre[i+3, j] - 9.0    * upre[i+4, j]
      )

      d2u_dz2 = (
        -9.0   * upre[i, j-4] + 128.0   * upre[i, j-3] - 1008.0 * upre[i, j-2] +
        8064.0 * upre[i, j-1] - 14350.0 * upre[i, j]   + 8064.0 * upre[i, j+1] -
        1008.0 * upre[i, j+2] + 128.0   * upre[i, j+3] - 9.0    * upre[i, j+4]
      )

      laplacian[i, j] = (d2u_dx2 + d2u_dz2) * inv_dh2

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):

      upas[i, j] = arg[i, j] * laplacian[i, j] + 2.0 * upre[i, j] - ufut[i, j]
      
      damp = damp_x[j] * damp_z[i]

      ufut[i, j] = upre[i, j] * damp
      upre[i, j] = upas[i, j] * damp