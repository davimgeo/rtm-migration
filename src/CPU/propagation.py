from __future__ import annotations
from os import system

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

from src.utils import measure_runtime

if TYPE_CHECKING:
  from . import (
    Config, Model, Seismogram, 
    Wavelet, Geometry
  )

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit, prange

class Propagation:

  def __init__(
      self, config: Config, model: Model, geometry: Geometry,
      seismogram: Seismogram, wavelet: Wavelet
    ) -> None:

    self.c = config
    self.m = model
    self.g = geometry
    self.s = seismogram
    self.w = wavelet

    shape = (model.nzz, model.nxx)

    self.u = Wavefield(shape)
    self.u_homo = Wavefield(shape)

    self.laplacian = np.zeros(shape)
    self.laplacian_homo = np.zeros(shape)

    model_homo = np.full(shape, 1500.0)

    self.kernel_arg = KernelArguments.make(
        config.dh, config.dt, model.model
      )
    self.kernel_arg_homo = KernelArguments.make(
        config.dh, config.dt, model_homo
    )

    self.damp_x = np.zeros(model.nxx)
    self.damp_z = np.zeros(model.nzz)

    self.nsnaps = 101
    self.snap_ratio = int((config.nt - 1) / self.nsnaps) + 1

    self.snapshots = np.zeros((self.nsnaps, model.nzz, model.nxx))

    self.ix, self.iz = 0, 0
    self.snap_id_src = 0

  def fdm_propagation(self, ix: int, iz: int, isSnap=False) -> None:
      self.ix, self.iz = ix, iz

      for t in range(1, self.c.nt - 1):

        _forward_kernel(
          self.u.past, self.u.present, self.u.future, self.laplacian,
          self.damp_x, self.damp_z, self.kernel_arg.inv_dh2,
          self.m.nzz, self.m.nxx, self.w.wavelet,
          self.ix, self.iz, self.kernel_arg.dh2, 
          self.kernel_arg.velocity_term, t
        )

        self.__get_seismogram(self.s.seismogram, self.u.present, t)

        self.get_snapshots(t, isSnap)

      self.u.past.fill(0.0)
      self.u.present.fill(0.0)
      self.u.future.fill(0.0)

  def get_snapshots(self, t: int, isSnap: bool):
    if isSnap and not t % self.snap_ratio:
      self.snapshots[self.snap_id_src] = self.u.present.copy()
      self.snap_id_src += 1   

  def remove_direct_wave_offset(self, ix: int, iz: int) -> None:
      self.ix, self.iz = ix, iz

      for t in range(1, self.c.nt - 1):

        _forward_kernel(
          self.u.past, self.u.present, self.u.future, self.laplacian,
          self.damp_x, self.damp_z, self.kernel_arg.inv_dh2,
          self.m.nzz, self.m.nxx, self.w.wavelet,
          self.ix, self.iz, self.kernel_arg.dh2, 
          self.kernel_arg.velocity_term, t
        )

        self.__get_seismogram(self.s.seismogram, self.u.present, t)

      self.s.remove_direct_wave(self.ix, self.iz)

  def remove_direct_wave_model(self, ix: int, iz: int) -> None:
      self.ix, self.iz = ix, iz
      
      self.zero_out_matrices()

      for t in range(1, self.c.nt - 1):

        self.forward_propagation(
          self.u,
          self.kernel_arg,
          t
        )

        self.__get_seismogram(self.s.seismogram, self.u.present, t)

        self.forward_propagation(
          self.u_homo,
          self.kernel_arg_homo,
          t
        )

        self.__get_seismogram(self.s.seismogram_homo, self.u_homo.present, t)

      self.s.seismogram -= self.s.seismogram_homo

  def zero_out_matrices(self):
    self.s.seismogram.fill(0.0)
    self.s.seismogram_homo.fill(0.0)

    self.u.past.fill(0.0)
    self.u.present.fill(0.0)
    self.u.future.fill(0.0)
    self.u_homo.past.fill(0.0)
    self.u_homo.present.fill(0.0)
    self.u_homo.future.fill(0.0)

  def forward_propagation(
      self,
      u_field: Wavefield,
      kernel_arg: KernelArguments,
      t: int
  ) -> None:

    _forward_kernel(
      u_field.past, u_field.present,
      u_field.future, self.damp_x, self.damp_z, 
      kernel_arg.inv_dh2, self.m.nzz, 
      self.m.nxx, self.w.wavelet,
      self.ix, self.iz, kernel_arg.dh2,
      kernel_arg.velocity_term, t
    )

  def __get_seismogram(self, seismogram: np.ndarray, upre: np.ndarray, t: int) -> None:
    for irec in range(len(self.g.recx)):
      rx = int(self.g.recx[irec]) + self.c.nb
      rz = int(self.g.recz[irec]) + self.c.nb
      seismogram[t, irec] = upre[rz, rx]

  def get_damp(self):
    for i in range(self.m.nzz):

      if self.c.nb <= i < self.c.nb + self.c.nz:
          self.damp_z[i] = 1.0

      elif i < self.c.nb:
          d = self.c.nb - i
          self.damp_z[i] = np.exp(-(self.c.factor * d) * (self.c.factor * d))

      else:
          d = i - (self.c.nb + self.c.nz - 1)
          self.damp_z[i] = np.exp(-(self.c.factor * d) * (self.c.factor * d))

    for j in range(self.m.nxx):

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
    progress = self.current/len(self.g.srcxId)
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

      ax.plot(self.g.recx, self.g.recz, 'bv')
      ax.plot(self.g.srcxId, self.g.srczId, 'r*')

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

@dataclass(slots=True)
class Wavefield:
  shape: Tuple[int, int]
  past: np.ndarray = field(init=False)
  present: np.ndarray = field(init=False)
  future: np.ndarray = field(init=False)

  def __post_init__(self):
    self.past = np.zeros(self.shape, dtype=np.float32)
    self.present = np.zeros(self.shape, dtype=np.float32)
    self.future = np.zeros(self.shape, dtype=np.float32)


@dataclass(slots=True)
class KernelArguments:
  dh2: float
  inv_dh2: float
  velocity_term: np.ndarray

  @classmethod
  def make(cls, dh, dt, model):
    dh2 = dh ** 2
    return cls(
      dh2=dh2,
      inv_dh2=1.0 / (5040.0 * dh2),
      velocity_term=dt**2 * model**2,
    )

@njit(parallel=True, fastmath=True)
def _forward_kernel(
  upas: np.ndarray,
  upre: np.ndarray,
  ufut: np.ndarray,
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

      laplacian = (d2u_dx2 + d2u_dz2) * inv_dh2

      upas[i, j] = arg[i, j] * laplacian + 2.0 * upre[i, j] - ufut[i, j]
    
  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      damp = damp_x[j] * damp_z[i]
  
      ufut[i, j] = upre[i, j] * damp
      upre[i, j] = upas[i, j] * damp
