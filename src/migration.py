from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from src import (
    Config, Modeling, Model, 
    Seismogram, Wavelet, Geometry
  )

from os import system

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit, prange

OUTPUT_PATH = "data/output/"

uncalled = True

class Migration:

  def __init__(
    self, c: Config, mod: Modeling, model: Model,
    seis: Seismogram, wl: Wavelet, geom: Geometry
  ) -> None:
    
    self.mdl = model
    self.geom = geom
    self.seis = seis
    self.wl = wl
    self.mod = mod
    self.c = c

    shape = (self.mdl.nzz, self.mdl.nxx)

    self.upas = np.zeros(shape)
    self.upre = np.zeros(shape)
    self.ufut = np.zeros(shape)

    self.laplacian = np.zeros(shape)

    self.depas = np.zeros(shape)
    self.depre = np.zeros(shape)
    self.defut = np.zeros(shape)

    self.dh2 = self.c.dh * self.c.dh
    self.inv_dh2 = 1.0 / (5040.0 * self.dh2)
    self.arg = self.c.dt**2 * self.mdl.model_smooth**2

    self.tstop = int(1.7 * (self.c.tlag / self.c.dt))

    if self.c.snap_num_nyquist:
      self.snap_ratio = int(1 / (4 * self.c.fmax * self.c.dt))
    else:
      self.snap_ratio = int(self.c.nt / self.c.snap_num)

    self.nsnaps = int((self.c.nt - self.tstop - 1) / (self.snap_ratio)) + 1
    self.dt_snaps = self.snap_ratio * self.c.dt

    self.snapshots_src = np.zeros((self.nsnaps, self.mdl.nzz, self.mdl.nxx))
    self.snapshots_rec = np.zeros((self.nsnaps, self.mdl.nzz, self.mdl.nxx))

    self.image = np.zeros(shape)
    self.gradient = np.zeros(shape)

    self.ix, self.iz = 0, 0
    self.num, self.den = np.zeros(shape), np.zeros(shape)

    self.snap_id_src = 0
    self.snap_id_rec = self.nsnaps - 1

    self.current = 1

  def rtm(self):

    for isrc in range(len(self.geom.srcxId)):

      self.__zero_out_matrices()

      self.__define_source_coordinates(isrc)

      self.mod.remove_direct_wave_model(self.ix, self.iz)

      for t in range(1, self.c.nt - 1):

        self.__forward_propagation(t)

        self.__get_src_snaps(t)

      for t in range(self.c.nt - 1, self.tstop, -1):

        self.__backward_propagation(t)

        self.__accumulate_cross_correlation(t)

      self.__image_condition()

      self.__show_modeling_status()

    if self.c.is_laplacian:
      self.__laplacian_filter()

    if self.c.save_image:
      self.__save()

  def __zero_out_matrices(self):
    self.seis.seismogram.fill(0.0)

    self.upas.fill(0.0)
    self.upre.fill(0.0)
    self.ufut.fill(0.0)

    self.depas.fill(0.0)
    self.depre.fill(0.0)
    self.defut.fill(0.0)

    self.snap_id_src = 0
    self.snap_id_rec = self.nsnaps - 1

    self.num.fill(0.0)
    self.den.fill(0.0)

  def __define_source_coordinates(self, isrc: int):
    self.ix = int(self.geom.srcxId[isrc]) + self.c.nb
    self.iz = int(self.geom.srczId[isrc]) + self.c.nb

  def __forward_propagation(self, t: int):
      _forward_kernel(
        self.upas, self.upre, self.ufut,
        self.laplacian, self.mod.damp_x,
        self.mod.damp_z, self.inv_dh2,
        self.mdl.nzz, self.mdl.nxx, 
        self.wl.wavelet, self.ix,
        self.iz, self.dh2,
        self.arg, t,
      )

  def __get_src_snaps(self, t: int):
    if t >= self.tstop and not t % self.snap_ratio:
      self.snapshots_src[self.snap_id_src] = self.upre.copy()
      self.snap_id_src += 1

  def __backward_propagation(self, t: int):
    _backward_kernel(
      self.depas, self.depre, self.defut,
      self.laplacian, self.mod.damp_x, self.mod.damp_z,
      self.inv_dh2, self.mdl.nzz, self.mdl.nxx, self.dh2,
      self.arg, self.geom.recx, self.geom.recz, self.c.nb,
      self.seis.seismogram, t
    )

  def __accumulate_cross_correlation(self, t: int, epsilon=1e-9):
    if t % self.snap_ratio:
      idx = int((t - self.tstop) / self.snap_ratio)

      src = self.snapshots_src[idx]
      rec = self.depre

      self.num += src * rec
      #self.den += src * src

  def __image_condition(self):
    self.image += self.dt_snaps * self.num
    #self.image += self.dt_snaps * (self.num / (self.den + 1e-9))

  def __get_rc_snaps(self, t: int):
    if not t % self.snap_ratio:
      self.snapshots_rec[self.snap_id_rec] = self.depre.copy()
      self.snap_id_rec -= 1

  def __laplacian_filter(self):
    inv_dh = 1.0 / (12.0 * self.c.dh * self.c.dh)

    for i in range(2, self.mdl.nzz - 2):
      for j in range(2, self.mdl.nxx - 2):
        d2u_dx2 = (
            - self.image[i-2, j]
            + 16.0 * self.image[i-1, j]
            - 30.0 * self.image[i, j]
            + 16.0 * self.image[i+1, j]
            - self.image[i+2, j]
        ) * inv_dh

        d2u_dz2 = (
            - self.image[i, j-2]
            + 16.0 * self.image[i, j-1]
            - 30.0 * self.image[i, j]
            + 16.0 * self.image[i, j+1]
            - self.image[i, j+2]
        ) * inv_dh

        self.gradient[i, j] = d2u_dx2 + d2u_dz2

    self.image = self.gradient

  def __save(self, path=None):
    if path is None:
      path = (
        OUTPUT_PATH +
          f"image_{self.nsnaps}snaps" +
          f"_{self.c.nx}x{self.c.nz}.bin"
      )

    cropped = self.image[
      self.c.nb:self.c.nb + self.c.nz,
      self.c.nb:self.c.nb + self.c.nx
    ]

    try:
      cropped.flatten('F').astype('float32', order='F').tofile(path)
      print(f"Successfully saved: {path}")

    except OSError as e:
      raise OSError(f"Could not save file: {path}") from e
    
  def __show_modeling_status(self):
    system("clear")
    progress = self.current/len(self.geom.srcxId)
    bar = 10 * "██"
    print(f"\n Shots: {round(100 * progress, 2)}% | {bar[:int((10.0 * progress))]} |")
    self.current += 1

  def plot_snapshots(self, snapshots: np.ndarray) -> None:
    xloc = np.linspace(0, self.c.nx-1, 11, dtype=int)
    xlab = np.array(xloc * self.c.dh, dtype=int)

    zloc = np.linspace(0, self.c.nz-1, 7, dtype=int)
    zlab = np.array(zloc * self.c.dh, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 5))

    ims = []
    for snap in snapshots:
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
      interval=(self.c.nt / len(snapshots) + 1) * self.c.dt * 1e3,
      blit=False,
      repeat_delay=0
    )

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)

    ax.set_yticks(zloc)
    ax.set_yticklabels(zlab)

    plt.show()
    return ani

  def plot(self, perc=99) -> None:
    if self.c.is_laplacian:
      label = "Image Laplacian Filter"
    else:
      label = "Image"

    xloc = np.linspace(0, self.c.nx - 1, 11, dtype=int)
    xlab = np.array(xloc * self.c.dh, dtype=int)

    zloc = np.linspace(0, self.c.nz - 1, 7, dtype=int)
    zlab = np.array(zloc * self.c.dh, dtype=int)

    _, ax = plt.subplots(figsize=(12, 5))

    img_data = self.image[
      self.c.nb:self.c.nb + self.c.nz,
      self.c.nb:self.c.nb + self.c.nx
    ]

    vmin = np.percentile(img_data, 100 - perc)
    vmax = np.percentile(img_data, perc)

    img = ax.imshow(
      img_data,
      aspect="auto",
      cmap="Greys",
      vmin=vmin,
      vmax=vmax
    )

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)
    ax.set_yticks(zloc)
    ax.set_yticklabels(zlab)

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(label)

    plt.colorbar(img, ax=ax)
    plt.show()

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

@njit(parallel=True, fastmath=True)
def _backward_kernel(
  depas: np.ndarray,
  depre: np.ndarray,
  defut: np.ndarray,
  laplacian: np.ndarray,
  damp_x: np.ndarray,
  damp_z: np.ndarray,
  inv_dh2: float,
  nzz: int,
  nxx: int,
  dh2: float,
  arg: np.ndarray,
  recx: np.ndarray,
  recz: np.ndarray,
  nb: int,
  seismogram: np.ndarray,
  t: int,
) -> None:

  for irec in prange(len(recx)):
    rx = int(recx[irec]) + nb
    rz = int(recz[irec]) + nb
    depre[rz, rx] += seismogram[t, irec] / dh2

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      d2u_dx2 = (
        -9.0   * depre[i-4, j] + 128.0   * depre[i-3, j] - 1008.0 * depre[i-2, j] +
        8064.0 * depre[i-1, j] - 14350.0 * depre[i,   j] + 8064.0 * depre[i+1, j] -
        1008.0 * depre[i+2, j] + 128.0   * depre[i+3, j] - 9.0    * depre[i+4, j]
      )

      d2u_dz2 = (
        -9.0   * depre[i, j-4] + 128.0   * depre[i, j-3] - 1008.0 * depre[i, j-2] +
        8064.0 * depre[i, j-1] - 14350.0 * depre[i, j]   + 8064.0 * depre[i, j+1] -
        1008.0 * depre[i, j+2] + 128.0   * depre[i, j+3] - 9.0    * depre[i, j+4]
      )

      laplacian[i, j] = (d2u_dx2 + d2u_dz2) * inv_dh2

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      depas[i, j] = arg[i, j] * laplacian[i, j] + 2.0 * depre[i, j] - defut[i, j]

      damp = damp_x[j] * damp_z[i]

      defut[i, j] = depre[i, j] * damp
      depre[i, j] = depas[i, j] * damp


