from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from src import (
    Config, Modeling, Model, 
    Seismogram, Wavelet, Geometry
  )

from os import system

from src import Migration

import cupy as cp
from numba import njit, prange

OUTPUT_PATH = "data/output/"

uncalled = True

class MigrationGPU(Migration):
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

    self.upas = cp.zeros(shape)
    self.upre = cp.zeros(shape)
    self.ufut = cp.zeros(shape)

    self.laplacian = cp.zeros(shape)

    self.depas = cp.zeros(shape)
    self.depre = cp.zeros(shape)
    self.defut = cp.zeros(shape)

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

    self.snapshots_src = cp.zeros((self.nsnaps, self.mdl.nzz, self.mdl.nxx))

    self.image = cp.zeros(shape)
    self.gradient = cp.zeros(shape)

    self.ix, self.iz = 0, 0
    self.num, self.den = cp.zeros(shape), cp.zeros(shape)

    self.snap_id_src = 0
    self.snap_id_rec = self.nsnaps - 1

    self.current = 1

  def rtm(self):

    for isrc in range(len(self.geom.srcxId)):

      self.zero_out_matrices()

      self.define_source_coordinates(isrc)

      self.mod.remove_direct_wave_model(self.ix, self.iz)

      for t in range(1, self.c.nt - 1):

        self.forward_propagation(t)

        self.get_src_snaps(t)

      for t in range(self.c.nt - 1, self.tstop, -1):

        self.backward_propagation(t)

        self.accumulate_cross_correlation(t)

      self.image_condition()

      self.show_modeling_status()

    if self.c.is_laplacian:
      self.laplacian_filter()

    if self.c.save_image:
      self.save()

  def zero_out_matrices(self):
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

  def define_source_coordinates(self, isrc: int):
    self.ix = int(self.geom.srcxId[isrc]) + self.c.nb
    self.iz = int(self.geom.srczId[isrc]) + self.c.nb

  def forward_propagation(self, t: int):
      _forward_kernel(
        self.upas, self.upre, self.ufut,
        self.laplacian, self.mod.damp_x,
        self.mod.damp_z, self.inv_dh2,
        self.mdl.nzz, self.mdl.nxx, 
        self.wl.wavelet_derivative, self.ix,
        self.iz, self.dh2,
        self.arg, t,
      )

  def get_src_snaps(self, t: int):
    if t >= self.tstop and not t % self.snap_ratio:
      self.snapshots_src[self.snap_id_src] = self.upre.copy()
      self.snap_id_src += 1

  def backward_propagation(self, t: int):
    _backward_kernel(
      self.depas, self.depre, self.defut,
      self.laplacian, self.mod.damp_x, self.mod.damp_z,
      self.inv_dh2, self.mdl.nzz, self.mdl.nxx, self.dh2,
      self.arg, self.geom.recx, self.geom.recz, self.c.nb,
      self.seis.seismogram, t
    )

  def accumulate_cross_correlation(self, t: int, epsilon=1e-9):
    if t % self.snap_ratio:
      idx = int((t - self.tstop) / self.snap_ratio)

      src = self.snapshots_src[idx]
      rec = self.depre

      self.num += src * rec
      #self.den += src * src

  def image_condition(self):
    self.image += self.dt_snaps * self.num
    #self.image += self.dt_snaps * (self.num / (self.den + 1e-9))

  def laplacian_filter(self):
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

  def save(self, path=None):
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
    
  def show_modeling_status(self):
    system("clear")
    progress = self.current/len(self.geom.srcxId)
    bar = 10 * "██"
    print(f"\n Shots: {round(100 * progress, 2)}% | {bar[:int((10.0 * progress))]} |")
    self.current += 1

@njit(parallel=True, fastmath=True)
def _forward_kernel(
  upas: cp.ndarray,
  upre: cp.ndarray,
  ufut: cp.ndarray,
  laplacian: cp.ndarray,
  damp_x: cp.ndarray,
  damp_z: cp.ndarray,
  inv_dh2: float,
  nzz: int,
  nxx: int,
  ricker: cp.ndarray,
  ix: int,
  iz: int,
  dh2: float,
  arg: cp.ndarray,
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
  depas: cp.ndarray,
  depre: cp.ndarray,
  defut: cp.ndarray,
  laplacian: cp.ndarray,
  damp_x: cp.ndarray,
  damp_z: cp.ndarray,
  inv_dh2: float,
  nzz: int,
  nxx: int,
  dh2: float,
  arg: cp.ndarray,
  recx: cp.ndarray,
  recz: cp.ndarray,
  nb: int,
  seismogram: cp.ndarray,
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


