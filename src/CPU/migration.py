from __future__ import annotations

from dataclasses import dataclass, field
from os import system
from typing import TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
  from src import (
    Config,
    Geometry,
    Propagation,
    Model,
    Seismogram,
    Wavelet,
  )

OUTPUT_PATH = "data/output/"

class Migration:
  def __init__(
    self, config: Config, 
    modeling: Propagation, 
    model: Model, 
    seismogram: Seismogram, 
    wavelet: Wavelet, 
    geometry: Geometry
) -> None:

    self.c = config
    self.m = model
    self.g = geometry
    self.md = modeling
    self.s = seismogram
    self.w = wavelet
    
    shape = (model.nzz, model.nxx)

    self.u = Wavefield(shape)  
    self.d = Wavefield(shape)
    self.laplacian = np.zeros(shape)
    
    self.image = np.zeros(shape)
    self.gradient = np.zeros(shape)

    self.num = np.zeros(shape)
    self.den = np.zeros(shape)

    dh2 = config.dh**2
    self.kernel_args = KernelArguments(
      dh2=dh2,
      inv_dh2=1.0 / (5040.0 * dh2),
      velocity_term=config.dt**2 * model.model_smooth**2
    )

    self.snaps = SnapshotManager.from_config(config, shape)
    
    self.ix, self.iz = 0, 0
    self.current_step = 1

  def rtm(self):

    for isrc in range(len(self.g.srcxId)):

      self.__zero_out_matrices()

      self.__define_source_coordinates(isrc)

      self.md.remove_direct_wave_model(self.ix, self.iz)

      for t in range(1, self.c.nt - 1):

        self.__forward_propagation(t)

        self.__get_src_snaps(t)

      for t in range(self.c.nt - 1, self.snaps.tstop, -1):

        self.__backward_propagation(t)

        self.__accumulate_cross_correlation(t)

      self.__image_condition()

      self.__show_modeling_status()

    if self.c.is_laplacian:
      self.__laplacian_filter()

    if self.c.save_image:
      self.__save()

  def __zero_out_matrices(self):
    self.s.seismogram.fill(0.0)

    self.u.past.fill(0.0)
    self.u.present.fill(0.0)
    self.u.future.fill(0.0)

    self.d.past.fill(0.0)
    self.d.present.fill(0.0)
    self.d.future.fill(0.0)

    self.snaps.current_src_id = 0
    self.snaps.current_rec_id = self.snaps.nsnaps - 1

    self.num.fill(0.0)
    self.den.fill(0.0)

  def __define_source_coordinates(self, isrc: int):
    self.ix = int(self.g.srcxId[isrc]) + self.c.nb
    self.iz = int(self.g.srczId[isrc]) + self.c.nb

  def __forward_propagation(self, t: int):
    _forward_kernel(
      self.u.past, self.u.present, self.u.future,
      self.laplacian, self.md.damp_x,
      self.md.damp_z, self.kernel_args.inv_dh2,
      self.m.nzz, self.m.nxx, 
      self.w.wavelet, self.ix,
      self.iz, self.kernel_args.dh2,
      self.kernel_args.velocity_term, t,
    )

  def __get_src_snaps(self, t: int):
    if t >= self.snaps.tstop and not t % self.snaps.ratio:
      self.snaps.src[self.snaps.current_src_id] = self.u.present.copy()
      self.snaps.current_src_id += 1

  def __backward_propagation(self, t: int):
    _backward_kernel(
      self.d.past, self.d.present, self.d.future,
      self.laplacian, self.md.damp_x, self.md.damp_z,
      self.kernel_args.inv_dh2, self.m.nzz, self.m.nxx, self.kernel_args.dh2,
      self.kernel_args.velocity_term, self.g.recx, self.g.recz, self.c.nb,
      self.s.seismogram, t
    )

  def __accumulate_cross_correlation(self, t: int, epsilon=1e-9):
    if t % self.snaps.ratio:
      idx = int((t - self.snaps.tstop) / self.snaps.ratio)

      src = self.snaps.src[idx]
      rec = self.d.present

      self.num += src * rec
      #self.den += src * src

  def __image_condition(self):
    self.image += self.snaps.dt * self.num
    #self.image += self.dt_snaps * (self.num / (self.den + 1e-9))

  def __get_rc_snaps(self, t: int):
    if not t % self.snaps.ratio:
      self.snaps.rec[self.snaps.current_rec_id] = self.d.present.copy()
      self.snaps.current_rec_id -= 1

  def __laplacian_filter(self):
    inv_dh = 1.0 / (12.0 * self.c.dh * self.c.dh)

    for i in range(2, self.m.nzz - 2):
      for j in range(2, self.m.nxx - 2):
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
          f"image_{self.snaps.nsnaps}snaps" +
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
    progress = self.current_step / len(self.g.srcxId)
    bar = 10 * "██"
    print(f"\n Shots: {round(100 * progress, 2)}% | {bar[:int((10.0 * progress))]} |")
    self.current_step += 1

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
        self.m.model[self.c.nb:self.c.nb+self.c.nz,
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

    fig, ax = plt.subplots(figsize=(12, 5)) 

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


@dataclass(slots=True)
class Wavefield:
  shape: Tuple[int, int]
  past: np.ndarray = field(init=False)
  present: np.ndarray = field(init=False)
  future: np.ndarray = field(init=False)

  def __post_init__(self):
    self.past = np.zeros(self.shape)
    self.present = np.zeros(self.shape)
    self.future = np.zeros(self.shape)


@dataclass(slots=True)
class KernelArguments:
  dh2: float
  inv_dh2: float
  velocity_term: np.ndarray  


@dataclass(slots=True)
class SnapshotManager:
  nsnaps: int
  tstop: int
  ratio: int
  dt: float
  src: np.ndarray
  rec: np.ndarray
  
  # internal counters
  current_src_id: int = 0
  current_rec_id: int = 0

  @classmethod
  def from_config(cls, c, shape):
    tstop = int(1.7 * (c.tlag / c.dt))
    
    if c.snap_num_nyquist:
      ratio = int(1 / (4 * c.fmax * c.dt))
    else:
      ratio = int(c.nt / c.snap_num)

    nsnaps = int((c.nt - tstop - 1) / ratio) + 1
    
    return cls(
      nsnaps=nsnaps,
      tstop=tstop,
      ratio=ratio,
      dt=ratio * c.dt,
      src=np.zeros((nsnaps, *shape)),
      rec=np.zeros((nsnaps, *shape)),
      current_rec_id=nsnaps - 1
    )

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


