from __future__ import annotations
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit, prange

OUTPUT_PATH = "data/output/"

class Migration:
  def __init__(self, model: Model, geom: Geometry, seis: Seismogram, c):
    self.mdl = model
    self.geom = geom
    self.seis = seis
    self.c = c

    self.damp2D = np.ones((self.mdl.nzz, self.mdl.nxx))

    self.ricker = np.zeros(self.c.nt)

    self.upas = np.zeros((self.mdl.nzz, self.mdl.nxx))
    self.upre = np.zeros((self.mdl.nzz, self.mdl.nxx))
    self.ufut = np.zeros((self.mdl.nzz, self.mdl.nxx))

    self.depas = np.zeros((self.mdl.nzz, self.mdl.nxx))
    self.depre = np.zeros((self.mdl.nzz, self.mdl.nxx))
    self.defut = np.zeros((self.mdl.nzz, self.mdl.nxx))

    self.snapshots_src = []
    self.snapshots_rec = []
    self.snap_ratio = 10

    self.image = np.zeros((self.mdl.nzz, self.mdl.nxx))
    self.gradient = np.zeros((self.mdl.nzz, self.mdl.nxx))

    # these change over time, bad practice it seems
    # ask rodrigo
    self.snap_id = 0
    self.rec_snap_id = 0

    self.ix, self.iz = 0, 0

  def rtm(self):
    d2u_dx2 = np.zeros((self.mdl.nzz, self.mdl.nxx))
    d2u_dz2 = np.zeros((self.mdl.nzz, self.mdl.nxx))

    dh2 = self.c.dh * self.c.dh
    arg = self.c.dt * self.c.dt * self.mdl.model * self.mdl.model
  
    for isrc in range(len(self.geom.srcxId)):

      self.seis.seismogram.fill(0.0)
      
      self.upas.fill(0.0)
      self.upre.fill(0.0)
      self.ufut.fill(0.0)

      self.depas.fill(0.0)
      self.depre.fill(0.0)
      self.defut.fill(0.0)

      ix = int(self.geom.srcxId[isrc]) + self.c.nb
      iz = int(self.geom.srczId[isrc]) + self.c.nb
      
      for t in range(1, self.c.nt - 1):

        self.upre[iz, ix] += self.ricker[t] / dh2

        dx2_dz2 = laplacian2d(
          self.upre,
          d2u_dx2,
          d2u_dz2,
          self.mdl.nzz,
          self.mdl.nxx,
          dh2
        )

        self.upas = (
          arg * dx2_dz2
          + 2.0 * self.upre
          - self.ufut
        )

        self.ufut = self.upre * self.damp2D
        self.upre = self.upas * self.damp2D

        for irec in range(self.geom.nrec):
          rx = int(self.geom.recx[irec]) + self.c.nb
          rz = int(self.geom.recz[irec]) + self.c.nb
          self.seis.seismogram[t, irec] = self.upre[rz, rx]

        if self.c.snap_bool and not t % self.snap_ratio:
          self.snapshots_src.append(self.upas.copy())

      self.seis.remove_direct_wave(ix, iz)
      self.seis.plot(self.seis.residual)

      for t in range(self.c.nt - 1, 1, -1):

        for irec in range(self.geom.nrec):
          rx = int(self.geom.recx[irec]) + self.c.nb
          rz = int(self.geom.recz[irec]) + self.c.nb
          self.depre[rz, rx] += self.seis.residual[t, irec] / dh2

        dx2_dz2 = laplacian2d(
          self.depre,
          d2u_dx2,
          d2u_dz2,
          self.mdl.nzz,
          self.mdl.nxx,
          dh2
        )

        self.depas = (
          arg * dx2_dz2
          + 2.0 * self.depre
          - self.defut
        )

        self.defut = self.depre * self.damp2D
        self.depre = self.depas * self.damp2D

        tf = self.c.nt - t

        if self.c.snap_bool and not tf % self.snap_ratio:
          self.snapshots_rec.append(self.depre.copy())

    for i in range(199):
      self.image += self.snapshots_src[i] * self.snapshots_rec[i]

#      if not i % 10:
#        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,8))
#
#        ax[0].imshow(self.snapshots_src[i])
#        ax[1].imshow(self.snapshots_rec[i])
#
#        plt.show()

  def get_ricker(self):
    fc = self.c.fmax / (3.0 * np.sqrt(np.pi))
    t = np.arange(self.c.nt) * self.c.dt - self.c.tlag

    arg = np.pi * (t * fc * np.pi) ** 2.0 

    self.ricker = (1.0 - 2.0 * arg) * np.exp(-arg)

  def set_damper(self):
    damp1D = np.zeros(self.c.nb)

    for i in range(self.c.nb):
      damp1D[i] = np.exp(-(self.c.factor*(self.c.nb - i))**2.0)

    for i in range(self.mdl.nzz):
      self.damp2D[i,:self.c.nb] *= damp1D
      self.damp2D[i,-self.c.nb:] *= damp1D[::-1]

    for j in range(self.mdl.nxx):
      self.damp2D[:self.c.nb,j] *= damp1D
      self.damp2D[-self.c.nb:,j] *= damp1D[::-1]

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

  def plot_snapshots(self):
    xloc = np.linspace(0, self.c.nx-1, 11, dtype=int)
    xlab = np.array(xloc * self.c.dh, dtype=int)

    zloc = np.linspace(0, self.c.nz-1, 7, dtype=int)
    zlab = np.array(zloc * self.c.dh, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 5))

    ims = []
    for snap in self.snapshots_src:
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
      interval=(self.c.nt / len(self.snapshots_src) + 1) * self.c.dt * 1e3,
      blit=False,
      repeat_delay=0
    )

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)

    ax.set_yticks(zloc)
    ax.set_yticklabels(zlab)

    plt.show()
    return ani

  def plot_model_and_geometry(self):
      xloc = np.linspace(0, self.c.nx - 1, 11, dtype=int)
      xlab = np.array(xloc * self.c.dh, dtype=int)

      zloc = np.linspace(0, self.c.nz - 1, 7, dtype=int)
      zlab = np.array(zloc * self.c.dh, dtype=int)

      fig, ax = plt.subplots(figsize=(12, 5))

      img = ax.imshow(
          self.mdl.model[
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

  def plot_image(self):
      xloc = np.linspace(0, self.c.nx - 1, 11, dtype=int)
      xlab = np.array(xloc * self.c.dh, dtype=int)

      zloc = np.linspace(0, self.c.nz - 1, 7, dtype=int)
      zlab = np.array(zloc * self.c.dh, dtype=int)

      fig, ax = plt.subplots(figsize=(12, 5))

      img = ax.imshow(
          self.image[
              self.c.nb:self.c.nb + self.c.nz,
              self.c.nb:self.c.nb + self.c.nx
          ],
          aspect="auto",
          cmap="Greys",
      )

      ax.set_xticks(xloc)
      ax.set_xticklabels(xlab)
      ax.set_yticks(zloc)
      ax.set_yticklabels(zlab)

      ax.set_xlabel("Distance [m]")
      ax.set_ylabel("Depth [m]")
      ax.set_title("Image")

      plt.show()

class Seismogram:
  def __init__(self, geom, c):
    self.geom = geom
    self.c = c

    self.residual = np.zeros((self.c.nt, self.geom.nrec))

    self.seismogram = np.zeros((self.c.nt, self.geom.nrec))

    self.direct_wave = np.array([])

  def load_residual(self, input_path):
    path = input_path

    self.residual = np.fromfile(
        path, dtype=np.float32, count=self.c.nt*self.geom.nrec
        ).reshape([self.c.nt, self.geom.nrec], order='F')

  def remove_direct_wave(self, ix, iz, epsilon=0.09):
      self.residual = self.seismogram.copy()

      nt = self.residual.shape[0]

      rx = self.geom.recx + self.c.nb
      rz = self.geom.recz + self.c.nb

      off = np.sqrt(
        (ix - rx)**2 + (iz - rz)**2
      ) * self.c.dh
      self.geom.direct_wave = (off / 1500.0) + self.c.tlag
      
      samples = np.arange(nt)

      for j in range(self.geom.nrec):

        t_center = self.geom.direct_wave[j]

        t0 = t_center - epsilon
        t1 = t_center + epsilon

        t0_idx = int(t0 / self.c.dt)
        t1_idx = int(t1 / self.c.dt)

        imin = max(0, min(t0_idx, t1_idx))
        imax = min(nt, max(t0_idx, t1_idx))

        mask = (samples >= imin) & (samples <= imax)

        self.residual[mask, j] = 0.0

  def plot(self, seismogram):
    tloc = np.linspace(0, self.c.nt - 1, 11, dtype=int)
    tlab = np.around(tloc * self.c.dt, decimals=1)

    xloc = np.linspace(0, self.geom.nrec - 1, 9)
    xlab = np.array(self.c.dh * xloc, dtype=int)

    scale_min = np.percentile(seismogram, 100 - self.c.perc)
    scale_max = np.percentile(seismogram, self.c.perc)

    fig, ax = plt.subplots(figsize=(10, 8))

    img = ax.imshow(seismogram, aspect="auto", cmap="Greys",
                    vmin=scale_min, vmax=scale_max)

    #plot of direct wave curve
    try:
      x_plot = np.arange(self.geom.nrec)
      y_plot = self.direct_wave / self.c.dt
    
      ax.plot(x_plot, y_plot, 'r--')
    except:
      pass

    ax.set_yticks(tloc)
    ax.set_yticklabels(tlab)

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)

    ax.set_xlabel("Offset (m)", fontsize=13)
    ax.set_ylabel("TWT (s)", fontsize=13)

    plt.show()

class Model:
  def __init__(self, c) -> None:
    self.c = c

    self.nxx = 2*self.c.nb + self.c.nx
    self.nzz = 2*self.c.nb + self.c.nz

    self.model = np.zeros((self.c.nz, self.c.nx))

  def load(self) -> None:
    self.model = np.fromfile(
      self.c.model_path, dtype=np.float32, count=self.c.nx*self.c.nz
    ).reshape([self.c.nz, self.c.nx], order='F')

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

class Geometry:
  def __init__(self, c) -> None:
    self.c = c

    self.recx, self.recz     = np.array([]), np.array([])
    self.srcxId, self.srczId = np.array([]), np.array([])

    self.nrec = 0

    self.dt_canditates = np.array([])

    self.max_dt = 0.0

  def get(self) -> None:
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

@njit(parallel=True)
def laplacian2d(
    upre, d2u_dx2, d2u_dz2, 
    nzz, nxx, dh2,
) -> None:
  inv_dh2 = 1.0 / (5040.0 * dh2)

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      d2u_dx2[i, j] = (
          -9   * upre[i-4, j] + 128   * upre[i-3, j] - 1008 * upre[i-2, j] +
          8064 * upre[i-1, j] - 14350 * upre[i,   j] + 8064 * upre[i+1, j] -
          1008 * upre[i+2, j] + 128   * upre[i+3, j] - 9    * upre[i+4, j]
      ) * inv_dh2

      d2u_dz2[i, j] = (
          -9   * upre[i, j-4] + 128   * upre[i, j-3] - 1008 * upre[i, j-2] +
          8064 * upre[i, j-1] - 14350 * upre[i, j]   + 8064 * upre[i, j+1] -
          1008 * upre[i, j+2] + 128   * upre[i, j+3] - 9    * upre[i, j+4]
      ) * inv_dh2

  return d2u_dx2 + d2u_dz2

def measure_runtime(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Runtime: {round(end - start, 4)} seconds")
    return result

  return wrapper

#self.transit_time = np.zeros((self.mdl.nzz, self.mdl.nxx))
#self.ref = np.zeros((self.mdl.nzz, self.mdl.nxx))

#@njit(parallel=True)
#def update_tt(
#    upre: np.ndarray,
#    ref: np.ndarray,
#    transit_time: np.ndarray,
#    current_time: float,
#    nzz: int,
#    nxx: int,
#) -> None:
#  for i in prange(4, nzz - 4):
#    for j in range(4, nxx - 4):
      # Criterio da Amplitude Maxima - Andre Bulcao
      # if abs(u(Ω,t)) >= abs(ref(Ω)) then
      # ref(Ω) = u(Ω,t)
      # T(Ω) = t
      # endif
#      if abs(upre[i,j]) >= abs(ref[i,j]):
#          ref[i,j] = upre[i,j]
#          transit_time[i,j] = current_time

#  def remove_direct_wave(self, ix, iz, epsilon=0.70e-1):
#    nt = self.seismogram.shape[0]
#
#    rx = self.geom.recx + self.c.nb
#    rz = self.geom.recz + self.c.nb
#
#    off = np.sqrt((ix - rx)**2 + (iz - rz)**2) * self.c.dh
#    self.direct_wave = (off / 1500.0) + self.c.tlag + epsilon
#
#    for j in range(self.geom.nrec):
#      t0 = self.direct_wave[j]
#      t0_idx = int(t0 / self.c.dt)
#
#      samples = np.arange(nt)
#
#      condition = samples <= t0_idx  
#
#      self.seismogram[condition, j] = 0.0