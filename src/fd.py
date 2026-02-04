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

    self.tlag = c.tlag
    self.ricker = np.zeros(self.c.nt)

    # src domain
    self.Psrc = np.zeros((self.mdl.nzz, self.mdl.nxx, self.c.nt))

    # rc domain
    self.seismogram = seis.seismogram
    self.Prec = np.zeros((self.mdl.nzz, self.mdl.nxx, self.c.nt))

    self.snapshots = []

    #self.transit_time = np.zeros((self.mdl.nzz, self.mdl.nxx))
    #self.ref = np.zeros((self.mdl.nzz, self.mdl.nxx))

    self.snap_id = 0
    self.rec_snap_id = 0

    self.image = np.zeros((self.mdl.nzz, self.mdl.nxx))

  def get_ricker(self):
    fc = self.c.fmax / (3.0 * np.sqrt(np.pi))
    t = np.arange(self.c.nt) * self.c.dt - self.tlag

    arg = np.pi * (t * fc * np.pi) ** 2.0 

    self.ricker = (1.0 - 2.0 * arg) * np.exp(-arg)

  def set_damper(self):
    damp1D = np.zeros(self.mdl.nb)

    for i in range(self.mdl.nb):
        damp1D[i] = np.exp(-(self.c.factor*(self.mdl.nb - i))**2.0)

    for i in range(self.mdl.nzz):
        self.damp2D[i,:self.mdl.nb] *= damp1D
        self.damp2D[i,-self.mdl.nb:] *= damp1D[::-1]

    for j in range(self.mdl.nxx):
        self.damp2D[:self.mdl.nb,j] *= damp1D
        self.damp2D[-self.mdl.nb:,j] *= damp1D[::-1]

  def fd(self):
    d2u_dx2 = np.zeros((self.mdl.nzz, self.mdl.nxx))
    d2u_dz2 = np.zeros((self.mdl.nzz, self.mdl.nxx))

    snap_ratio = int(self.c.nt / self.c.snap_num)

    dh2 = self.c.dh * self.c.dh
    arg = self.c.dt * self.c.dt * self.mdl.model * self.mdl.model

    for i in range(len(self.geom.srcxId)):
      self.seismogram.fill(0.0)
      self.Psrc.fill(0.0)
      self.Prec.fill(0.0)

      ix = int(self.geom.srcxId[i]) + self.mdl.nb
      iz = int(self.geom.srczId[i]) + self.mdl.nb

      for t in range(1, self.c.nt - 1):
        self.Psrc[iz, ix, t] += self.ricker[t] / dh2

        dx2_dz2 = laplacian2d(
            self.Psrc[:, :, t], d2u_dx2, d2u_dz2,
            self.mdl.nzz, self.mdl.nxx, dh2
        )

        self.Psrc[:, :, t+1] = arg * dx2_dz2 + 2 * self.Psrc[:, :, t] - self.Psrc[:, :, t-1]
        self.Psrc[:, :, t+1] *= self.damp2D

        for irec in range(self.geom.nrec):
          rx = int(self.geom.recx[irec]) + self.mdl.nb
          rz = int(self.geom.recz[irec]) + self.mdl.nb
          self.seismogram[t, irec] = self.Psrc[rz, rx, t]

        if self.c.save_snapshots:
          if not t % snap_ratio:
            #self.snapshots.append(self.Psrc[:, :, t].copy())
            self.save_src_domain(t)
            self.snap_id += 1

      #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,8))
    
      #ax[0].imshow(self.Psrc[self.mdl.nb:self.mdl.nb+self.mdl.nz, self.mdl.nb:self.mdl.nb+self.mdl.nx, 301])
      #ax[1].imshow(self.Psrc[self.mdl.nb:self.mdl.nb+self.mdl.nz, self.mdl.nb:self.mdl.nb+self.mdl.nx, 501])
      #ax[2].imshow(self.Psrc[self.mdl.nb:self.mdl.nb+self.mdl.nz, self.mdl.nb:self.mdl.nb+self.mdl.nx, 801])

      #ax[0].set_title("Psrc, nt = 301")
      #ax[1].set_title("Psrc, nt = 501")
      #ax[2].set_title("Psrc, nt = 801")
      
      #plt.tight_layout()
      #plt.show()

      self.seis.remove_direct_wave(ix, iz)
      self.fd_reverse()
      self.get_image()

      self.snap_id = 0
      self.rec_snap_id = 0

      #self.seis.plot(self.seismogram)

    if self.c.save_seismogram:
      self.save_seismogram()

  def fd_reverse(self):
    d2u_dx2 = np.zeros((self.mdl.nzz, self.mdl.nxx))
    d2u_dz2 = np.zeros((self.mdl.nzz, self.mdl.nxx))

    snap_ratio = int(self.c.nt / self.c.snap_num)

    dh2 = self.c.dh * self.c.dh
    arg = self.c.dt * self.c.dt * self.mdl.model * self.mdl.model

    #self.seis.remove_direct_wave()

    for t in range(self.c.nt - 2, 0, -1):
      for i in range(self.geom.nrec):
        ix = int(self.geom.recx[i]) + self.mdl.nb
        iz = int(self.geom.recz[i]) + self.mdl.nb
        self.Prec[iz, ix, t] += self.seismogram[t, i] / dh2

      dx2_dz2 = laplacian2d(
          self.Prec[:, :, t], d2u_dx2, d2u_dz2,
          self.mdl.nzz, self.mdl.nxx, dh2
      )

      self.Prec[:, :, t-1] = arg * dx2_dz2 + 2 * self.Prec[:, :, t] - self.Prec[:, :, t+1]
      self.Prec[:, :, t-1] *= self.damp2D

      if self.c.save_snapshots:
        if not t % snap_ratio:
          self.save_rec_domain(t)
          self.rec_snap_id += 1

    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,8))
    
    #ax[0].imshow(self.Prec[self.mdl.nb:self.mdl.nb+self.mdl.nz, self.mdl.nb:self.mdl.nb+self.mdl.nx, 301])
    #ax[1].imshow(self.Prec[self.mdl.nb:self.mdl.nb+self.mdl.nz, self.mdl.nb:self.mdl.nb+self.mdl.nx, 501])
    #ax[2].imshow(self.Prec[self.mdl.nb:self.mdl.nb+self.mdl.nz, self.mdl.nb:self.mdl.nb+self.mdl.nx, 801])

    #ax[0].set_title("Prec, nt = 301")
    #ax[1].set_title("Prec, nt = 501")
    #ax[2].set_title("Prec, nt = 801")
    
    #plt.tight_layout()
    #plt.show()

  def get_image(self):
    nsnap = self.c.snap_num - 1

    for i in range(nsnap):
      src = np.fromfile(
        OUTPUT_PATH + "/src_domain/" +
           f"snapshot_{self.mdl.nxx}x{self.mdl.nzz}_{i}.bin",
              dtype="float32"
            ).reshape((self.mdl.nzz, self.mdl.nxx), order="F")

      rec = np.fromfile(
        OUTPUT_PATH + "/rec_domain/" +
          f"rec_snapshot_{i}_dt{self.c.dt}_nrec{self.geom.nrec}.bin",
            dtype="float32"
          ).reshape((self.mdl.nzz, self.mdl.nxx), order="F")

      self.image += src * rec

  def save_seismogram(self):
    (
      self.seis.seismogram
      .flatten('F')
      .astype("float32", order='F')
      .tofile(
        self.c.seismogram_output_path +
        f"seismogram_nt{self.c.nt}_dt{self.c.dt}_nrec{self.geom.nrec}.bin"
      )
    )

  def save_src_domain(self, t):
    (
      self.Psrc[:, :, t]
      .flatten('F')
      .astype("float32", order='F')
      .tofile(
        OUTPUT_PATH + "/src_domain/" +
        f"snapshot_{self.mdl.nxx}x{self.mdl.nzz}_{self.snap_id}.bin"
      )
    )

  def save_rec_domain(self, t):
    (
      self.Prec[:, :, t]
      .flatten('F')
      .astype("float32", order='F')
      .tofile(
        OUTPUT_PATH + "/rec_domain/" +
        f"rec_snapshot_{self.rec_snap_id}_dt{self.c.dt}"
        f"_nrec{self.geom.nrec}.bin"
      )
    )

  def plot_snapshots(self):
    xloc = np.linspace(0, self.mdl.nx-1, 11, dtype=int)
    xlab = np.array(xloc * self.c.dh, dtype=int)

    zloc = np.linspace(0, self.mdl.nz-1, 7, dtype=int)
    zlab = np.array(zloc * self.c.dh, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 5))

    ims = []
    for snap in self.snapshots:
      scale = 2.0 * np.std(snap)

      model_frame = ax.imshow(
        self.mdl.model[self.mdl.nb:self.mdl.nb+self.mdl.nz,
                             self.mdl.nb:self.mdl.nb+self.mdl.nx],
        aspect="auto", cmap="jet", alpha=0.5
      )

      snap_frame = ax.imshow(
        snap[self.mdl.nb:self.mdl.nb+self.mdl.nz,
             self.mdl.nb:self.mdl.nb+self.mdl.nx],
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

  def plot_model_and_geometry(self):
      xloc = np.linspace(0, self.mdl.nx - 1, 11, dtype=int)
      xlab = np.array(xloc * self.c.dh, dtype=int)

      zloc = np.linspace(0, self.mdl.nz - 1, 7, dtype=int)
      zlab = np.array(zloc * self.c.dh, dtype=int)

      fig, ax = plt.subplots(figsize=(12, 5))

      img = ax.imshow(
          self.mdl.model[
              self.mdl.nb:self.mdl.nb + self.mdl.nz,
              self.mdl.nb:self.mdl.nb + self.mdl.nx
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
      xloc = np.linspace(0, self.mdl.nx - 1, 11, dtype=int)
      xlab = np.array(xloc * self.c.dh, dtype=int)

      zloc = np.linspace(0, self.mdl.nz - 1, 7, dtype=int)
      zlab = np.array(zloc * self.c.dh, dtype=int)

      fig, ax = plt.subplots(figsize=(12, 5))

      img = ax.imshow(
          self.image[
              self.mdl.nb:self.mdl.nb + self.mdl.nz,
              self.mdl.nb:self.mdl.nb + self.mdl.nx
          ],
          aspect="auto",
          cmap="jet",
      )

      ax.set_xticks(xloc)
      ax.set_xticklabels(xlab)
      ax.set_yticks(zloc)
      ax.set_yticklabels(zlab)

      ax.set_xlabel("Distance [m]")
      ax.set_ylabel("Depth [m]")
      ax.set_title("Image")

      #plt.colorbar(img, ax=ax, label="VP [m/s]")
      #ax.legend()

      plt.show()

class Seismogram:
  def __init__(self, geom, c):
    self.geom = geom
    self.c = c

    self.seismogram_load = np.zeros((self.c.nt, self.geom.nrec))

    self.seismogram = np.zeros((self.c.nt, self.geom.nrec))

    self.direct_wave = np.array([])

  def load(self, input_path):
    path = input_path or self.c.seismogram_input_path

    self.seismogram_load = np.fromfile(
        path, dtype=np.float32, count=self.c.nt*self.geom.nrec
        ).reshape([self.c.nt, self.geom.nrec], order='F')

  def remove_direct_wave(self, ix, iz, epsilon=0.70e-1):
    nt = self.seismogram.shape[0]

    rx = self.geom.recx + self.c.nb
    rz = self.geom.recz + self.c.nb

    off = np.sqrt((ix - rx)**2 + (iz - rz)**2) * self.c.dh
    self.direct_wave = (off / 1500.0) + self.c.tlag + epsilon

    for j in range(self.geom.nrec):
      t0 = self.direct_wave[j]
      t0_idx = int(t0 / self.c.dt)

      samples = np.arange(nt)

      condition = samples <= t0_idx  

      self.seismogram[condition, j] = 0.0

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
    self.nx = c.nx
    self.nz = c.nz
    self.nb = c.nb

    self.nxx = 2*self.nb + self.nx
    self.nzz = 2*self.nb + self.nz

    self.model = np.zeros((self.nz, self.nx))

    self.interfaces = c.interfaces
    self.value_interfaces = c.velocity_interfaces

  def get_model(self) -> None:
    if not len(self.interfaces):
      self.model[:, :] = self.value_interfaces[0]
    else:
      self.model[:self.interfaces[0], :] = self.value_interfaces[0]
      for layer, vel in enumerate(self.value_interfaces[1:]):
        self.model[self.interfaces[layer]:, :] = vel

  def set_boundary(self) -> None:
    model_ext = np.zeros((self.nzz, self.nxx))

    for j in range(self.nx):
      for i in range(self.nz):
        model_ext[i+self.nb, j+self.nb] = self.model[i, j]

    for j in range(self.nb, self.nx+self.nb):
      for i in range(self.nb):
        model_ext[i, j] = model_ext[self.nb, j]
        model_ext[self.nz+self.nb+i, j] = model_ext[self.nz+self.nb-1, j]

    for i in range(self.nzz):
      for j in range(self.nb):
        model_ext[i, j] = model_ext[i, self.nb]
        model_ext[i, self.nx+self.nb+j] = model_ext[i, self.nx+self.nb-1]

    self.model = model_ext

class Geometry:
  def __init__(self, c) -> None:
    self.c = c

    self.recx, self.recz     = np.array([]), np.array([])
    self.srcxId, self.srczId = np.array([]), np.array([])

    self.nrec = 0

    self.dt_canditates = np.array([])

    self.max_dt = 0.0

  def get_geometry(self) -> None:
    receivers = np.loadtxt(self.c.receivers, delimiter=',', skiprows=1)

    if receivers.ndim == 1:
      self.recx = np.array([receivers[1]])
      self.recz = np.array([receivers[2]])
    else:
      self.recx = receivers[:, 1]
      self.recz = receivers[:, 2]

    sources = np.loadtxt(self.c.sources, delimiter=',', skiprows=1)

    if sources.ndim == 1:
      self.srcxId = np.array([sources[1]])
      self.srczId = np.array([sources[2]])
    else:
      self.srcxId = sources[:, 1]
      self.srczId = sources[:, 2]

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

@njit(parallel=True)
def update_tt(
    upre: np.ndarray,
    ref: np.ndarray,
    transit_time: np.ndarray,
    current_time: float,
    nzz: int,
    nxx: int,
) -> None:
  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      # Criterio da Amplitude Maxima - Andre Bulcao
      # if abs(u(Ω,t)) >= abs(ref(Ω)) then
      # ref(Ω) = u(Ω,t)
      # T(Ω) = t
      # endif
      if abs(upre[i,j]) >= abs(ref[i,j]):
          ref[i,j] = upre[i,j]
          transit_time[i,j] = current_time

def measure_runtime(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Runtime: {round(end - start, 4)} seconds")
    return result

  return wrapper