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

    self.d2u_dx2 = np.zeros((self.mdl.nzz, self.mdl.nxx))
    self.d2u_dz2 = np.zeros((self.mdl.nzz, self.mdl.nxx))

    self.dh2 = self.c.dh * self.c.dh
    self.arg = self.c.dt * self.c.dt * self.mdl.model * self.mdl.model

    self.snap_ratio = int(self.c.nt / self.c.snap_num)
    self.nsnaps = self.c.nt // self.snap_ratio

    self.snapshots_src = np.zeros((self.nsnaps, self.mdl.nzz, self.mdl.nxx))
    self.snapshots_rec = np.zeros((self.nsnaps, self.mdl.nzz, self.mdl.nxx))

    self.image = np.zeros((self.mdl.nzz, self.mdl.nxx))
    self.gradient = np.zeros((self.mdl.nzz, self.mdl.nxx))

    self.ix, self.iz = 0, 0

    # snapshot indices
    self.snap_id_src = 0
    self.snap_id_rec = self.nsnaps - 1

  def rtm(self):
    for isrc in range(len(self.geom.srcxId)):

      self.zero_out_matrices()

      self.ix = int(self.geom.srcxId[isrc]) + self.c.nb
      self.iz = int(self.geom.srczId[isrc]) + self.c.nb

      for t in range(1, self.c.nt - 1):

        self.foward_propagation(t)

        self.register_seismogram(t)

        if self.c.snap_bool and not t % self.snap_ratio:
          self.get_src_snaps()

      if not self.c.load_residual:
        self.seis.remove_direct_wave(self.ix, self.iz)

      for t in range(self.c.nt - 2, 500, -1):

        self.inject_residual(t)

        self.backward_propagation()

        if self.c.snap_bool and not t % self.snap_ratio:
          self.get_rc_snaps()

      self.image_condition()

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

  def foward_propagation(self, t):
    self.upre[self.iz, self.ix] += self.ricker[t] / self.dh2

    lap = laplacian2d(
      self.upre,
      self.d2u_dx2,
      self.d2u_dz2,
      self.mdl.nzz,
      self.mdl.nxx,
      self.dh2
    )

    self.upas = (
      self.arg * lap
      + 2.0 * self.upre
      - self.ufut
    )

    self.ufut = self.upre * self.damp2D
    self.upre = self.upas * self.damp2D

  def register_seismogram(self, t):
    for irec in range(self.geom.nrec):
      rx = int(self.geom.recx[irec]) + self.c.nb
      rz = int(self.geom.recz[irec]) + self.c.nb
      self.seis.seismogram[t, irec] = self.upre[rz, rx]

  def get_src_snaps(self):
    self.snapshots_src[self.snap_id_src] = self.upas
    self.snap_id_src += 1

  def inject_residual(self, t):
    for irec in range(self.geom.nrec):
      rx = int(self.geom.recx[irec]) + self.c.nb
      rz = int(self.geom.recz[irec]) + self.c.nb
      self.depre[rz, rx] += self.seis.residual[t, irec] / self.dh2

  def backward_propagation(self):
    lap = laplacian2d(
      self.depre,
      self.d2u_dx2,
      self.d2u_dz2,
      self.mdl.nzz,
      self.mdl.nxx,
      self.dh2
    )

    self.depas = (
      self.arg * lap
      + 2.0 * self.depre
      - self.defut
    )

    self.defut = self.depre * self.damp2D
    self.depre = self.depas * self.damp2D

  def get_rc_snaps(self):
    self.snapshots_rec[self.snap_id_rec] = self.depre
    self.snap_id_rec -= 1

  def image_condition(self, epsilon=1e-9):
    scale = self.snap_ratio * self.c.dt

    prod = self.snapshots_src * self.snapshots_rec

    id = 0
    if id == 0:
      # I_0
      self.image += scale * np.sum(
        prod,
        axis=0
      )

    elif id == 1:
      # I_1
      mask = prod > 0.0
      self.image += scale * np.sum(prod * mask, axis=0)  

    elif id == 2:
      # I_2
      mask = prod < 0.0
      self.image += scale * np.sum(prod * mask, axis=0)

    elif id == 3:
      # I_3
      self.image += scale * np.sum(
          (prod) /
          (np.sqrt(
              self.auto_correlation(self.snapshots_src) *
              self.auto_correlation(self.snapshots_rec)
          ) + epsilon),
          axis=0
      )

    elif id == 5:
      # I_5
      w = 1 / (self.snapshots_src ** 2 + epsilon)

      self.image += scale * np.sum(
        (prod) / (self.snapshots_src ** 2 + w*(epsilon**2)),
        axis=0
      )

    elif id == 6:
      # I_4
      self.image += scale * np.sum(
        (prod) /
        (np.sqrt(
          self.snapshots_src * self.snapshots_src
          ) + epsilon ** 2),
          axis=0
      )

    elif id == 7:
      # I_4
      mask = prod > 0.0
      self.image += scale * np.sum(
        (prod * mask) /
        (np.sqrt(
          self.snapshots_src * self.snapshots_src * mask
          ) + epsilon ** 2),
          axis=0
      )

    save = 0
    if save:
      export_bin(self.image[
            self.c.nb:self.c.nb + self.c.nz,
            self.c.nb:self.c.nb + self.c.nx
        ], OUTPUT_PATH, width=self.c.nx, height=self.c.nz, name="image_700snaps.bin"
        )
      
      print(f"Sucessfuly saved {OUTPUT_PATH + "image_700snaps.bin"}")

  def auto_correlation(self, A):
    return np.sum(A * A, axis=0)

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

  def plot_image(self, perc=99):
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
      ax.set_title("Image")

      plt.colorbar(img, ax=ax)
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
  def __init__(self, c, geom) -> None:
    self.c = c
    self.geom = geom

    self.nxx = 2*self.c.nb + self.c.nx
    self.nzz = 2*self.c.nb + self.c.nz

    self.model = np.zeros((self.c.nz, self.c.nx))

  def get(self):
    if self.c.load_model:
      self.load()
    else:
      self.create()

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

  def model_gaussian_smooth(self, sigma: float, truncate: float = 4.0) -> None:
      radius = int(truncate * sigma + 0.5)

      x = np.arange(-radius, radius + 1)
      gaussian = np.exp(-0.5 * (x**2) / (sigma**2))

      gaussian /= gaussian.sum()

      for col in range(self.c.nx):
        data = self.model[:, col]

        padded = np.pad(data, radius, mode="reflect")

        smoothed = np.convolve(padded, gaussian, mode="same")

        self.model[:, col] = smoothed[radius:-radius]

  def plot_model_and_geometry(self):
      xloc = np.linspace(0, self.c.nx - 1, 11, dtype=int)
      xlab = np.array(xloc * self.c.dh, dtype=int)

      zloc = np.linspace(0, self.c.nz - 1, 7, dtype=int)
      zlab = np.array(zloc * self.c.dh, dtype=int)

      fig, ax = plt.subplots(figsize=(12, 5))

      img = ax.imshow(
          self.model[
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

class Geometry:
  def __init__(self, c) -> None:
    self.c = c

    self.recx, self.recz     = np.array([]), np.array([])
    self.srcxId, self.srczId = np.array([]), np.array([])

    self.nrec = 0

  def get(self):
    if self.c.create_geom:
      self.create()
    else:
      self.load()

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
      self.recx = np.arange(0, self.nrec) * self.c.offset * self.c.dh
      self.recz = np.full(self.nrec, self.c.rec_depth)

      data = np.column_stack((recId, self.recx, self.recz))

      np.savetxt(
          "data\\input\\geometry\\receivers.txt",
          data,
          fmt="%.0f",
          delimiter=",",
          header="recId, recx, recz",
      )

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

def export_bin(
    a: np.array, path: str, width: int, height: int, name: str
  ):
  a.flatten('F').astype('float32', order='F').tofile(path + f"{name}_{width}x{height}.bin")

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