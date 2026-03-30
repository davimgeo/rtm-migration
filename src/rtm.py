from __future__ import annotations
import time
from os import system

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit, prange

OUTPUT_PATH = "data/output/"

uncalled = True

def measure_runtime(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Function {func.__name__} took: {round(end - start, 4)} seconds")
    return result

  return wrapper

class Migration:
  def __init__(
    self, model: Model, geom: Geometry, 
    seis: Seismogram, c, wl: Wavelet,
    mod: Modeling
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

    self.energy = np.zeros(self.c.nt)
    self.current = 1

  def rtm(self):

    for isrc in range(len(self.geom.srcxId)):

      self.zero_out_matrices()

      self.define_source_coordinates(isrc)

      self.mod.remove_direct_wave_model(self.ix, self.iz)

      for t in range(1, self.c.nt - 1):

        self.forward_propagation(t)

        #self.energy[t] = np.sum(self.upre * self.upre)

        self.get_src_snaps(t)

      for t in range(self.c.nt - 1, self.tstop, -1):

        self.backward_propagation(t)

        self.accumulate_cross_correlation(t)

      self.image_condition()

      self.show_modeling_status()

    if self.c.save_image:
      self.save()

    #plt.loglog(np.arange(self.c.nt), self.energy)
    #plt.show()

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
        self.wl.ricker, self.ix,
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

  def __image_condition2(self, epsilon=1e-9):
    src = self.snapshots_src
    rec = self.snapshots_rec

    id = 1
    if id == 0:
      numerator = src * rec
      denominator = (src * src)

      self.image += np.sum(
        numerator / (np.sum(denominator, axis=0) + epsilon),
        axis=0
      )
    elif id == 1:
      numerator = src * rec

      self.image += self.dt_snaps * np.sum(
        numerator, axis=0
      )   

  def __get_rc_snaps(self, t: int):
    if not t % self.snap_ratio:
      self.snapshots_rec[self.snap_id_rec] = self.depre.copy()
      self.snap_id_rec -= 1

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
    print(f"\n Shots: {100 * progress}% | {bar[:int((10.0 * progress))]} |")
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

  def plot(self, gradient=False, perc=99) -> None:
    if gradient:
      image = self.gradient
      label = "Image Gradient"
    else:
      image = self.image
      label = "Image"

    xloc = np.linspace(0, self.c.nx - 1, 11, dtype=int)
    xlab = np.array(xloc * self.c.dh, dtype=int)

    zloc = np.linspace(0, self.c.nz - 1, 7, dtype=int)
    zlab = np.array(zloc * self.c.dh, dtype=int)

    _, ax = plt.subplots(figsize=(12, 5))

    img_data = image[
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

class Modeling:

  def __init__(
      self, c, mdl: Model, geom: Geometry,
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
    #self.nsnaps = int((self.c.nt - 1) / self.snap_ratio) + 1

    self.snapshots = np.zeros((self.nsnaps, self.mdl.nzz, self.mdl.nxx))

    self.ix, self.iz = 0, 0
    self.snap_id_src = 0
    self.current = 1

  def fdm_propagation(self, ix: int, iz: int, isSnap=False) -> None:
      self.ix, self.iz = ix, iz

      for t in range(1, self.c.nt - 1):

        _forward_kernel(
          self.upas, self.upre, self.ufut, self.laplacian,
          self.damp_x, self.damp_z, self.inv_dh2,
          self.mdl.nzz, self.mdl.nxx, self.wl.ricker,
          self.ix, self.iz, self.dh2, self.arg, t
        )

        self.__get_seismogram(self.seis.seismogram, self.upre, t)

        self.get_snapshots(t, isSnap)

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
          self.mdl.nzz, self.mdl.nxx, self.wl.ricker,
          self.ix, self.iz, self.dh2, self.arg, t
        )

        self.__get_seismogram(self.seis.seismogram, self.upre, t)

      self.seis.remove_direct_wave(self.ix, self.iz)

  def remove_direct_wave_model(self, ix: int, iz: int) -> None:
      self.ix, self.iz = ix, iz

      for t in range(1, self.c.nt - 1):

        _forward_kernel(
          self.upas, self.upre, self.ufut, self.laplacian,
          self.damp_x, self.damp_z, self.inv_dh2,
          self.mdl.nzz, self.mdl.nxx, self.wl.ricker,
          self.ix, self.iz, self.dh2, self.arg, t
        )

        self.__get_seismogram(self.seis.seismogram, self.upre, t)

        _forward_kernel(
          self.upas_homo, self.upre_homo, self.ufut_homo, 
          self.laplacian_homo, self.damp_x, self.damp_z, 
          self.inv_dh2, self.mdl.nzz, self.mdl.nxx, 
          self.wl.ricker, self.ix, self.iz, self.dh2, self.arg2, t
        )

        self.__get_seismogram(self.seis.seismogram_homo, self.upre_homo, t)

      #self.seis.seismogram -= self.seis.seismogram_homo

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

class Wavelet:
  def __init__(self, c):
    self.c = c

    self.ricker = np.zeros(self.c.nt)

  def get_ricker(self):
    fc = self.c.fmax / (3.0 * np.sqrt(np.pi))
    t = np.arange(self.c.nt) * self.c.dt - self.c.tlag

    arg = np.pi * (t * fc * np.pi) ** 2.0 

    self.ricker = (1.0 - 2.0 * arg) * np.exp(-arg)

class Seismogram:
  def __init__(self, geom, c):
    self.geom = geom
    self.c = c

    self.seismogram = np.zeros((self.c.nt, self.geom.nrec))

    self.seismogram_homo = np.zeros((self.c.nt, self.geom.nrec))

    if self.c.seismogram_mode.upper() == "LOAD":
      self.load()

    self.direct_wave = np.array([])

  def load(self, input_path=None):
    if input_path is None:
      path = self.c.load_seis_path

    self.seismogram = np.fromfile(
        path, dtype=np.float32, count=self.c.nt*self.geom.nrec
        ).reshape([self.c.nt, self.geom.nrec], order='F')

  def save(self, ix, iz, path=None):
    if path is None:
      path = (
        OUTPUT_PATH +
          f"seismogram_{self.c.nt}nt" +
          f"_{self.geom.nrec}nrec_({ix, iz})shot.bin"
      )

    try:
      self.seismogram.flatten('F').astype('float32', order='F').tofile(path)
      print(f"Successfully saved: {path}")

    except OSError as e:
      raise OSError(f"Could not save file: {path}") from e

  def remove_direct_wave(self, ix, iz, epsilon=0.09):
      nt = self.seismogram.shape[0]

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

        self.seismogram[mask, j] = 0.0

  def apply_agc(self, time_window : float):
      sliding_window = int(time_window / self.c.dt) + 1

      for i in range(self.geom.nrec):
        trace = self.seismogram[:, i]
        l, h = 0, sliding_window - 1
        mid = (l + h) // 2

        while h < self.c.nt - 1:
          window_samples = trace[l:h]
          mean_amplitude = np.mean(np.abs(window_samples))
          trace[mid] /= mean_amplitude + 1e-6

          l, h, mid = l + 1, h + 1, mid + 1

        self.seismogram[:,i] = trace

  def plot(self, seismogram=None):
    if seismogram is None:
      seismogram = self.seismogram

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

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Amplitude", fontsize=13)

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

  def plot_model_and_geometry(self, model=None):
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

class Geometry:
  def __init__(self, c) -> None:
    self.c = c

    self.recx, self.recz     = np.array([]), np.array([])
    self.srcxId, self.srczId = np.array([]), np.array([])

    self.nrec = 0

  def get(self):
    mode = self.c.geometry_mode.upper()
    if mode == "LOAD":
      self.load()
    elif mode == "CREATE":
      self.create()
      self.load()
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
      self.nrec = int(self.c.nx / self.c.offset)

      recId = np.arange(1, self.nrec + 1)
      self.recx = np.arange(0, self.nrec) * self.c.offset
      self.recz = np.full(self.nrec, self.c.rec_depth)

      data = np.column_stack((recId, self.recx * self.c.dh, self.recz))

      np.savetxt(
          "data/input/geometry/receivers.txt",
          data,
          fmt="%.0f",
          delimiter=",",
          header="recId, recx, recz",
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


