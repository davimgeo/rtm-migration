from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from . import Config, Geometry

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "data/output/"

class Seismogram:
  def __init__(self, c: Config, geom: Geometry):
    self.geom = geom
    self.c = c

    self.seismogram = np.zeros((self.c.nt, self.geom.nrec))

    self.seismogram_homo = np.zeros((self.c.nt, self.geom.nrec))

    self.direct_wave = np.array([])

    # WARNING: not implemented yet
    if self.c.seismogram_mode.upper() == "LOAD":
      self.load()

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

  def remove_direct_wave_range(self, ix, iz, epsilon=0.09):
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

  def remove_direct_wave(self, ix, iz, epsilon=0.05):
      nt = self.seismogram.shape[0]

      rx = self.geom.recx + self.c.nb
      rz = self.geom.recz + self.c.nb

      off = np.sqrt(
        (ix - rx)**2 + (iz - rz)**2
      ) * self.c.dh
      self.geom.direct_wave = (off / 1500.0) + self.c.tlag
      
      samples = np.arange(nt)

      for j in range(self.geom.nrec):

        t0 = self.geom.direct_wave[j] + epsilon
        t0_idx = int(t0 / self.c.dt)

        mask = (samples <= t0_idx)

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
