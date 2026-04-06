from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from . import Config, Geometry

import matplotlib.pyplot as plt

import cupy as cp

OUTPUT_PATH = "data/output/"

class Seismogram:
  def __init__(self, c: Config, geom: Geometry):
    self.geom = geom
    self.c = c

    shape = (c.nt, geom.nrec)

    self.seismogram = cp.ones(shape, dtype=cp.float32)

    self.seismogram_homo = cp.ones(shape, dtype=cp.float32)

    self.direct_wave = cp.array([])

    # WARNING: not implemented yet
    if self.c.seismogram_mode.upper() == "LOAD":
      self.load()

  def load(self, icput_path=None):
    if icput_path is None:
      path = self.c.load_seis_path

    self.seismogram = cp.fromfile(
        path, dtype=cp.float32, count=self.c.nt*self.geom.nrec
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

  def apply_agc(self, time_window : float):
      sliding_window = int(time_window / self.c.dt) + 1

      for i in range(self.geom.nrec):
        trace = self.seismogram[:, i]
        l, h = 0, sliding_window - 1
        mid = (l + h) // 2

        while h < self.c.nt - 1:
          window_samples = trace[l:h]
          mean_amplitude = cp.mean(cp.abs(window_samples))
          trace[mid] /= mean_amplitude + 1e-6

          l, h, mid = l + 1, h + 1, mid + 1

        self.seismogram[:,i] = trace

  def plot(self, seismogram=None):
    import numpy as np

    if seismogram is None:
      seismogram = cp.asnumpy(self.seismogram)

    tloc = np.linspace(0, self.c.nt - 1, 11, dtype=int)
    tlab = np.around(tloc * self.c.dt, decimals=1)

    xloc = np.linspace(0, self.geom.nrec - 1, 9)
    xlab = np.array(self.c.dh * xloc, dtype=int)

    scale_min = np.percentile(seismogram, 100 - self.c.perc)
    scale_max = np.percentile(seismogram, self.c.perc)

    fig, ax = plt.subplots(figsize=(10, 8))

    img = ax.imshow(seismogram, aspect="auto", cmap="Greys",
                    vmin=scale_min, vmax=scale_max)

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Amplitude", fontsize=13)

    ax.set_yticks(tloc)
    ax.set_yticklabels(tlab)

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)

    ax.set_xlabel("Offset (m)", fontsize=13)
    ax.set_ylabel("TWT (s)", fontsize=13)

    plt.show()

