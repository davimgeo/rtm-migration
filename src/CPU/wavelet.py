from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src import Config

import matplotlib.pyplot as plt
import numpy as np

class Wavelet:
  def __init__(self, c: Config):
    self.c = c

    self.wavelet = np.zeros(self.c.nt)
    self.wavelet_derivative = np.zeros(self.c.nt)

  def get(self):
    fc = self.c.fmax / (3.0 * np.sqrt(np.pi))
    t = np.arange(self.c.nt) * self.c.dt - self.c.tlag

    arg = np.pi * (t * fc * np.pi) ** 2.0 

    self.wavelet = (1.0 - 2.0 * arg) * np.exp(-arg)

  def second_derivative(self):
    inv_dh = 1.0 / (12.0 * self.c.dt * self.c.dt)

    d2 = np.zeros_like(self.wavelet)

    for i in range(2, self.c.nt - 2):
      d2u_dx2 = (
        - self.wavelet[i-2]
        + 16.0 * self.wavelet[i-1]
        - 30.0 * self.wavelet[i]
        + 16.0 * self.wavelet[i+1]
        - self.wavelet[i+2]
      ) * inv_dh

      d2[i] = d2u_dx2 

    d2 = (d2 - d2.min()) / (d2.max() - d2.min())

    self.wavelet_derivative = d2

  def plot(self, wavelet: np.ndarray):
    
    tloc = np.linspace(0, self.c.nt - 1, 11, dtype=int)
    tlab = np.around(tloc * self.c.dt, decimals=1)

    _, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(wavelet)

    ax.set_xlabel("Time [s]", fontsize=13)
    ax.set_ylabel("Amplitude", fontsize=13)

    ax.set_xticks(tloc)
    ax.set_xticklabels(tlab)

    plt.grid(True)

    plt.tight_layout()
    plt.show()