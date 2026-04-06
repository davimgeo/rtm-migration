from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from src import Config

import cupy as cp
import matplotlib.pyplot as plt

class Wavelet:
  def __init__(self, c: Config):
    self.c = c

    self.wavelet = cp.zeros(self.c.nt)
    self.wavelet_derivative = cp.zeros(self.c.nt)

  def get(self):
    fc = self.c.fmax / (3.0 * cp.sqrt(cp.pi))
    t = cp.arange(self.c.nt) * self.c.dt - self.c.tlag

    arg = cp.pi * (t * fc * cp.pi) ** 2.0 

    self.wavelet = (1.0 - 2.0 * arg) * cp.exp(-arg)

  def second_derivative(self):
    inv_dh = 1.0 / (12.0 * self.c.dt * self.c.dt)

    d2 = cp.zeros_like(self.wavelet)

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

  def plot(self, wavelet: cp.ndarray):
    
    tloc = cp.linspace(0, self.c.nt - 1, 11, dtype=int)
    tlab = cp.around(tloc * self.c.dt, decimals=1)

    _, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(wavelet)

    ax.set_xlabel("Time [s]", fontsize=13)
    ax.set_ylabel("Amplitude", fontsize=13)

    ax.set_xticks(tloc)
    ax.set_xticklabels(tlab)

    plt.grid(True)

    plt.tight_layout()
    plt.show()

