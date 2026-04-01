from __future__ import annotations

import sys
import os

os.chdir(os.path.join(os.path.dirname(sys.argv[0]), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======================================================== #

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

OUTPUT_PATH = "data/input/models/"

uncalled = True

def measure_runtime(func):
  import time
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Function {func.__name__} took: {round(end - start, 4)} seconds")
    return result

  return wrapper

@njit(parallel=True, fastmath=True)
def _smooth_kernel(
  sigma: float,
  amp: float,
  base_model: np.ndarray,
  R2: np.ndarray,
  nx: int,
  nz: int,
  epsilon=1e-9
) -> np.ndarray:

  gaussian = np.empty((nz, nx), dtype=np.float64)

  arg = 1.0 / (2.0 * (sigma + epsilon))**2

  for i in prange(nz):
    for j in range(nx):

      gaussian[i, j] = amp * np.exp(
          -R2[i, j] * arg
      )

  return (1 + gaussian) * base_model

class SmoothCircle:
  def __init__(
    self, cfg: Parameters
  ) -> None:

    self.cfg = cfg

    self.nz, self.nx = cfg.nz, cfg.nx

    self.base_model = np.full((cfg.nz, cfg.nx), cfg.value)

    self.vmin, self.vmax = cfg.varying_range

    self.ref_circle_smooth = np.zeros((cfg.nz, cfg.nx))

    self.sigma = np.linspace(self.vmin, self.vmax, cfg.size)
    self.amp   = np.linspace(0, 80, cfg.size)

    #self.sigma = np.sort(np.unique(np.append(self.sigma, self.cfg.ref_sigma)))
    #self.amp = np.sort(np.unique(np.append(self.amp, self.cfg.ref_amp)))

    self.delta_sigma = self.cfg.ref_sigma - self.sigma
    self.delta_amp = self.cfg.ref_amp - self.amp

    self.l2 = np.zeros(cfg.size)
    self.l1 = np.zeros(cfg.size)

    x = np.arange(cfg.nz)
    y = np.arange(cfg.nx)

    X, Y = np.meshgrid(x, y, indexing="ij")
    self.R2 = (X - cfg.center[0])**2 + (Y - cfg.center[1])**2

    self.X, self.Y = np.meshgrid(self.delta_sigma, self.delta_amp)

  @measure_runtime
  def varying_circles(self) -> None:

    if self.cfg.prop_type == "sigma":

      for idx, sigma in enumerate(self.sigma):

        d_calc = self.__smooth_kernel(sigma=sigma, amp=self.cfg.ref_amp)

        self.l2[idx] = self.l2_norm(self.ref_circle_smooth, d_calc)
        self.l1[idx] = self.l1_norm(self.ref_circle_smooth, d_calc)

    elif self.cfg.prop_type == "amp":

      for idx, amp in enumerate(self.amp):

        d_calc = self.__smooth_kernel(self.cfg.ref_sigma, amp)

        self.l2[idx] = self.l2_norm(self.ref_circle_smooth, d_calc)
        self.l1[idx] = self.l1_norm(self.ref_circle_smooth, d_calc)

    elif self.cfg.prop_type == "both":

      self.l2 = np.zeros_like(self.X)
      self.l1 = np.zeros_like(self.Y)

      for i, sigma in enumerate(self.sigma):
       for j, amp in enumerate(self.amp):
      
         d_calc = self.__smooth_kernel(sigma=sigma, amp=amp)
      
         self.l2[i, j] = self.l2_norm(self.ref_circle_smooth, d_calc)
         self.l1[i, j] = self.l1_norm(self.ref_circle_smooth, d_calc)

    else:
      raise TypeError("Make sure you choose a valid type. [sigma, amp, both]")

  def get_ref_circle(self):
    self.ref_circle_smooth = self.__smooth_kernel(self.cfg.ref_sigma, self.cfg.ref_amp)

  def __smooth_kernel(self, sigma, amp):
    return _smooth_kernel(
      sigma, amp, self.base_model,
      self.R2, self.nx, self.nz
    )

  def l2_norm(self, d_obs, d_calc) -> np.ndarray:
    return np.sqrt(np.sum((d_obs - d_calc)**2))

  def l1_norm(self, d_obs, d_calc) -> np.ndarray:
    return np.sum(np.abs(d_obs - d_calc))

  def plot(self, model=None) -> None:
    if model is None:
      model = self.ref_circle_smooth

    fig, ax = plt.subplots(figsize=(12, 5))

    img = ax.imshow(
      model,
      aspect="auto",
      cmap="jet",
    )

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Velocity Model")

    plt.colorbar(img, ax=ax, label="VP [m/s]")

    plt.show()

  def plot_varying_circles(self) -> None:

    if self.cfg.prop_type == "sigma":

      l2_norm = (self.l2 - self.l2.min()) / (self.l2.max() - self.l2.min())
      l1_norm = (self.l1 - self.l1.min()) / (self.l1.max() - self.l1.min())

      fig, ax = plt.subplots(figsize=(12, 5))

      ax.plot(self.delta_sigma, l2_norm, label="L2", color='b')
      ax.plot(self.delta_sigma, l1_norm, label="L1", color='r')

      ax.set_xlabel(r'$\Delta \sigma$', fontsize=13)
      ax.set_ylabel("Obj Function Normalized", fontsize=13)

      ax.legend()
      plt.grid(True)
      plt.show()

    elif self.cfg.prop_type == "amp":

      l2_norm = (self.l2 - self.l2.min()) / (self.l2.max() - self.l2.min())
      l1_norm = (self.l1 - self.l1.min()) / (self.l1.max() - self.l1.min())

      fig, ax = plt.subplots(figsize=(12, 5))

      ax.plot(self.delta_amp, l2_norm, label="L2", color='b')
      ax.plot(self.delta_amp, l1_norm, label="L1", color='r')

      ax.set_xlabel(r'$\sigma$', fontsize=13)
      ax.set_ylabel("Obj Function Normalized", fontsize=13)

      ax.legend()
      plt.grid(True)
      plt.show()

    elif self.cfg.prop_type == "both":

      l2_norm = (self.l2 - self.l2.min()) / (self.l2.max() - self.l2.min())
      l1_norm = (self.l1 - self.l1.min()) / (self.l1.max() - self.l1.min())

      plt.imshow(l2_norm)

      plt.title("L2 Normalized", fontsize=13)
      plt.xlabel(r"$\Delta \sigma$", fontsize=13)
      plt.ylabel(r"$\Delta amp$", fontsize=13)

      plt.colorbar()
      plt.tight_layout()
      plt.show()

      fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": "3d"}
      )

      surf1 = ax.plot_surface(
        self.X, self.Y, l2_norm,
        cmap="viridis",
        alpha=0.8
      )

      # ax.scatter(
      #   self.cfg.ref_sigma, self.cfg.ref_amp, 0.0,
      #   color='b',
      #   s=50,
      #   label='Reference'
      # )

      mask = np.isclose(l2_norm, np.min(l2_norm), atol=1e-6)

      i, j = np.unravel_index(np.argmin(self.l2), self.l2.shape)
      print(self.X[i, j], self.Y[i, j])

      # ax.scatter(
      #   self.X[i, j],
      #   self.Y[i, j],
      #   l2_norm[i, j],
      #   s=50,
      #   color='r',
      #   label="Minimum"
      # )

      fig.colorbar(surf1, shrink=0.5, aspect=5)

      ax.set_box_aspect([1, 1, 1])

      ax.set_xlabel(r'$\Delta \sigma$', fontsize=13)
      ax.set_ylabel(r'$\Delta amp$', fontsize=13)
      ax.set_zlabel(self.cfg.cfg_type, fontsize=13)

      ax.view_init(elev=15, azim=4)

      #ax.legend(loc="lower right")
      plt.show()
 
@dataclass
class Parameters:
  nz: int = 300
  nx: int = 900

  value: int = 2500.0

  center: tuple = (nz // 2, nx // 2)

  ref_sigma: int = 40
  ref_amp: float = 0.40

  varying_range = [0, 80]
  size = 51

  cfg_type: str = "L2"
  prop_type: str = "both"

cfg = Parameters()

smooth_circle = SmoothCircle(cfg)
smooth_circle.get_ref_circle()
smooth_circle.varying_circles()

smooth_circle.plot()
smooth_circle.plot_varying_circles()

# 20.099999999999998 0.6985

# compare test and smooth_circle.ref_circle_smooth

#ref = smooth_circle.ref_circle_smooth
#test = smooth_circle._SmoothCircle__smooth_kernel(20.1, 0.6985)

