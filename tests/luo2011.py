from __future__ import annotations

import sys
import os

os.chdir(os.path.join(os.path.dirname(sys.argv[0]), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======================================================== #

import matplotlib.pyplot as plt
import numpy as np

from src import measure_runtime

OUTPUT_PATH = "data/input/models/"

uncalled = True

class SmoothCircle:
  def __init__(
    self, obj: ObjectiveFunctions, nz: int, 
    nx: int, center: tuple, value: int, 
    ref_sigma: int, ref_amp: float
  ) -> None:

    self.obj = obj

    self.nz, self.nx = nz, nx

    self.center = center
    self.ref_sigma = ref_sigma
    self.ref_amp = ref_amp

    self.value = value

    self.ref_circle_smooth = np.zeros((nz, nx))

    self.delta_sigma = np.array([])
    self.delta_amp = np.array([])

    self.l2 = np.array([])
    self.l1 = np.array([])

  @measure_runtime
  def varying_circles(self, ratio: float, prop_type: str, iterations=11) -> None:
    assert ~(iterations % 2) and iterations > 1

    half_it = int(iterations * 0.5)

    self.delta_sigma = np.zeros(iterations)
    self.delta_amp = np.zeros(iterations)

    self.l2 = np.zeros(iterations)
    self.l1 = np.zeros(iterations)

    if prop_type == "sigma":

      for it in range(-half_it, half_it + 1):
        idx = it + half_it

        sigma = self.ref_sigma - it * ratio

        d_calc = self.__smooth_kernel(sigma=sigma, amp=0.40)

        self.delta_sigma[idx] = self.ref_sigma - sigma

        self.l2[idx] = self.obj.l2_norm(self.ref_circle_smooth, d_calc)
        self.l1[idx] = self.obj.l1_norm(self.ref_circle_smooth, d_calc)

    elif prop_type == "amp":

      for it in range(-half_it, half_it + 1):
        idx = it + half_it

        amp =  self.ref_amp - it * ratio

        d_calc = self.__smooth_kernel(self.ref_sigma, amp)

        self.delta_amp[idx] = self.ref_amp - amp

        self.l2[idx] = self.obj.l2_norm(self.ref_circle_smooth, d_calc)
        self.l1[idx] = self.obj.l1_norm(self.ref_circle_smooth, d_calc)

    elif prop_type == "both":
      
      for i, it in enumerate(range(-half_it, half_it + 1)):
        self.delta_sigma[i] = it * ratio
        self.delta_amp[i] = it * ratio

      X, Y = np.meshgrid(self.delta_sigma, self.delta_amp)

      self.l2 = np.zeros_like(X)
      self.l1 = np.zeros_like(X)

      for i in range(iterations):
        for j in range(iterations):
          sigma = self.ref_sigma - X[i, j]
          amp   = self.ref_amp   - Y[i, j]

          d_calc = self.__smooth_kernel(sigma=sigma, amp=amp)

          self.l2[i, j] = self.obj.l2_norm(self.ref_circle_smooth, d_calc)
          self.l1[i, j] = self.obj.l1_norm(self.ref_circle_smooth, d_calc)

    else:
      raise TypeError("Make sure you choose a valid type. [sigma, amp, both]")

  def get_ref_circle(self):
    self.ref_circle_smooth = self.__smooth_kernel(self.ref_sigma, self.ref_amp)

  def __smooth_kernel(
    self,
    sigma: float,
    amp: float,
    center=None,
    epsilon=1e-9
  ) -> np.ndarray:
    if center is None:
      center = self.center

    model = np.full((self.nz, self.nx), self.value)

    x, y = np.ogrid[:self.nz, :self.nx]

    x2 = (x - center[0])**2
    y2 = (y - center[1])**2

    gaussian = amp * (
        np.exp(-(x2 + y2) / (2.0 * (sigma + epsilon)**2))
    )

    return (1 + gaussian) * model

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

  def plot_obj(self, obj_type: str, prop_type: str) -> None:

    if obj_type == "L2":
      obj = self.l2
    elif obj_type == "L1":
      obj= self.l1
    else:
      raise TypeError("Make sure you choose a valid type. [L2, L1]")

    if prop_type == "sigma":
      label = "Δσ"
      x = self.delta_sigma
    elif prop_type == "amp":
      label = "Δ_amp"
      x = self.delta_amp

    fig, ax = plt.subplots(figsize=(12, 5))

    img = ax.plot(x, obj)

    ax.set_xlabel(label, fontsize=13)
    ax.set_ylabel(obj_type, fontsize=13)
    #ax.set_title("")

    plt.grid(True)

    plt.show()

class ObjectiveFunctions:

  def l2_norm(self, d_obs, d_calc) -> np.ndarray:
    return np.sqrt(np.sum((d_obs - d_calc)**2))

  def l1_norm(self, d_obs, d_calc) -> np.ndarray:
    return np.sum(np.abs(d_obs - d_calc))

nz, nx = 300, 900

value = 2500.0

center = (nz // 2, nx // 2)

ref_sigma = 40
ref_amp = 0.40

prop_type = "both"

obj = ObjectiveFunctions()

smooth_circle = SmoothCircle(obj, nz, nx, center, value, ref_sigma, ref_amp)
smooth_circle.get_ref_circle()
smooth_circle.varying_circles(ratio=0.3, prop_type=prop_type, iterations=11)

#smooth_circle.plot()
#smooth_circle.plot_obj(obj_type="L1", prop_type=prop_type)

try:
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

  surf = ax.plot_surface(
    smooth_circle.delta_sigma, 
    smooth_circle.delta_amp, 
    smooth_circle.l2, 
    linewidth=0, 
    antialiased=False
  )

  plt.show()
except:
  pass
