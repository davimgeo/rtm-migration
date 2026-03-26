import sys
import os

os.chdir(os.path.join(os.path.dirname(sys.argv[0]), ".."))

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "data/input/models/"

class SmoothCircle:
  def __init__(self, model, center, ref_sigma):
    self.model = model
    self.center = center
    self.ref_sigma = ref_sigma

  def export_varying_circles(self,ratio, iterations=11) -> None:
    assert ~(iterations % 2) and iterations > 1

    width, height = self.model.shape

    half_it = int(iterations * 0.5)

    for it in range(-half_it, half_it + 1):
      sigma = self.ref_sigma - it * ratio

      d_calc = self.get(
          model=model,
          center=center,
          sigma=ref_sigma,
          amp=0.40
      )

      if it == int(iterations * 0.5):
        d_calc.flatten('F')              \
            .astype('float32', order='F') \
            .tofile(
            OUTPUT_PATH + f"ref_smooth_circle_sigma{sigma}_{width}x{height}.bin"
        )
      else:
        d_calc.flatten('F')              \
            .astype('float32', order='F') \
            .tofile(
            OUTPUT_PATH + f"smooth_circle_sigma{sigma}_{width}x{height}.bin"
        )

  def get(
    self,
    model: np.ndarray,
    center: tuple,
    sigma: float,
    amp: float
  ) -> np.ndarray:

    nz, nx = model.shape
    x, y = np.ogrid[:nz, :nx]

    x2 = (x - center[0])**2
    y2 = (y - center[1])**2

    gaussian = amp * (
        np.exp(-(x2 + y2) / (2.0 * sigma**2))
    )

    return (1 + gaussian) * model

  def l1_norm(self, d_obs: np.ndarray, d_calc: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(d_obs - d_calc))

  def l2_norm(self, d_obs: np.ndarray, d_calc: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((d_obs - d_calc)**2))

nz, nx = 300, 900

model = np.full((nz, nx), 2500.0)

center = (nz // 2, nx // 2)

ref_sigma = 40

smooth_circle = SmoothCircle(model, center, ref_sigma)
smooth_circle.export_varying_circles(ratio=2)

# fig, ax = plt.subplots(3, 3)

# sigmas = np.array([
#     [ref_sigma - 10, ref_sigma - 8, ref_sigma - 4],
#     [ref_sigma - 2 , ref_sigma     , ref_sigma - 3],
#     [ref_sigma - 6, ref_sigma - 9, ref_sigma - 12]
# ])

# _, s_dim = sigmas.shape

# for i in range(s_dim):
#   for j in range(s_dim):
#     sigma = sigmas[i][j]
        
#     model_smooth = smooth_circle_model(
#       model=model,
#       center=center,
#       sigma=sigma,
#       amp=4
#     )

#     im = ax[i, j].imshow(model_smooth, cmap="seismic")
#     ax[i, j].set_title(f"sigma = {sigma}")

# plt.show()