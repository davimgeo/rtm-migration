import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

# ================================== // ================================

import matplotlib.pyplot as plt
import numpy as np

class Plotting:
  
  def load(path: str, height: int, weight: int) -> np.ndarray:
    return np.fromfile(
      path, dtype=np.float32, count=height*weight
    ).reshape([height, weight], order='F')

  def compare(
    self, 
    model1: np.ndarray, 
    model2: np.ndarray, 
    title1=None,
    title2=None
    ) -> None:

    if title1 is None:
      title1 = "Image 1"

    if title2 is None:
      title2 = "Image 2"

    diff = model1 - model2
    diff_norm = diff / np.max(np.abs(model1))

    vmin = min(model1.min(), model2.min())
    vmax = max(model1.max(), model2.max())

    _, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    im0 = axs[0].imshow(model1, aspect='auto', cmap="Greys", vmin=vmin, vmax=vmax)
    axs[0].set_title(title1)
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(model2, aspect='auto', cmap="Greys", vmin=vmin, vmax=vmax)
    axs[1].set_title(title2)
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(diff_norm, aspect='auto', cmap="Greys")
    axs[2].set_title("Difference (%)")
    plt.colorbar(im2, ax=axs[2])

    rel_error = np.max(np.abs(diff)) / np.max(np.abs(A))
    plt.suptitle(f"Relative Error: {rel_error * 100:.2f}%")

    plt.tight_layout()
    plt.show()
