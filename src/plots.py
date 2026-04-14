import matplotlib.pyplot as plt
import numpy as np

class Plotting:
  
  def load(self, path: str, height: int, weight: int) -> np.ndarray:
    return np.fromfile(
      path, dtype=np.float32, count=height*weight
    ).reshape([height, weight], order='F')

  def plot(self, model, perc=99) -> None:
    _, ax = plt.subplots(figsize=(12, 5))

    vmin = np.percentile(model, 100 - perc)
    vmax = np.percentile(model, perc)

    img = ax.imshow(
      model, aspect="auto", cmap="Greys", 
      vmin=vmin, vmax=vmax
    )

    plt.colorbar(img, ax=ax)

    plt.tight_layout()
    plt.show()

  def compare(
    self, 
    model1: np.ndarray, 
    model2: np.ndarray, 
    title1=None,
    title2=None,
    ) -> None:

    if title1 is None:
      title1 = "Image 1"

    if title2 is None:
      title2 = "Image 2"

    perc = 99

    vmin1 = np.percentile(model1, 100 - perc)
    vmax1 = np.percentile(model1, perc)

    vmin2 = np.percentile(model2, 100 - perc)
    vmax2 = np.percentile(model2, perc)

    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(17, 6))

    im0 = axs[0].imshow(
      model1, aspect='auto', cmap="Greys", 
      vmin=vmin1, vmax=vmax1
    )
    axs[0].set_title(title1)
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(
      model2, aspect='auto', cmap="Greys", 
      vmin=vmin2, vmax=vmax2
    )
    axs[1].set_title(title2)
    plt.colorbar(im1, ax=axs[1])

    plt.tight_layout()
    plt.show()

  def compare_model_and_traces(
    self, 
    model1: np.ndarray, 
    model2: np.ndarray, 
    trace_number: int,
    title1=None,
    title2=None,
    ) -> None:

    if title1 is None:
      title1 = "Image 1"

    if title2 is None:
      title2 = "Image 2"

    nz, nx = model1.shape

    perc = 99

    vmin1 = np.percentile(model1, 100 - perc)
    vmax1 = np.percentile(model1, perc)

    vmin2 = np.percentile(model2, 100 - perc)
    vmax2 = np.percentile(model2, perc)

    _, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    im0 = axs[0].imshow(
      model1, aspect='auto', cmap="Greys", 
      vmin=vmin1, vmax=vmax1
    )
    axs[0].set_title(title1)
    plt.colorbar(im0, ax=axs[0])

    axs[0].plot(np.full(nz, trace_number), np.arange(nz), 'b--')

    im1 = axs[1].imshow(
      model2, aspect='auto', cmap="Greys", 
      vmin=vmin2, vmax=vmax2
    )
    axs[1].set_title(title2)
    plt.colorbar(im1, ax=axs[1])

    axs[1].plot(np.full(nz, trace_number), np.arange(nz), 'r--')

    axs[2].plot(model1[:, trace_number], np.arange(nz), label=f"{title1} Trace")
    axs[2].plot(-model2[:, trace_number], np.arange(nz), label=f"{title2} Trace")

    axs[2].invert_yaxis() 
    axs[2].legend()

    plt.tight_layout()
    plt.show()

  def compare_diff(
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

    rel_error = np.max(np.abs(diff)) / np.max(np.abs(diff_norm))
    plt.suptitle(f"Relative Error: {rel_error * 100:.2f}%")

    plt.tight_layout()
    plt.show()
