import sys
import os

os.chdir(os.path.join(os.path.dirname(sys.argv[0]), ".."))

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "data/output/"

nx, nz = 200, 100

image_full = np.fromfile(
    OUTPUT_PATH + "image_full_200x100.bin", dtype=np.float32,
    count=nx*nz
).reshape([nz, nx], order='F')

image_700snaps = np.fromfile(
    OUTPUT_PATH + "image_700snaps.bin_200x100.bin", dtype=np.float32,
    count=nx*nz
).reshape([nz, nx], order='F')

diff = image_full - image_700snaps
diff_norm = diff / np.max(np.abs(image_full))

vmin = min(image_full.min(), image_700snaps.min())
vmax = max(image_full.max(), image_700snaps.max())

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

im0 = axs[0].imshow(image_full, aspect='auto', cmap="Greys", vmin=vmin, vmax=vmax)
axs[0].set_title("Image Full")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(image_700snaps, aspect='auto', cmap="Greys", vmin=vmin, vmax=vmax)
axs[1].set_title("Image 700 Snaps")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(diff_norm, aspect='auto', cmap="Greys")
axs[2].set_title("Difference (%)")
cbar = plt.colorbar(im2, ax=axs[2])

rel_error = np.max(np.abs(diff)) / np.max(np.abs(image_full))
plt.suptitle(f"Relative Error: {rel_error * 100:.2f}%")

plt.tight_layout()
plt.show()

runtime_full = 81.2951
runtime_700 = 10.5836

plt.bar(["Full", "700 snaps"], [runtime_full, runtime_700])

plt.suptitle(f"Percentual Difference: {((runtime_full - runtime_700) / runtime_full * 100):.2f}%")
plt.title("Runtime Comparison")
plt.ylabel("Time [s]")

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()