from __future__ import annotations

import sys
import os

os.chdir(os.path.join(os.path.dirname(sys.argv[0]), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======================================================== #

import numpy as np
import matplotlib.pyplot as plt

from src import Config, Geometry, Wavelet, Seismogram
from fdm import *

import numpy as np

# v(alpha, v0) = v_0 + alpha * depth

def get_model(nz, nx, v0, alpha):
  model = np.zeros((nz, nx))

  i, j = np.ogrid[:nz, :nx]

  model[i, j] = v0 + i / alpha

  return model

# ====== get circles/basemodel ======
nz, nx = 101, 201
v0 = 2500

ref_alpha = 0.3

size = 21

alpha_min = 0.1
alpha_max = 5

alphas = np.linspace(alpha_min, alpha_max, size)
idx = np.abs(alphas - ref_alpha).argmin()
alphas[idx] = ref_alpha

vmin = 1500
vmax = 3500

initial_velocities = np.linspace(vmin, vmax, size)
idx = np.abs(initial_velocities - v0).argmin()
initial_velocities[idx] = v0

base_model = get_model(nz, nx, v0, ref_alpha)

# ====== global parameters ======

config = Config().load()
geom = Geometry(config); geom.get()
wavelet = Wavelet(config); wavelet.get()

nzz = nz + 2 * config.nb
nxx = nx + 2 * config.nb

ix = int(geom.srcxId[0]) + config.nb
iz = int(geom.srczId[0]) + config.nb

shape = (nzz, nxx)

damp_x, damp_z = get_damp(nzz, nxx, config.nb, config.factor)

upas = np.zeros(shape)
upre = np.zeros(shape)
ufut = np.zeros(shape)

seismogram = np.zeros((config.nt, geom.nrec))
# ====== get d_obs ======

base_model_ext = set_boundary(base_model, nzz, nxx, config.nb)

dh2 = config.dh**2
inv_dh2 = 1.0 / (5040.0 * dh2)
arg = config.dt**2 * base_model_ext**2

nsnaps = 101

snapshots = []
snap_ratio = max(1, (config.nt - 1) // nsnaps)

d_calc = np.ndarray
for t in range(1, config.nt - 1):

  d_calc = fdm_propagation(
    upas, upre, ufut, seismogram,
    damp_x, damp_z, dh2, inv_dh2,
    arg, wavelet.wavelet, ix, iz,
    nzz, nxx, geom.recx, geom.recz,
    config.nb, t
  )

  save_snapshots(snapshots, upre, snap_ratio, t)

plot_snapshots(
  snapshots, base_model_ext, nx, nz, 
  config.nb, config.dh, geom.recx, 
  geom.recz, geom.srcxId, geom.srczId, 
  config.nt, config.dt
)

seis = Seismogram(config, geom)
seis.plot(d_calc)

# ====== l2 norm for each circle ======
d_obs = np.ndarray

seismogram_loop = np.zeros((config.nt, geom.nrec))

import time

start = time.time()

alpha_grid, vel_grid = np.meshgrid(np.linspace(alpha_min, alpha_max, size),
                                   np.linspace(vmin, vmax, size))

l2 = np.zeros((size, size))

for i in range(size):
  for j in range(size):
    seismogram_loop.fill(0.0)
    upas.fill(0.0)
    upre.fill(0.0)
    ufut.fill(0.0)

    model = get_model(nz, nx, vel_grid[i, j], alpha_grid[i, j])

    model_ext = set_boundary(model, nzz, nxx, config.nb)

    arg = config.dt**2 * model_ext**2

    for t in range(1, config.nt - 1):

      d_obs = fdm_propagation(
        upas, upre, ufut, seismogram_loop,
        damp_x, damp_z, dh2, inv_dh2,
        arg, wavelet.wavelet, ix, iz,
        nzz, nxx, geom.recx, geom.recz,
        config.nb, t
      )

    l2[i, j] = np.sqrt(np.sum((d_obs - d_calc)**2))

end = time.time()
print(f"Runtime: {round(end - start, 4)} seconds")

# === plot 3d ===

fig, ax = plt.subplots(
  figsize=(10, 8),
  subplot_kw={"projection": "3d"}
)

surf1 = ax.plot_surface(
  alpha_grid, vel_grid, l2,
  cmap="viridis",
  alpha=0.8
)

fig.colorbar(surf1, shrink=0.5, aspect=5)

ax.set_box_aspect([1, 1, 1])

ax.set_xlabel(r'$\alpha$', fontsize=13)
ax.set_ylabel(r'$v_0$', fontsize=13)
ax.set_zlabel("L2", fontsize=13)

ax.view_init(elev=15, azim=4)

#ax.legend(loc="lower right")
plt.show()

# === plot 2d ===

plt.imshow(l2)

plt.title("L2", fontsize=13)
plt.xlabel(r"$\alpha$", fontsize=13)
plt.ylabel(r"$v_0$", fontsize=13)

plt.colorbar()
plt.tight_layout()
plt.show()

# save l2
path = "cube_l2.bin"

l2.flatten('F').astype('float32', order='F').tofile(path)
print(f"Successfully saved: {path}")
