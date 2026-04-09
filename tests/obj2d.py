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

# v(alpha) = v_0 + alpha * depth

def get_model(nz, nx, v0, alpha):
  model = np.zeros((nz, nx))

  i, j = np.ogrid[:nz, :nx]

  model[i, j] = v0 + i / alpha

  return model

def get_models_cube(alphas, v0, nz, nx):

  models_cube = np.zeros((len(alphas), nz, nx))

  i, j = np.ogrid[:nz, :nx]

  for k, alpha in enumerate(alphas):
    models_cube[k, i, j] = v0 + i / alpha

  return models_cube

# ====== get circles/basemodel ======
nz, nx = 101, 201
v0 = 2500

ref_alpha = 0.3

size = 51

alpha_min = 0.1
alpha_max = 0.9

alphas = np.linspace(alpha_min, alpha_max, size)
idx = np.abs(alphas - ref_alpha).argmin()
# include ref_alpha in the problem
alphas[idx] = ref_alpha

base_model = get_model(nz, nx, v0, ref_alpha)
model_cube = get_models_cube(alphas, v0, nz, nx)

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

l2 = np.zeros(len(alphas))
for i, circle in enumerate(model_cube):
  seismogram_loop.fill(0.0)
  upas.fill(0.0)
  upre.fill(0.0)
  ufut.fill(0.0)

  circle_ext = set_boundary(circle, nzz, nxx, config.nb)

  arg = config.dt**2 * circle_ext**2

  for t in range(1, config.nt - 1):

    d_obs = fdm_propagation(
      upas, upre, ufut, seismogram_loop,
      damp_x, damp_z, dh2, inv_dh2,
      arg, wavelet.wavelet, ix, iz,
      nzz, nxx, geom.recx, geom.recz,
      config.nb, t
    )

  #seis.plot(d_obs)

  l2[i] = np.sqrt(np.sum((d_obs - d_calc)**2))

end = time.time()
print(f"Runtime: {round(end - start, 4)} seconds")

# save l2
np.savetxt("l2.txt", l2, delimiter=",")

min_alpha = np.argmin(l2)

print(f"Reference alpha: {ref_alpha}, Min alpha: {alphas[min_alpha]}")

plt.plot(alphas, l2)
plt.show()

