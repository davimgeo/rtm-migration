from __future__ import annotations

import sys
import os

os.chdir(os.path.join(os.path.dirname(sys.argv[0]), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======================================================== #

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from src import Config, Geometry, Wavelet, Seismogram
from fdm import fdm_propagation, get_damp, set_boundary

import numpy as np

def set_boundary(model, nzz, nxx, nb) -> np.ndarray:

  nz = nzz - 2*nb
  nx = nxx - 2*nb

  model_ext = np.zeros((nzz, nxx))

  for j in range(nx):
    for i in range(nz):
      model_ext[i + nb, j + nb] = model[i, j]

  for j in range(nb, nx + nb):
    for i in range(nb):
      model_ext[i, j] = model_ext[nb, j]
      model_ext[nz + nb + i, j] = model_ext[nz + nb - 1, j]

  for i in range(nzz):
    for j in range(nb):
      model_ext[i, j] = model_ext[i, nb]
      model_ext[i, nx + nb + j] = model_ext[i, nx + nb - 1]

  return model_ext

def get_damp(nzz, nxx, nb, factor):
  damp_x = np.zeros(nxx)
  damp_z = np.zeros(nzz)

  nz = nzz - 2*nb
  nx = nxx - 2*nb

  for i in range(nzz):

    if nb <= i < nb + nz:
      damp_z[i] = 1.0

    elif i < nb:
      d = nb - i
      damp_z[i] = np.exp(-(factor * d) * (factor * d))

    else:
      d = i - (nb + nz - 1)
      damp_z[i] = np.exp(-(factor * d) * (factor * d))

  for j in range(nxx):

    if nb <= j < nb + nx:
      damp_x[j] = 1.0

    elif j < nb:
      d = nb - j
      damp_x[j] = np.exp(-(factor * d) * (factor * d))

    else:
      d = j - (nb + nx - 1)
      damp_x[j] = np.exp(-(factor * d) * (factor * d))

  return damp_x, damp_z

@njit(parallel=True, fastmath=True)
def fdm_propagation(
    upas, upre, ufut,
    seismogram,
    damp_x, damp_z,
    dh2, inv_dh2, arg,
    ricker, ix, iz,
    nzz, nxx, recx, recz,
    nb, t
):

    upre[iz, ix] += ricker[t] / dh2

    for i in prange(4, nzz - 4):
      for j in range(4, nxx - 4):
        d2u_dx2 = (
          -9.0   * upre[i-4, j] + 128.0   * upre[i-3, j] - 1008.0 * upre[i-2, j] +
          8064.0 * upre[i-1, j] - 14350.0 * upre[i,   j] + 8064.0 * upre[i+1, j] -
          1008.0 * upre[i+2, j] + 128.0   * upre[i+3, j] - 9.0    * upre[i+4, j]
        )

        d2u_dz2 = (
          -9.0   * upre[i, j-4] + 128.0   * upre[i, j-3] - 1008.0 * upre[i, j-2] +
          8064.0 * upre[i, j-1] - 14350.0 * upre[i, j]   + 8064.0 * upre[i, j+1] -
          1008.0 * upre[i, j+2] + 128.0   * upre[i, j+3] - 9.0    * upre[i, j+4]
        )

        laplacian = (d2u_dx2 + d2u_dz2) * inv_dh2

        upas[i, j] = arg[i, j] * laplacian + 2.0 * upre[i, j] - ufut[i, j]
      
    for i in prange(4, nzz - 4):
      for j in range(4, nxx - 4):
        damp = damp_x[j] * damp_z[i]
    
        ufut[i, j] = upre[i, j] * damp
        upre[i, j] = upas[i, j] * damp

    for irec in range(len(recx)):
      rx = int(recx[irec]) + nb
      rz = int(recz[irec]) + nb
      seismogram[t, irec] = upre[rz, rx]

    return seismogram

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
nz, nx = 300, 900
v0 = 2500

ref_alpha = 0.3

size = 51

alpha_min = 0.1
alpha_max = 0.9

alphas = np.linspace(alpha_min, alpha_max, size)

base_model = get_model(nz, nx, v0, ref_alpha)
model_cube = get_models_cube(alphas, v0, nz, nx)

plt.imshow(base_model)
plt.colorbar()
plt.show()
# ====== global parameters ======

config = Config().load()
geom = Geometry(config); geom.get()
wavelet = Wavelet(config); wavelet.get()

nzz = nz + 2 * config.nb
nxx = nx + 2 * config.nb

ix = int(nxx / 2) + config.nb
iz = 8

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

d_calc = np.ndarray
for t in range(1, config.nt - 1):

  d_calc = fdm_propagation(
    upas, upre, ufut, seismogram,
    damp_x, damp_z, dh2, inv_dh2,
    arg, wavelet.wavelet, ix, iz,
    nzz, nxx, geom.recx, geom.recz,
    config.nb, t
  )

seis = Seismogram(config, geom)
seis.plot(d_calc)

# ====== l2 norm for each circle ======
d_obs = np.ndarray

seismogram_loop = np.zeros((config.nt, geom.nrec))

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

    if not t % 400:
      plt.imshow(upre)
      plt.show()

      plt.imshow(circle_ext)
      plt.show()

  seis.plot(d_obs)

  l2[i] = np.sqrt(np.sum((d_obs - d_calc)**2))
  print(i, l2[i])

