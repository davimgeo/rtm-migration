from __future__ import annotations

import sys
import os

os.chdir(os.path.join(os.path.dirname(sys.argv[0]), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======================================================== #
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from src import Config, Geometry, Wavelet

@njit(parallel=True, fastmath=True)
def smooth_kernel(sigma, amp, base_model, R2, nz, nx, epsilon=1e-9):
    gaussian = np.empty((nz, nx), np.float32)
    arg = 1.0 / (2.0 * (sigma + epsilon))**2

    for i in prange(nz):
        for j in range(nx):
            gaussian[i, j] = amp * np.exp(-R2[i, j] * arg)

    return (1 + gaussian) * base_model

def build_model_cube(sigmas, amps, base_model, R2):
    nz, nx = base_model.shape
    cube = np.zeros((len(sigmas), nz, nx), np.float32)

    for i, sigma in enumerate(sigmas):
        amp = amps[i]
        cube[i] = smooth_kernel(sigma, amp, base_model, R2, nz, nx)

    return cube

@njit(parallel=True, fastmath=True)
def _forward_kernel(
    upas, upre, ufut,
    damp_x, damp_z,
    inv_dh2, nzz, nxx,
    ricker, ix, iz,
    dh2, arg, t
):
    upre[iz, ix] += ricker[t] / dh2

    for i in prange(4, nzz-4):
        for j in range(4, nxx-4):

            d2u_dx2 = (
              -9*upre[i-4,j] +128*upre[i-3,j] -1008*upre[i-2,j]
              +8064*upre[i-1,j] -14350*upre[i,j]
              +8064*upre[i+1,j] -1008*upre[i+2,j]
              +128*upre[i+3,j] -9*upre[i+4,j]
            )

            d2u_dz2 = (
              -9*upre[i,j-4] +128*upre[i,j-3] -1008*upre[i,j-2]
              +8064*upre[i,j-1] -14350*upre[i,j]
              +8064*upre[i,j+1] -1008*upre[i,j+2]
              +128*upre[i,j+3] -9*upre[i,j+4]
            )

            laplacian = (d2u_dx2 + d2u_dz2) * inv_dh2
            upas[i,j] = arg[i,j]*laplacian + 2*upre[i,j] - ufut[i,j]

    for i in prange(4, nzz-4):
        for j in range(4, nxx-4):
            damp = damp_x[j] * damp_z[i]
            ufut[i,j] = upre[i,j] * damp
            upre[i,j] = upas[i,j] * damp


# =========================================================
# Forward modeling (stateless)
# =========================================================
def fdm_propagation(
    u_past, u_present, u_future,
    seismogram,
    damp_x, damp_z,
    kernel_arg,
    wavelet,
    ix, iz,
    nt, nzz, nxx,
    recx, recz
):
    seismogram.fill(0)
    u_past.fill(0)
    u_present.fill(0)
    u_future.fill(0)

    for t in range(1, nt-1):

        _forward_kernel(
            u_past, u_present, u_future,
            damp_x, damp_z,
            kernel_arg.inv_dh2,
            nzz, nxx,
            wavelet,
            ix, iz,
            kernel_arg.dh2,
            kernel_arg.velocity_term,
            t
        )

        for r in range(len(recx)):
            seismogram[t, r] = u_present[recz[r], recx[r]]

nz, nx = 300, 900
value = 2500
center = (nz//2, nx//2)

ref_sigma, ref_amp = 40, 0.40
vmin, vmax = 0, 80
size = 51

sigma = np.linspace(vmin, vmax, size)
amp   = np.linspace(vmin, vmax, size)

# build radius grid
x = np.arange(nz)
y = np.arange(nx)
X, Y = np.meshgrid(x, y, indexing="ij")
R2 = (X-center[0])**2 + (Y-center[1])**2

model0 = np.full((nz, nx), value, np.float32)
baseModel = smooth_kernel(ref_sigma, ref_amp, model0, R2, nz, nx)
cubeOfVaryingModels = build_model_cube(sigma, amp, baseModel, R2)

# --- load acquisition
config = Config().load()
geom = Geometry(config); geom.get()
wavelet = Wavelet(config); wavelet.get()

ix = ((nx*10) + config.nb)//2
iz = 8

# --- allocate buffers ONCE
shape = (nz, nx)
u_past    = np.zeros(shape, np.float32)
u_present = np.zeros(shape, np.float32)
u_future  = np.zeros(shape, np.float32)

seismogram = np.zeros((config.nt, geom.nrec), np.float32)

damp_x = np.ones(nx, np.float32)
damp_z = np.ones(nz, np.float32)

# --- reference data
kernel_ref = KernelArguments.make(config.dh, config.dt, baseModel)
fdm_propagation(
    u_past, u_present, u_future,
    seismogram,
    damp_x, damp_z,
    kernel_ref,
    wavelet.wavelet,
    ix, iz,
    config.nt, nz, nx,
    geom.recx, geom.recz
)
d_obs = seismogram.copy()
plt.imshow(d_obs)
plt.show()

l2 = np.zeros(len(cubeOfVaryingModels))

for i, circle in enumerate(cubeOfVaryingModels):

    seismogram_new = np.zeros((config.nt, geom.nrec), np.float32)

    kernel_arg = KernelArguments.make(config.dh, config.dt, circle)

    fdm_propagation(
        u_past, u_present, u_future,
        seismogram_new,
        damp_x, damp_z,
        kernel_arg,
        wavelet.wavelet,
        ix, iz,
        config.nt, nz, nx,
        geom.recx, geom.recz
    )

    d_calc = seismogram_new.copy()
    l2[i] = np.sqrt(np.sum((d_obs - d_calc)**2))
    print(i, l2[i])

plt.plot(l2)
plt.title("Misfit curve")
plt.grid()
plt.show()
