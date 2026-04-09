import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from matplotlib import animation

def plot_snapshots(
    snapshots: list,    
    model: np.ndarray,       
    nx: int, nz: int, nb: int, dh: float, 
    recx: np.ndarray, recz: np.ndarray, 
    srcxId: np.ndarray, srczId: np.ndarray, 
    nt: int, dt: float
) -> None:

    xloc = np.linspace(0, nx-1, 11, dtype=int)
    xlab = np.array(xloc * dh, dtype=int)

    zloc = np.linspace(0, nz-1, 7, dtype=int)
    zlab = np.array(zloc * dh, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 5))
    ims = []

    for snap in snapshots:
        scale = 2.0 * np.std(snap)

        model_frame = ax.imshow(
            model[nb:nb+nz, nb:nb+nx],
            aspect="auto",
            cmap="jet",
            alpha=0.5
        )

        snap_frame = ax.imshow(
            snap[nb:nb+nz, nb:nb+nx],
            aspect="auto",
            cmap="Greys",
            vmin=-scale,
            vmax=scale,
            alpha=0.7
        )

        ax.plot(recx, recz, 'bv')
        ax.plot(srcxId, srczId, 'r*')

        ims.append([model_frame, snap_frame])

    ani = animation.ArtistAnimation(
        fig, ims,
        interval=(nt / len(snapshots) + 1) * dt * 1e3,
        blit=False,
        repeat_delay=0
    )

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)
    ax.set_yticks(zloc)
    ax.set_yticklabels(zlab)

    plt.show()
    return ani

def save_snapshots(
  snapshots: list, upre: np.ndarray,
  snap_ratio: int, t: int
  ) -> None:

  if not t % snap_ratio:
    snapshots.append(upre.copy())

def plot_model_and_geometry(
    model: np.ndarray,
    nx: int, nz: int, nb: int, dh: int,
    recx: int, recz: int,
    srcxId: int, srczId: int
) -> None:
    xloc = np.linspace(0, nx - 1, 11, dtype=int)
    xlab = np.array(xloc * dh, dtype=int)

    zloc = np.linspace(0, nz - 1, 7, dtype=int)
    zlab = np.array(zloc * dh, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 5))

    img = ax.imshow(
        model[nb:nb + nz, nb:nb + nx],
        aspect="auto",
        cmap="jet",
    )

    ax.plot(recx, recz, 'bv', label="Receivers")
    ax.plot(srcxId, srczId, 'r*', markersize=12, label="Source")

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)
    ax.set_yticks(zloc)
    ax.set_yticklabels(zlab)

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Velocity Model")

    plt.colorbar(img, ax=ax, label="VP [m/s]")
    ax.legend()

    plt.show()

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

