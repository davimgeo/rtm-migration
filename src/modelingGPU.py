from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
  from src import (
    Config, Model, Seismogram, 
    Wavelet, Geometry
  )

from numpy import numpy.float32
import cupy as cp

class ModelingGPU:

    def __init__(
        self, config: Config, 
        model: Model, 
        seismogram: Seismogram, 
        wavelet: Wavelet, 
        geometry: Geometry
    ) -> None:

        self.c = config
        self.m = model
        self.g = geometry
        self.s = seismogram
        self.w = wavelet

        shape = (self.m.nzz, self.m.nxx)

        self.u = Wavefield(shape)
        self.u_homo = Wavefield(shape)

        self.laplacian = cp.zeros(shape, dtype=cp.float32)
        self.laplacian_homo = cp.zeros(shape, dtype=cp.float32)

        model_homo = cp.full(shape, 1500, dtype=cp.float32)
        self.kernel_arg = KernelArguments(
            dh2=self.c.dh ** 2,
            inv_dh2=1.0 / (5040.0 * (self.c.dh ** 2)),
            velocity_term=(self.c.dt ** 2 * self.m.model ** 2).astype(cp.float32)
        )

        self.kernel_arg_homo = KernelArguments(
            dh2=self.c.dh ** 2,
            inv_dh2=1.0 / (5040.0 * (self.c.dh ** 2)),
            velocity_term=(self.c.dt ** 2 * model_homo ** 2).astype(cp.float32)
        )

        self.block = (16, 16)
        self.grid = ((model.nxx + self.block[0] - 1) // self.block[0],
                    (model.nzz + self.block[1] - 1) // self.block[1]) 

        # move to device
        self.ricker_gpu = cp.asarray(self.w.wavelet)

        self.damp_x_gpu = cp.zeros(self.m.nzz)
        self.damp_z_gpu = cp.zeros(self.m.nzz)

        self.recx_gpu = cp.asarray(self.g.recx)
        self.recz_gpu = cp.asarray(self.g.recz)

        self.seismogram_gpu = cp.asarray(self.s.seismogram)
        self.seismogram_homo_gpu = cp.asarray(self.s.seismogram_homo)

        self.snaps = SnapshotManager.from_config(self.c, shape)

        # Contadores internos
        self.ix, self.iz = 0, 0
        self.snap_id_src = 0

  def remove_direct_wave_model(self, ix: int, iz: int) -> None:
    self.ix, self.iz = ix, iz
    
    self.zero_out_matrices()

    _forward_kernel(self.block, self.grid, (
            self.u.past, self.u.present, self.u.future, 
            self.laplacian, self.damp_x_gpu, self.damp_z_gpu,
            self.inv_dh2, self.nzz, self.nxx, self.ricker_gpu, self.ix, self.iz,
            self.dh2, self.velocity_term, self.nt, self.seismogram_gpu,
            self.recx, self.recz, self.nrec
        )
    )

    _forward_kernel(
        self.block, self.grid, (
            self.u.past_homo, self.u.present_homo, self.u.future_homo, self.laplacian_homo, 
            self.damp_x_gpu, self.damp_z_gpu, self.inv_dh2, self.nzz, 
            self.nxx, self.ricker_gpu, self.ix, self.iz, self.dh2, 
            self.kernel_arg_homo.velocity_term_homo, self.nt, self.seismogram_homo_gpu,
            self.recx_gpu, self.recz_gpu, self.nrec
        )
    )

    self.seis.seismogram -= self.seis.seismogram_homo

   def zero_out_matrices(self):
        self.s.seismogram.fill(0.0)
        self.s.seismogram_homo.fill(0.0)

        self.u.past.fill(0.0)
        self.u.present.fill(0.0)
        self.u.future.fill(0.0)
        self.u.past_homo.fill(0.0)
        self.u.present_homo.fill(0.0)
        self.u.future_homo.fill(0.0)

@dataclass(slots=True)
class Wavefield:
  shape: Tuple[int, int]
  past: cp.ndarray = field(init=False)
  present: cp.ndarray = field(init=False)
  future: cp.ndarray = field(init=False)

  def __post_init__(self):
    self.past = cp.zeros(self.shape, dtype=cp.float32)
    self.present = cp.zeros(self.shape, dtype=cp.float32)
    self.future = cp.zeros(self.shape, dtype=cp.float32)


@dataclass(slots=True)
class KernelArguments:
  dh2: float | numpy.float32
  inv_dh2: float | numpy.float32
  velocity_term: cp.ndarray  


@dataclass(slots=True)
class SnapshotManager:
  nsnaps: int
  tstop: int
  ratio: int
  dt: float
  src: cp.ndarray
  rec: cp.ndarray
  
  # internal counters
  current_src_id: int = 0
  current_rec_id: int = 0

  @classmethod
  def from_config(cls, c, shape):
    tstop = int(1.7 * (c.tlag / c.dt))
    
    if c.snap_num_nyquist:
      ratio = int(1 / (4 * c.fmax * c.dt))
    else:
      ratio = int(c.nt / c.snap_num)

    nsnaps = int((c.nt - tstop - 1) / ratio) + 1
    
    return cls(
      nsnaps=nsnaps,
      tstop=tstop,
      ratio=ratio,
      dt=cp.float32(ratio * c.dt),
      src=cp.zeros((nsnaps, *shape)),
      rec=cp.zeros((nsnaps, *shape)),
      current_rec_id=nsnaps - 1
    )

_forward_kernel = cp.RawKernel(r'''
extern "C" __global__
void forward_kernel(
   float* upas, float* upre, float* ufut,
   float* laplacian, float* damp_x, float* damp_z,
   float inv_dh2, int nzz, int nxx, float* ricker,
   int ix, int iz, float dh2, float* velocity_term,
   int nt, float* seismogram, float* recx, float* recz,
   int nrec, int dh2
)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if ((t > 1) && (t < nt - 1))
    {
        upre[iz, ix] += ricker[t] / dh2;

        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.y + threadIdx.y;

        if ((i > 3 && i < nzz - 4) && (j > 3 && j < nxx - 4))
        {
            float d2u_dx2 = (
                -9.0   * upre[(i-4) * nxx + j] +
                128.0  * upre[(i-3) * nxx + j] -
                1008.0 * upre[(i-2) * nxx + j] +
                8064.0 * upre[(i-1) * nxx + j] -
                14350.0* upre[(i)   * nxx + j] +
                8064.0 * upre[(i+1) * nxx + j] -
                1008.0 * upre[(i+2) * nxx + j] +
                128.0  * upre[(i+3) * nxx + j] -
                9.0    * upre[(i+4) * nxx + j]
            );

            float d2u_dz2 = (
                -9.0   * upre[i * nxx + (j-4)] +
                128.0  * upre[i * nxx + (j-3)] -
                1008.0 * upre[i * nxx + (j-2)] +
                8064.0 * upre[i * nxx + (j-1)] -
                14350.0* upre[i * nxx + (j)] +
                8064.0 * upre[i * nxx + (j+1)] -
                1008.0 * upre[i * nxx + (j+2)] +
                128.0  * upre[i * nxx + (j+3)] -
                9.0    * upre[i * nxx + (j+4)]
            );

            laplacian[i * nxx + j] = (d2u_dx2 + d2u_dz2) * inv_dh2;
        }
        
        int k = blockIdx.y * blockDim.y + threadIdx.y;
        int l = blockIdx.x * blockDim.x + threadIdx.x;

        if ((k > 3 && k < nzz - 4) && (l > 3 && l < nxx - 4))
        {
            int idx = k * nxx + l;

            upas[idx] = arg[idx] * laplacian[idx] + 2.0 * upre[idx] - ufut[idx];

            float damp = damp_x[l] * damp_z[k];

            ufut[idx] = upre[idx] * damp;
            upre[idx] = upas[idx] * damp;
        }

        int irec = blockIdx.x * blockDim.x + threadIdx.x;

        if (irec < nrec)
        {
            int rx = recx[irec] + nb;
            int rz = recz[irec] + nb;
            seismogram[t, irec] = upre[rz * nxx + rx];
        }

    }
}
''', 'forward_kernel')




