from __future__ import annotations

from dataclasses import dataclass, field

from os import system
from typing import TYPE_CHECKING, Tuple

from numpy import numpy.float32
import cupy as cp

if TYPE_CHECKING:
    from src import (
        Config,
        Modeling,
        Model,
        Seismogram,
        Wavelet,
        Geometry,
    )

from src import Migration

OUTPUT_PATH = "data/output/"

class MigrationGPU(Migration):
  def __init__(
    self, config: Config, 
    modeling: Modeling, 
    model: Model, 
    seismogram: Seismogram, 
    wavelet: Wavelet, 
    geometry: Geometry
) -> None:

    self.c = config
    self.m = model
    self.g = geometry
    self.md = modeling
    self.s = seismogram
    self.w = wavelet
    
    shape = (model.nzz, model.nxx)

    self.u = Wavefield(shape)
    self.d = Wavefield(shape)

    self.laplacian = cp.zeros(shape, dtype=cp.float32)
    self.image = cp.zeros(shape, dtype=cp.float32)
    self.gradient = cp.zeros(shape, dtype=cp.float32)
    self.num = cp.zeros(shape, dtype=cp.float32)
    self.den = cp.zeros(shape, dtype=cp.float32)

    dh2 = config.dh**2

    velocity_term = cp.asarray(model.model_smooth, dtype=cp.float32)
    velocity_term = (config.dt**2) * velocity_term**2

    self.kernel_args = KernelArguments(
    dh2=cp.float32(dh2),
    inv_dh2=cp.float32(1.0 / (5040.0 * dh2)),
    velocity_term=velocity_term
    )

    self.snaps = SnapshotManager.from_config(config, shape)
    self.snap_idx = cp.arange(self.snaps.tstop, config.nt - 1, self.snaps.ratio)

    self.block = (16, 16)
    self.grid = ((model.nxx + self.block[0] - 1) // self.block[0],
                (model.nzz + self.block[1] - 1) // self.block[1]) 

    # copying into device
    self.ricker_gpu = cp.asarray(self.w.wavelet)
    self.damp_x_gpu = cp.asarray(self.md.damp_x)
    self.damp_z_gpu = cp.asarray(self.md.damp_z)
    self.recx_gpu = cp.asarray(self.g.recx)
    self.recz_gpu = cp.asarray(self.g.recz)
    self.seismogram_gpu = cp.asarray(self.s.seismogram)

    # internal counters
    self.ix, self.iz = 0, 0
    self.current_step = 1

  def rtm(self):
    for isrc in range(len(self.g.srcxId)):
        self.zero_out_matrices()

        self.define_source_coordinates(isrc)

        self.md.remove_direct_wave_model()

        self.forward_propagation()

        self.backward_propagation()

        self.image_condition()

        self.show_modeling_status()

    if self.c.is_laplacian:
      self.laplacian_filter()

    if self.c.save_image:
      self.save()

  def zero_out_matrices(self):
    self.s.seismogram.fill(0.0)

    self.u.past.fill(0.0)
    self.u.present.fill(0.0)
    self.u.future.fill(0.0)

    self.d.past.fill(0.0)
    self.d.present.fill(0.0)
    self.d.future.fill(0.0)

    self.snap_id_src = 0

    self.num.fill(0.0)
    #self.den.fill(0.0)

  def define_source_coordinates(self, isrc: int):
    self.ix = int(self.g.srcxId[isrc]) + self.c.nb
    self.iz = int(self.g.srczId[isrc]) + self.c.nb

  def forward_propagation(self):
    _forward_kernel(self.grid, self.block, (
            self.u.past, self.u.present, self.u.future, self.laplacian, 
            self.damp_x_gpu, self.damp_z_gpu, self.kernel_args.inv_dh2, 
            self.m.nzz, self.m.nxx, self.ricker_gpu, self.ix, self.iz,
            self.kernel_args.dh2, self.kernel_args.velocity_term, self.c.nt,
            self.snaps.ratio, self.snaps.src
        )
    )

  def backward_propagation(self):
    _backward_kernel(self.grid, self.block, (
            self.d.past, self.d.present, self.d.future, self.laplacian, 
            self.damp_x_gpu, self.damp_z_gpu, self.kernel_args.inv_dh2, 
            self.m.nzz, self.m.nxx, self.kernel_args.dh2, self.kernel_args.velocity_term,
            self.recx_gpu, self.recz_gpu, self.c.nb, self.seismogram_gpu, self.g.nrec,
            self.snaps.tstop, self.snaps.ratio
        )
    )

  def image_condition(self):
    _image_kernel(
        self.grid,
        self.block,
        (
            self.image,
            self.num,
            self.snaps.dt
        )
    )

  def laplacian_filter(self):
    inv_dh = 1.0 / (12.0 * self.c.dh * self.c.dh)

    for i in range(2, self.m.nzz - 2):
      for j in range(2, self.m.nxx - 2):
        d2u_dx2 = (
            - self.image[i-2, j]
            + 16.0 * self.image[i-1, j]
            - 30.0 * self.image[i, j]
            + 16.0 * self.image[i+1, j]
            - self.image[i+2, j]
        ) * inv_dh

        d2u_dz2 = (
            - self.image[i, j-2]
            + 16.0 * self.image[i, j-1]
            - 30.0 * self.image[i, j]
            + 16.0 * self.image[i, j+1]
            - self.image[i, j+2]
        ) * inv_dh

        self.gradient[i, j] = d2u_dx2 + d2u_dz2

    self.image = self.gradient

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
   int nt, int snap_ratio, float* snaps, int tstop
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

        int snap_idx = (t - tstop) / snap_ratio;

        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.y + threadIdx.y;

        if ((i > 3 && i < nzz - 4) && (j > 3 && j < nxx - 4))
        {
            snaps[snap_idx * nzz * nxx + i * nxx + j] = upre[i * nxx + j];    
        }
        
    }
}
''', 'forward_kernel')

_backward_kernel = cp.RawKernel(r'''
extern "C" __global__
void backward_kernel(
   float* depas, float* depre, float* defut,
   float* laplacian, float* damp_x, float* damp_z,
   float inv_dh2, int nzz, int nxx, float dh2,
   float* arg, float* recx, float* recz, int nb,
   float* seismogram, int nrec, int tstop, int snap_ratio
) 
{

    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if ((t < nt - 1) && (t > 1))
    {

        int irec = blockIdx.x * blockDim.x + threadIdx.x;

        if (irec < nrec)
        {
            int rx = rec[irec] + nb
            int rz = recz[irec] + nb
            depre[rz, rx] += seismogram[t, irec] / dh2
        }
        
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

            depas[idx] = arg[idx] * laplacian[idx] + 2.0 * depre[idx] - defut[idx];

            float damp = damp_x[l] * damp_z[k];

            defut[idx] = depre[idx] * damp;
            depre[idx] = depas[idx] * damp;
        }

        int snap_idx = (t - tstop) / snap_ratio;

        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.y + threadIdx.y;

        if ((i > 3 && i < nzz - 4) && (j > 3 && j < nxx - 4))
        {
            float* src = snaps[snap_idx * nzz * nxx + i * nxx + j];  
            float* rec = depre[i * nxx + j];

            num[i * nxx + j] += src[i * nxx + j] * rec[i * nxx + j]
        }

    }
}

''', 'backward_kernel')

_image_kernel = cp.rawKernel(r'''
extern "C" __global__
void image_kernel(float* image, float dt, float* numerator)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.y + threadIdx.y;

    if ((i > 3 && i < nzz - 4) && (j > 3 && j < nxx - 4))
    {
        image[i * nxx + j] += dt * numerator[i * nxx + j];
    }
}
''', 'image_kernel')
