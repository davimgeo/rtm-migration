from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

from ..utils import measure_runtime

if TYPE_CHECKING:
  from . import Config, Model, Seismogram, Wavelet, Geometry

import cupy as cp

class Propagation:

  def __init__(
    self, config: Config, 
    model: Model, 
    geometry: Geometry,
    seismogram: Seismogram, 
    wavelet: Wavelet
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

    self.kernel_arg = KernelArguments.make(
        config.dh, config.dt, model.model
      )
    self.kernel_arg_homo = KernelArguments.make(
        config.dh, config.dt, model_homo
    )

    self.damp_x = cp.zeros(model.nxx, dtype=cp.float32)
    self.damp_z = cp.zeros(model.nzz, dtype=cp.float32)

    self.block = (16, 16)
    self.grid = (
      (model.nxx + self.block[0] - 1) // self.block[0],
      (model.nzz + self.block[1] - 1) // self.block[1]
    )

    self.ix, self.iz = 0, 0

  @measure_runtime
  def remove_direct_wave_model(self, ix: int, iz: int) -> None:
    self.ix, self.iz = ix, iz

    for t in range(1, self.c.nt - 1):

      self.zero_out_matrices()

      self.forward_propagation(
         self.u,
         self.laplacian,
         self.kernel_arg,
         t
      ) 

      _get_seismogram(
        self.s.seismogram,
        self.u.present,
        self.g.nrec
      )

      if not t % 400:
        import matplotlib.pyplot as plt
        plt.imshow(cp.asnumpy(self.u.present))
        plt.show()

      self.forward_propagation(
         self.u_homo,
         self.laplacian_homo,
         self.kernel_arg_homo,
         t
      ) 

      _get_seismogram(
        self.s.seismogram_homo,
        self.u_homo.present,
        self.g.nrec
      )

    self.s.seismogram -= self.s.seismogram_homo

  def zero_out_matrices(self):
    self.s.seismogram.fill(0.0)
    self.s.seismogram_homo.fill(0.0)

    self.u.past.fill(0.0)
    self.u.present.fill(0.0)
    self.u.future.fill(0.0)

    self.u_homo.past.fill(0.0)
    self.u_homo.present.fill(0.0)
    self.u_homo.future.fill(0.0)

  def forward_propagation(
    self,
    u_field,
    laplacian_field,
    kernel_arg,
    t
  ):

    _forward_kernel(
        self.grid, self.block, (
        u_field.past,
        u_field.present,
        u_field.future,
        laplacian_field,
        self.damp_x,
        self.damp_z,
        kernel_arg.inv_dh2,
        self.m.nzz,
        self.m.nxx,
        self.w.wavelet,
        self.ix,
        self.iz,
        kernel_arg.dh2,
        kernel_arg.velocity_term,
        self.c.nt,
        t
      )
    )
    
    u_field.future = u_field.present
    u_field.present = u_field.past

  def get_damp(self):
    for i in range(self.m.nzz):

      if self.c.nb <= i < self.c.nb + self.c.nz:
          self.damp_z[i] = 1.0

      elif i < self.c.nb:
          d = self.c.nb - i
          self.damp_z[i] = cp.exp(-(self.c.factor * d) * (self.c.factor * d))

      else:
          d = i - (self.c.nb + self.c.nz - 1)
          self.damp_z[i] = cp.exp(-(self.c.factor * d) * (self.c.factor * d))

    for j in range(self.m.nxx):

      if self.c.nb <= j < self.c.nb + self.c.nx:
          self.damp_x[j] = 1.0

      elif j < self.c.nb:
          d = self.c.nb - j
          self.damp_x[j] = cp.exp(-(self.c.factor * d) * (self.c.factor * d))

      else:
          d = j - (self.c.nb + self.c.nx - 1)
          self.damp_x[j] = cp.exp(-(self.c.factor * d) * (self.c.factor * d))

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
  dh2: float
  inv_dh2: float
  velocity_term: cp.ndarray

  @classmethod
  def make(cls, dh, dt, model):
    dh2 = dh ** 2
    return cls(
      dh2=dh2,
      inv_dh2=1.0 / (5040.0 * dh2),
      velocity_term=dt**2 * model**2,
    )


_forward_kernel = cp.RawKernel(r'''
extern "C" __global__
void forward_kernel(
    float* upas, float* upre, float* ufut,
    float* laplacian, float* damp_x, float* damp_z,
    float inv_dh2, int nzz, int nxx, float* ricker,
    int ix, int iz, float dh2, float* arg,
    int nt, int t
)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == iz && j == ix) {
      upre[iz * nxx + ix] += ricker[t] / dh2;
    }

    if (i >= 4 && i < nzz - 4 && j >= 4 && j < nxx - 4) 
    {
        float d2u_dx2 =
          -9.0   * upre[(i-4) * nxx + j] +
          128.0  * upre[(i-3) * nxx + j] -
          1008.0 * upre[(i-2) * nxx + j] +
          8064.0 * upre[(i-1) * nxx + j] -
          14350.0* upre[i * nxx + j] +
          8064.0 * upre[(i+1) * nxx + j] -
          1008.0 * upre[(i+2) * nxx + j] +
          128.0  * upre[(i+3) * nxx + j] -
          9.0    * upre[(i+4) * nxx + j];

        float d2u_dz2 =
            -9.0   * upre[i * nxx + (j-4)] +
            128.0  * upre[i * nxx + (j-3)] -
            1008.0 * upre[i * nxx + (j-2)] +
            8064.0 * upre[i * nxx + (j-1)] -
            14350.0* upre[i * nxx + j] +
            8064.0 * upre[i * nxx + (j+1)] -
            1008.0 * upre[i * nxx + (j+2)] +
            128.0  * upre[i * nxx + (j+3)] -
            9.0    * upre[i * nxx + (j+4)];

        laplacian = (d2u_dx2 + d2u_dz2) * inv_dh2;

        float damp = damp_x[l] * damp_z[k];

        upas[i * nxx + j] = (
          arg[i * nxx + j] * laplacian 
          + 2.0  * upre[i * nxx + j] 
          - ufut[i * nxx + j]
        ) * damp;

      }
}
''', 'forward_kernel')

_get_seismogram = cp.RawKernel(r'''
extern "C" __global__
void get_seismogram(float* seismogram, float* upre, int nrec)
{
  int irec = blockIdx.x * blockDim.x + threadIdx.x;

  int rx = (int)recx[irec];
  int rz = (int)recz[irec];

  seismogram[t * nrec + irec] = upre[rz * nxx + rx];
}

''', 'get_seismogram')
