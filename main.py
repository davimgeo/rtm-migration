from src import *

PATH = "config/parameters.toml"

@measure_runtime
def main():
  cfg = Config(PATH).load()

  model = Model(cfg)
  model.load()
  model.set_boundary()

  geom = Geometry(cfg)
  geom.get()

  seis = Seismogram(geom, cfg)

  mig = Migration(model, geom, seis, cfg)
  mig.get_ricker()
  mig.set_damper()
  mig.rtm()
  mig.laplacian_filter()

  import matplotlib.pyplot as plt
  import numpy as np
  xloc = np.linspace(0, cfg.nx - 1, 11, dtype=int)
  xlab = np.array(xloc * cfg.dh, dtype=int)

  zloc = np.linspace(0, cfg.nz - 1, 7, dtype=int)
  zlab = np.array(zloc * cfg.dh, dtype=int)

  fig, ax = plt.subplots(1, 1, figsize=(12, 5))
  
  grad = mig.gradient[
      cfg.nb:cfg.nb + cfg.nz,
      cfg.nb:cfg.nb + cfg.nx
  ]

  vmax = np.percentile(np.abs(grad), 99)
  vmin = -vmax

  img2 = ax.imshow(
      grad,
      aspect="auto",
      cmap="Greys",
      vmin=vmin,
      vmax=vmax
  )

  ax.set_xticks(xloc)
  ax.set_xticklabels(xlab)
  ax.set_yticks(zloc)
  ax.set_yticklabels(zlab)

  ax.set_xlabel("Distance [m]")
  ax.set_ylabel("Depth [m]")
  ax.set_title("Gradient")

  plt.show()
  return mig, seis

if __name__ == "__main__":
  mig, seis = main()

  #mig.plot_snapshots()
  mig.plot_model_and_geometry()
  mig.plot_image()



