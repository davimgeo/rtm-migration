from src import *

PATH = "config/parameters.toml"

@measure_runtime
def main():
  cfg = Config(PATH).load()

  geom = Geometry(cfg)
  geom.get()

  model = Model(cfg, geom)
  model.get()
  #model.model_gaussian_smooth(sigma=3)
  model.set_boundary()

  seis = Seismogram(geom, cfg)
  if cfg.load_residual:
    seis.load_residual()

  mig = Migration(model, geom, seis, cfg)
  mig.get_ricker()
  mig.set_damper()
  mig.rtm()  
  #mig.laplacian_filter()

  return mig, model

if __name__ == "__main__":
  mig, model = main()

  #mig.plot_snapshots()
  model.plot_model_and_geometry()
  mig.plot_image()



