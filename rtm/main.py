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
  #mig.laplacian_filter()

  return mig, seis

if __name__ == "__main__":
  mig, seis = main()

  #mig.plot_snapshots()
  mig.plot_model_and_geometry()
  mig.plot_image()



