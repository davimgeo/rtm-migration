from src import *

PATH = "config/parameters.toml"

@measure_runtime
def main():
  cfg = Config(PATH).load()

  model = Model(cfg)
  model.get_model()
  model.set_boundary()

  geom = Geometry(cfg)
  geom.get_geometry()

  seis = Seismogram(geom, cfg)

  mig = Migration(model, geom, seis, cfg)
  mig.get_ricker()
  mig.set_damper()
  mig.fd()
  mig.fd_reverse()

  return mig, seis

if __name__ == "__main__":
  mig, seis = main()

  #mig.plot_snapshots()
  seis.plot(seis.seismogram)



