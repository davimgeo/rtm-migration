from src import *

PATH = "config/parameters.toml"

@measure_runtime
def main():
  cfg = Config(PATH).load()

  model = Model(cfg)
  model.get()
  model.set_boundary()
  model.smooth()

  geom = Geometry(cfg)
  geom.get_geometry()

  seis = Seismogram(geom, cfg)

  acous = Acoustic(model, geom, seis, cfg)
  mig = Migration(acous)
  acous.get_ricker()
  acous.set_damper()
  acous.fd(mig)

  mig.plot_transit_time()

  return acous, seis

if __name__ == "__main__":
  acous, seis = main()

  #acous.plot_snapshots()
  acous.plot_model_and_geometry()

  seis.remove_direct_wave()
  seis.plot(seis.seismogram)
