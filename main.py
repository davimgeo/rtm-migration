from src import *

PATH = "config/parameters.toml"

@measure_runtime
def main():
  config = Config(PATH).load()

  geom = Geometry(config)
  geom.get()

  model = Model(config, geom)
  model.get()
  model.set_boundary()
  model.gaussian_smooth(sigma=3)

  wavelet = Wavelet(config)
  wavelet.get_ricker()

  seis = Seismogram(geom, config)
  if config.load_residual:
    seis.load()

  modeling = Modeling(
    config, model, geom, seis, wavelet
  )
  modeling.set_damper()

  migration = Migration(
    model, geom, seis, 
    config, wavelet, modeling
  )
  migration.rtm()  
  migration.laplacian_filter()

  return migration, model

if __name__ == "__main__":
  migration, model = main()

  model.plot_model_and_geometry()
  migration.plot_snapshots(migration.snapshots_src)
  #migration.plot_snapshots(migration.snapshots_rec)
  migration.plot_image()



