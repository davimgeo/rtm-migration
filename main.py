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

  modeling = Modeling(
    config, model, geom, seis, wavelet
  )
  modeling.get_damp()

  migration = Migration(
    model, geom, seis, 
    config, wavelet, modeling
  )
  migration.rtm()  
  migration.laplacian_filter()

  return migration, model

if __name__ == "__main__":
  migration, model = main()

  model.plot_model_and_geometry(model.model_smooth)
  #migration.plot_snapshots(migration.snapshots_src)
  #migration.plot_snapshots(migration.snapshots_rec)
  migration.plot(gradient=True)



