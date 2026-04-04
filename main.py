from src import *

PATH = "config/parameters.toml"

@measure_runtime
def main():
  config = Config(PATH)
  config.load()

  geom = Geometry(config)
  geom.get()

  model = Model(config, geom)
  model.get()
  model.set_boundary()
  model.gaussian_smooth(sigma=3)

  wavelet = Wavelet(config)
  wavelet.get()
  wavelet.second_derivative()

  seis = Seismogram(config, geom)

  modeling = Modeling(config, model, geom, seis, wavelet)
  modeling.get_damp()

  # botar if else GPU na main

  migration = Migration(config, modeling, model, seis, wavelet, geom)
  migration.rtm()  

  return migration, model

if __name__ == "__main__":
  migration, model = main()

  model.plot(model.model_smooth)
  migration.plot()



