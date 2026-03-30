import sys
import os

os.chdir(os.path.join(os.path.dirname(sys.argv[0]), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======================================================== #

import matplotlib.pyplot as plt
import numpy as np

from src import *

PATH = "config/parameters.toml"

config = Config(PATH).load()

geom = Geometry(config)
geom.get()

model = Model(config, geom)
model.get()
model.set_boundary()
model.gaussian_smooth(sigma=10)
#model.plot_model_and_geometry(model.model_smooth)

wavelet = Wavelet(config)
wavelet.get_ricker()

seis = Seismogram(geom, config)

modeling = Modeling(
config, model, geom, seis, wavelet
)
modeling.get_damp()

for isrc in range(len(geom.srcxId)):
    ix = int(geom.srcxId[isrc]) + config.nb
    iz = int(geom.srczId[isrc]) + config.nb

    modeling.remove_direct_wave_model(ix, iz)
    #modeling.fdm_propagation(ix, iz, isSnap=False)

    #modeling.plot_snapshots()

    #seis.plot(seis.seismogram_homo)
    seis.plot()
    seis.save(ix, iz)