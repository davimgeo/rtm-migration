from __future__ import annotations

import sys
import os

os.chdir(os.path.join(os.path.dirname(sys.argv[0]), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======================================================== #

import matplotlib.pyplot as plt
import numpy as np

from src import *

PATH = "config/parameters.toml"

config = Config(PATH)
config.load()

geom = Geometry(config)
geom.get()

model = Model(config, geom)
model.get()
model.set_boundary()
model.gaussian_smooth(sigma=3)
model.plot()

wavelet = Wavelet(config)
wavelet.get()

seis = Seismogram(config, geom)

modeling = Modeling(config, model, geom, seis, wavelet)
modeling.get_damp()

ix = int(1200 / 10) + config.nb
iz = int(8 / 10) + config.nb

modeling.fdm_propagation(ix, iz, isSnap=True)

modeling.plot_snapshots()
seis.plot()
