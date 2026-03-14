import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

# ================================== // ================================

import matplotlib.pyplot as plt
import numpy as np

from src import *

PATH = "config/parameters.toml"
SEIS_PATH = "data\\output\\rec_domain\\seismogram_nt0.2_dt0.001_nrec40_5.bin"

cfg = Config(PATH).load()

geom = Geometry(cfg)
geom.get()

seis = Seismogram(geom, cfg)
seis.load(SEIS_PATH)
seis.plot(seis.seismogram)

SNAP_PATH = "data\\output\\src_domain\\snapshot_400x300_35.bin"

snap = np.fromfile(
    SNAP_PATH, dtype=np.float32, count=400*300
).reshape([300, 400], order='F')

fig, ax = plt.subplots()

scale = 2.0 * np.std(snap)

snap_frame = ax.imshow(
    snap[cfg.nb:cfg.nb+cfg.nz,
        cfg.nb:cfg.nb+cfg.nx],
    aspect="auto", cmap="Greys",
    vmin=-scale, vmax=scale, alpha=0.7
)

plt.show()
