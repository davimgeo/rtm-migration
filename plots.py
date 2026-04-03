import sys
import os
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

PATH = Path("data/output")

os.chdir(PATH)

# ========================================================

from src import Plotting

plt = Plotting()

try:
    default = plt.load("image_219snaps_200x100.bin", height=100, weight=200)
    plt.plot(default)
except:
    pass

model1 = plt.load("20shots_2nd_deriv_200x100.bin", height=100, weight=200)
model2 = plt.load("20shots_no_2nd_deriv_200x100.bin", height=100, weight=200)

plt.compare(model1, model2)