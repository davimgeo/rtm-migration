from pathlib import Path

PATH = "data/output/"

from src import Plotting

plt = Plotting()

model1 = plt.load(
    PATH + "image_219snaps_200x100.bin", height=100, weight=200
)
model2 = plt.load(
    PATH + "image_den_219snaps_200x100.bin", height=100, weight=200
)

plt.compare(
    model1, model2, 
    "Normal Correlation", 
    "Correlation with Ilumination"
)
