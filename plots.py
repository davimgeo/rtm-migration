from src import Plotting

PATH = "data/output/"

pltm = Plotting()

height, weight = 351, 1701

model_no_gradient = pltm.load(
    PATH + "marmousi_no_gradient_594snaps_1701x351.bin", height, weight
)
model_gradient = pltm.load(
    PATH + "marmousi_gradient_594snaps_1701x351.bin", height, weight
)

import numpy as np
mask1 = np.zeros_like(model_no_gradient, dtype=bool)
mask1[:55, :] = True

model_no_gradient[mask1] = 0.0
model_gradient[mask1] = 0.0

title2 = "Gradient"
pltm.plot(model_gradient)

trace_number = 550

#pltm.compare_model_and_traces(
#    model_no_gradient, model_gradient, trace_number,
#)
