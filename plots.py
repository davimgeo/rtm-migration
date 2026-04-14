from src import Plotting

PATH = "data/output/"

pltm = Plotting()

height, weight = 351, 1701

model1 = pltm.load(
    PATH + "marmousi_no_gradient_594snaps_1701x351.bin", height, weight
)
model2 = pltm.load(
    PATH + "marmousi_gradient_594snaps_1701x351.bin", height, weight
)

title1 = "No Gradient"
title2 = "Gradient"

trace_number = 50:801

pltm.compare_model_and_traces(
    model1, model2, trace_number,
    title1, title2
)
