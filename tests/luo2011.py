import matplotlib.pyplot as plt
import numpy as np

def create_circle(model, r, center, value):
  nz, nx = model.shape
  x0, y0 = center

  z, x = np.ogrid[:nz, :nx]
  dist = np.sqrt((z - x0)**2 + (x - y0)**2)
  mask_corona = dist <= r
  mask = dist <= r - 5

  model[mask_corona] = value - 500
  model[mask] = value

  return model

def smooth_circle(model, center, radius):
    nz, nx = model.shape
    x0, y0 = center

    z, x = np.ogrid[:nz, :nx]
    dist = np.sqrt((z - x0)**2 + (x - y0)**2)

    smooth = np.clip(1 - dist / radius, 0, 1)

    return smooth

nz, nx = 300, 900

center = (nz//2,nx//2)

r = 75

model = np.full((nz, nx), 2500.0)
model = create_circle(
  model, r, center, value=3500.0
)
model_smooth = smooth_circle(model, center, radius=r)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.imshow(model_smooth)

plt.show()



