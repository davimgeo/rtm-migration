import matplotlib.pyplot as plt
import numpy as np

def create_circle(model, r: int, value, center: tuple):
  nx, nz = model.shape

  mask = np.zeros((nx, nz))

  for i in range(nx):
    for j in range(nz):
      mask[i, j] = value * (
        (i - center[0])**2 + (j - center[1])**2 <= r**2
      )

  return model * mask
   
def gaussian_filter_polar(
  arr: np.ndarray, 
  sigma: float, 
  truncate: float,
) -> np.ndarray:
      
  radius = int(truncate * sigma + 0.5)

  theta = np.arange(-radius, radius + 1)
  gaussian = np.exp(-0.5 * (theta**2) / (sigma**2))

  gaussian /= gaussian.sum()

  result = arr.copy()

  for col in range(arr.shape[1]):
    _, r_data = cartesian2polar(arr[:, col])

    padded = np.pad(r_data, radius, mode="reflect")

    smoothed = np.convolve(padded, gaussian, mode="same")

    result[:, col] = smoothed[radius:-radius]
      
  return result

def cartesian2polar(arr, plot=0):
  x = np.arange(len(arr))
  y = arr

  r = np.sqrt(x**2 + y**2)
  theta = np.arctan2(y, x)


  if plot:
    fig, ax = plt.subplots(
          subplot_kw={'projection': 'polar'},
          figsize=(5, 8),
          layout='constrained'
    )
    ax.plot(theta, r)

    plt.show()

  return theta, r

nz, nx = 300, 900

model = np.full((nz, nx), 1500.0)
model = create_circle(
  model, r=75, value=2000.0, center=(nz//2, nx//2)
)
#model = gaussian_filter_polar(model, sigma=10, truncate=4)

plt.imshow(model)
plt.show()




