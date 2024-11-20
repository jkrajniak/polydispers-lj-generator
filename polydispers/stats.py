import numpy as np
from scipy.special import gamma, gammainc


def sz_distribution_inverse_transform(Mn, PDI, size=1):
  """
  Generates molecular weights from the Schulz-Zimm distribution
  using inverse transform sampling with the Newton-Raphson method.

  Args:
    Mn: Number-average molecular weight.
    PDI: Polydispersity index.
    size: Number of molecular weights to generate.

  Returns:
    An array of molecular weights.
  """
  Mn_local = Mn / size
  z = 1 / (PDI - 1)
  u = np.random.uniform(0, 1, size=size)

  def sz_cdf(x):
    return gammainc(z + 1, (z + 1) * x / Mn_local)

  def sz_cdf_derivative(x):
    return ((z + 1) / Mn_local) * ((z + 1) * x / Mn_local)**z * np.exp(-(z + 1) * x / Mn_local) / gamma(z + 1)

  molecular_weights = np.zeros(size)
  for i in range(size):
    x = Mn_local  # Initial guess
    tolerance = 1e-6
    max_iterations = 100
    for _ in range(max_iterations):
      x_next = x - (sz_cdf(x) - u[i]) / sz_cdf_derivative(x)
      if abs(x_next - x) < tolerance:
        break
      x = x_next
    molecular_weights[i] = x

  return molecular_weights