import numpy as np
from scipy.signal import convolve2d

def solve(initial, fixed, tol=1e-15):
    """Simulate heat diffusion to identify the steady state.

    Parameters
    ----------
    initial: float array
      the initial temperate at every point in the grid
    fixed: bool array
      elements that are set to True will be kept fixed and not allowed to change
    tol: float
      iteration continues until no element changes by more than this
    """
    mask = np.array([[0, 0.25, 0], [0.25, 0.0, 0.25], [0, 0.25, 0]])
    fixed_values = initial[fixed]
    array = initial
    while True:
        new_array = convolve2d(array, mask, mode='same', boundary='symm')
        new_array[fixed] = fixed_values
        change = np.max(np.abs(array-new_array))
        if change <= tol:
            return array
        array = new_array
    return array