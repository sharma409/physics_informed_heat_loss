import numpy as np
from scipy.ndimage import zoom
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
    width = initial.shape[0]
    if width > 32:
        # Start by solving a scaled down version of the grid, then use that
        # as the initial guess.  This can lead to much faster convergence.

        x, y = np.where(fixed)
        small_initial = zoom(initial, 0.25)
        small_initial[x//4, y//4] = fixed_values
        small_fixed = np.zeros(small_initial.shape, dtype=np.bool)
        small_fixed[x//4, y//4] = True
        small_solution = solve(small_initial, small_fixed, tol)
        array = zoom(small_solution, 4.0)
    else:
        # As the initial guess, just set the whole array to the average of all fixed elements.

        array = np.ones(initial.shape) * np.mean(fixed_values)
    array[fixed] = fixed_values

    # Iterate until convergence is reached.

    while True:
        new_array = convolve2d(array, mask, mode='same', boundary='symm')
        new_array[fixed] = fixed_values
        change = np.max(np.abs(array-new_array))
        if change <= tol:
            return array
        array = new_array
    return array
