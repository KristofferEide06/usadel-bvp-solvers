import numpy as np
import numpy.typing as npt
from numba import njit

@njit
def diff_system_1(x: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([y[1], -4.0 * np.sin(2.0 * x)])

@njit
def diff_system_2(x: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([y[1], y[0] + np.sin(x)])

def diff_system_2_plain(x:float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """defines the second system in the format solve_bvp expects.
    
    Args:
    x: the x value
    y: an array of y and y' values

    Returns:
    y_der: an array of the calculated y' and y'' values
    """
    return np.array([y[1], y[0] + np.sin(x)])

def bc_2(y_a: npt.NDArray[np.float64],y_b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """defines the boundary conditions for solve_bvp.
    
    Args:
    y_a: the solution vector at the left boundary
    y_b: the solution vector at the right boundary

    Returns:
    bc: an array describing the boundary residuals
    """
    return np.array([y_a[0],y_b[0]])