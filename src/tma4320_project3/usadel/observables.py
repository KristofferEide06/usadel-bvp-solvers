import numpy as np
import numpy.typing as npt
from typing import cast
from collections.abc import Callable

from .riccati import(
    rho_3_fun,
    green_fun,
    green_fun_deriv
)

def dos_fun(
    gamma: npt.NDArray[np.complex128],
    gamma_tilde: npt.NDArray[np.complex128]
) -> np.float64:
    """Calculates density of states for Riccati amplitudes

    Args:
        gamma (npt.NDArray[np.complex128]): Riccati amplitude
        gamma_tilde (npt.NDArray[np.complex128]): Riccati amplitude conjugate

    Returns:
        np.float64: Density of states
    """
    rho_3 = rho_3_fun()
    greens = green_fun(gamma, gamma_tilde)
    
    return np.real(np.trace(rho_3 @ greens)) / 4

def current_integrand_fun(
    gamma: npt.NDArray[np.complex128],
    gamma_tilde: npt.NDArray[np.complex128],
    w: npt.NDArray[np.complex128],
    w_tilde: npt.NDArray[np.complex128]
) -> np.float64:
    """Calculates the current integrand

    Args:
        gamma (npt.NDArray[np.complex128]): Riccati amplitude
        gamma_tilde (npt.NDArray[np.complex128]): conjugate Riccati amplitude
        w (npt.NDArray[np.complex128]): d_x gamma
        w_tilde (npt.NDArray[np.complex128]): d_x gamma_tilde

    Returns:
        np.float64: Current integrand
    """
    
    rho_3 = rho_3_fun()
    g = green_fun(gamma, gamma_tilde)
    d_g = green_fun_deriv(gamma, gamma_tilde, w, w_tilde)
    
    return np.real(np.trace(rho_3 @ (g @ d_g - d_g @ g)))
