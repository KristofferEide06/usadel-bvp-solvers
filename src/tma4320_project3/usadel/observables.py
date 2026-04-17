import numpy as np
import numpy.typing as npt

from .riccati import(
    rho_3_fun,
    green_fun,
    green_fun_deriv,
)

from .einsum import(
    tr
)

def dos_fun(
    gamma: npt.NDArray[np.complex128],
    gamma_tilde: npt.NDArray[np.complex128],
) -> np.float64 | npt.NDArray[np.float64]:
    """Calculates density of states for Riccati amplitudes

    Args:
        gamma (npt.NDArray[np.complex128]): Riccati amplitude
        gamma_tilde (npt.NDArray[np.complex128]): conjugate Riccati amplitude

    Returns:
        np.float64: density of states
    """
    rho_3 = rho_3_fun()
    greens = green_fun(gamma, gamma_tilde)

    if greens.ndim == 2:
        return np.real(tr(rho_3 @ greens)) / 4

    return np.real(np.einsum('iik->k', np.einsum('ij,jkl->ikl', rho_3, greens, optimize=True), optimize=True)) / 4

def current_integrand_fun(
    gamma: npt.NDArray[np.complex128],
    gamma_tilde: npt.NDArray[np.complex128],
    w: npt.NDArray[np.complex128],
    w_tilde: npt.NDArray[np.complex128],
) -> np.float64 | npt.NDArray[np.float64]:
    """Calculates the current integrand

    Args:
        gamma (npt.NDArray[np.complex128]): Riccati amplitude
        gamma_tilde (npt.NDArray[np.complex128]): conjugate Riccati amplitude
        w (npt.NDArray[np.complex128]): d_x gamma
        w_tilde (npt.NDArray[np.complex128]): d_x gamma_tilde

    Returns:
        np.float64: current integrand
    """
    
    rho_3 = rho_3_fun()
    greens = green_fun(gamma, gamma_tilde)
    d_greens = green_fun_deriv(gamma, gamma_tilde, w, w_tilde)

    if greens.ndim == 2:
        return np.real(tr(rho_3 @ (greens @ d_greens - d_greens @ greens)))

    commutator = np.einsum('ikn,kjn->ijn', greens, d_greens, optimize=True) - np.einsum('ikn,kjn->ijn', d_greens, greens, optimize=True)
    return np.real(np.einsum('iik->k', np.einsum('ij,jkn->ikn', rho_3, commutator, optimize=True), optimize=True))
