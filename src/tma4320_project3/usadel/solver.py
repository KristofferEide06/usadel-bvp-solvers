import numpy as np
import numpy.typing as npt
from typing import cast
from collections.abc import Callable
from scipy.integrate import solve_bvp

from .transform import (
    complex_to_real,
    real_to_complex,
    reshape_vec,
    expand_vec,
    usadel_matrix_to_vec,
    vec_to_usadel_matrix,
)

from .riccati import (
  N_fun,
  Riccati_superconductor  
)

def vec_deriv(
    vec: npt.NDArray[np.float64],
    epsilon: float,
    delta: float
    ) -> npt.NDArray[np.float64]:
    """Calcualtes derivative of flattened vector

    Args:
        vec (npt.NDArray[np.float64]): Flattened vector
        epsilon (np.float): Quasiparticle excitation energy
        delta (np.float): Imaginary energy shift

    Returns:
        npt.NDArray[np.float64]: derivative of flattened vector
    """
    gamma, gamma_tilde, w, w_tilde = tuple(vec_to_usadel_matrix(vec))
    
    N, N_tilde = N_fun(gamma, gamma_tilde)    
    
    d_gamma = w
    d_gamma_tilde = w_tilde
    d_w = -2j * (epsilon + 1j * delta) * gamma - 2 * w @ N_tilde @ gamma_tilde @ w
    d_w_tilde = -2j * (epsilon + 1j * delta) * gamma_tilde - 2 * w_tilde @ N @ gamma @ w_tilde

    return usadel_matrix_to_vec(
        d_gamma,
        d_gamma_tilde,
        d_w,
        d_w_tilde
    )

def make_diff_system(
    epsilon: float,
    delta: float,
    ) -> Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    npt.NDArray[np.float64]
]:
    """Crates diff system function for bvp solver, dependent on necessary physical parameters

    Args:
        epsilon (np.float): Quasiparticle excitation energy
        delta (np.float): Imaginary energy shift

    Returns:
        Callable[ [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64] ]: diff_system function
    """
    def diff_system(
        x: npt.NDArray[np.float64], 
        vec: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
        result = np.zeros_like(vec)
        
        for i in range(len(x)):
            result[:, i] = vec_deriv(vec[:,i], epsilon , delta)
        return result
    
    return diff_system

def make_bc(
    epsilon: float,
    delta: float,
    zeta: float,
    l: float,
    phi_L: float,
    phi_R: float,
    superconductor: bool
    ) -> Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    npt.NDArray[np.float64]
]:
    """Creates boundary condition function for bvp_solver

    Args:
        epsilon (np.float): Quasiparticle excitation energy
        delta (np.float): Imaginary energy shift
        zeta (np.float): Interface parameter
        l (np.float): Length of normal region
        phi_L (float): Left superconducting phase.
        phi_R (float): Right superconducting phase.
        superconductor (bool): True if normal metal is interfaced with two superconductors

    Returns:
        Callable[ [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64] ]: bc function
    """
    def bc(
        v_left: npt.NDArray[np.float64], 
        v_right: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        l_gamma, l_gamma_tilde, l_w, l_w_tilde = vec_to_usadel_matrix(v_left)
        r_gamma, r_gamma_tilde, r_w, r_w_tilde = vec_to_usadel_matrix(v_right)
        
        if superconductor:
            gamma_L, gamma_tilde_L, gamma_R, gamma_tilde_R = tuple(Riccati_superconductor(epsilon, delta, phi_L, phi_R))
        else:
            gamma_L, gamma_tilde_L, gamma_R, gamma_tilde_R = (
                np.zeros_like(l_gamma, dtype = np.complex128), 
                np.zeros_like(l_gamma_tilde, dtype = np.complex128), 
                np.zeros_like(r_gamma, dtype = np.complex128), 
                np.zeros_like(r_gamma_tilde, dtype = np.complex128)
                )

        N_L, N_tilde_L = N_fun(gamma_L, gamma_tilde_L)
        N_R, N_tilde_R = N_fun(gamma_R, gamma_tilde_R)
        I = np.eye(l_gamma.shape[0])
        
        l_w_boundary = l_w + 1/(zeta * l) * (I - l_gamma @ gamma_tilde_L) @ N_L @ (gamma_L - l_gamma)
        l_w_tilde_boundary = l_w_tilde + 1/(zeta * l) * (I - l_gamma_tilde @ gamma_L) @ N_tilde_L @ (gamma_tilde_L - l_gamma_tilde)
        r_w_boundary = r_w - 1/(zeta * l) * (I - r_gamma @ gamma_tilde_R) @ N_R @ (gamma_R - r_gamma)
        r_w_tilde_boundary = r_w_tilde - 1/(zeta * l) * (I - l_gamma_tilde @ gamma_R) @ N_tilde_R @ (gamma_tilde_R - l_gamma_tilde)
        
        return usadel_matrix_to_vec(l_w_boundary, l_w_tilde_boundary, r_w_boundary, r_w_tilde_boundary)
    
    return bc

def usadel_solver(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    zeta: float,
    l: float,
    phi_L: float,
    phi_R: float,
    superconductor: bool
    ) -> tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128]
    ]:
    """Calcualtes the Riccati parameters for given x and y

    Args:
        x (np.NDArray[np.float64]): Array to find solution on
        y (np.NDARray[np.float64]): Initial guess
        epsilon (np.float): Quasiparticle excitation energy
        delta (np.float): Imaginary energy shift
        zeta (np.float): Interface parameter
        l (np.float): Length of normal region
        phi_L (float): Left superconducting phase.
        phi_R (float): Right superconducting phase.
        superconductor (bool): True if normal metal is interfaced with two superconductors

    Returns:
        tuple[npt.NDArray[
            np.complex128],
            npt.NDArray[np.complex128],
            npt.NDArray[np.complex128],
            npt.NDArray[np.complex128]
            ]: Arrays of values for gamma, gamma_tilde, w, w_tilde arrays at all x positions
    """
    solution = solve_bvp(
        make_diff_system(epsilon, delta),
        make_bc(epsilon, delta, zeta, l, phi_L, phi_R, superconductor),
        x, y
    )  
    
    sol = solution.sol(x)
    m = sol.shape[1]
    
    gamma_arr, gamma_tilde_arr, w_arr, w_tilde_arr = tuple([np.empty((m, 2, 2), dtype = np.complex128) for i in range(4)])
    
    for j in range(m):
        gamma_arr[j], gamma_tilde_arr[j], w_arr[j], w_tilde_arr[j] = vec_to_usadel_matrix(sol[:, j])

    
    return gamma_arr, gamma_tilde_arr, w_arr, w_tilde_arr
  
  