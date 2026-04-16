import numpy as np
import numpy.typing as npt
from typing import Literal
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from pathlib import Path

from .solver import(
    usadel_solver
)

from .observables import(
    dos_fun,
    current_integrand_fun
)

def plot_usadel_sol(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    zeta: float,
    l: float,
    phi_L: float,
    phi_R: float,
    superconductor: bool,
    matrix_label: Literal['gamma', 'gamma_tilde', 'w', 'w_tilde'],
    filename: str,
    figsize: tuple[int, ...] = (8,4),
):
    """Plots components of riccati matrix solution

    Args:
        x (npt.NDArray[np.float64]): Array to find solution on
        y (npt.NDARray[np.float64]): Initial guess
        epsilon (float): Quasiparticle excitation energy
        delta (float): Imaginary energy shift
        zeta (float): Interface parameter
        l (float): Length of normal region
        phi_L (float): Left superconducting phase.
        phi_R (float): Right superconducting phase.
        superconductor (bool): True if normal metal is interfaced with two superconductors
        matrix_label (str): Which matrix to plot, can be 'gamma', 'gamma_tilde', 'w', 'w_tilde'
        filename (str): name of file for plot to be saved to, will automatically be put in folder plots/usadel_sol/
        figsize (tuple[int, ...], optional): desired figure size. Defaults to (8,4).

    Raises:
        ValueError: If matrix_label is not amongst available matrices
    """
    
    if matrix_label not in ['gamma', 'gamma_tilde', 'w', 'w_tilde']:
        raise ValueError('argument matrix_label must be gamma, gamma_tilde, w or w_tilde')
        
    gamma, gamma_tilde, w, w_tilde = usadel_solver(
        x, y,
        epsilon, delta, zeta, l,
        phi_L, phi_R,
        superconductor
    )
    
    matrices = {
        "gamma": gamma,
        "gamma_tilde": gamma_tilde,
        "w": w,
        "w_tilde": w_tilde
    }
    
    matrix = matrices[matrix_label]
    matrix_shape = matrix[0].shape
    fig, ax = plt.subplots(*matrix_shape, figsize = figsize, squeeze = False)
    
    for row in range(matrix.shape[1]):
        for col in range(matrix.shape[2]):
            component = matrix[:, row, col]
            
            ax[row,col].plot(x, np.real(component), "r-", label = "Re")
            ax[row][col].plot(x, np.imag(component), "b--", label = "Im")
            ax[row][col].set_xlabel('x')
            ax[row][col].set_ylabel(rf"$\gamma_{{{row}{col}}}$")
            ax[row][col].grid()
            ax[row][col].legend()
        
    plt.tight_layout()
    
    #codex
    base_dir = Path(__file__).resolve().parents[3]
    save_path = base_dir / "plots" / "usadel" / "usadel_sol" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    #codex

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    

    plt.show()
    
def plot_observable(
    observable: Literal['dos', 'current', 'current_integrand'],
    variable_plot: Literal['x', 'epsilon', 'delta', 'zeta', 'l', 'phi_L', 'phi_R'],
    x: npt.NDArray[np.float64],
    x_index : int,
    y: npt.NDArray[np.float64],
    epsilon: float | npt.NDArray[np.float64],
    delta: float | npt.NDArray[np.float64],
    zeta: float | npt.NDArray[np.float64],
    l: float | npt.NDArray[np.float64],
    phi_L: float | npt.NDArray[np.float64],
    phi_R: float | npt.NDArray[np.float64],
    superconductor: bool,
    filename : str,
    figsize: tuple[int, ...] = (8,4),
):
    """Plots observable as function of variable_plot

    Args:
        observable (Literal['dos', 'current', 'current_integrand']): observable to plot
        variable_plot (Literal['x', 'epsilon', 'delta', 'zeta', 'l', 'phi_L', 'phi_R']): value to plot observable over
        x (npt.NDArray[np.float64]): array to find solution
        x_index (int): index of which to evaluate observable if variable_Plot != 'x'
        location of which to evaluate solution, can be any integer if variable_plot == 'x'
        y (npt.NDARray[np.float64]): initial guess guess
        epsilon (float): quasiparticle excitation energy
        delta (float:) imaginary energy shift
        zeta (float): interface parameter
        l (float): length of normal region
        phi_L (float): left superconducting phase.
        phi_R (float): right superconducting phase.
        superconductor (bool): True if normal metal is interfaced with two superconductors
        matrix_label (str): which matrix to plot, can be 'gamma', 'gamma_tilde', 'w', 'w_tilde'
        filename (str): name of file for plot to be saved to, will automatically be put in folder plots/usadel/observables/
        figsize (tuple[int, ...], optional): desired figure size. Defaults to (8,4).

    Raises:
        ValueError: if observable not amongst 'docs', 'current', 'current_integrand'
        ValueError: if variable_plot not amongst 'x', 'epsilon', 'delta', 'zeta', 'l', 'phi_L', 'phi_R'
        ValueError: if (variable_plot == 'epsilon') and observable == 'current'
    """
    if observable not in ['dos', 'current', 'current_integrand']:
        raise ValueError("observable must be 'dos', 'current', or 'current_integrand'")
    
    if variable_plot not in ['x', 'epsilon', 'delta', 'zeta', 'l', 'phi_L', 'phi_R']:
        raise ValueError("variable_plot must be 'x', 'epsilon', 'delta', 'zeta', 'l', 'phi_L' or 'phi_R'")
    
    if variable_plot == 'epsilon' and observable == 'current':
        raise ValueError("variable_plot == 'x' or 'epsilon' and observable == 'current' is meaningless combination")
    
    variables = {
        'x': x,
        'y': y,
        'epsilon': epsilon,
        'delta': delta,
        'zeta': zeta,
        'l': l,
        'phi_L': phi_L,
        'phi_R': phi_R,
        'superconductor': superconductor
    }
    
    dynamic_var = variables[variable_plot]
    fixed_vars = {k: v for k, v in variables.items() if k!= variable_plot}
    
    if variable_plot == 'x':
        gamma, gamma_tilde, w, w_tilde = usadel_solver(**variables)
        
        x_vals = x
        y_vals = np.zeros(len(x), dtype = np.float64)
        
        for i in range(len(x)):
            if observable == 'dos':
                y_vals[i] = dos_fun(gamma[i], gamma_tilde[i])
            elif observable == 'current_integrand':
                y_vals[i] = current_integrand_fun(gamma[i], gamma_tilde[i], w[i], w_tilde[i])
                
    else:
        dynamic_vals = np.asarray(dynamic_var, dtype = np.float64)
            
        x_vals = dynamic_var
        y_vals = np.zeros(len(x_vals), dtype = np.float64)
        
        for i, value in enumerate(dynamic_vals):
            current_vars = fixed_vars | {variable_plot: value}

            if observable == 'dos':
                gamma, gamma_tilde, w, w_tilde = usadel_solver(**current_vars)
                y_vals[i] = dos_fun(
                    gamma[x_index],
                    gamma_tilde[x_index],
                )
                
            elif observable == 'current_integrand':
                gamma, gamma_tilde, w, w_tilde = usadel_solver(**current_vars)
                y_vals[i] = current_integrand_fun(
                    gamma[x_index],
                    gamma_tilde[x_index],
                    w[x_index],
                    w_tilde[x_index]
                )    
            elif observable == 'current':
                epsilon_vals = np.linspace(0, 2,  100)
                integrand_vals = np.zeros_like(epsilon_vals)
                
                for j, epsilon_val in enumerate(epsilon_vals):
                    epsilon_vars = current_vars | {'epsilon': epsilon_val}
                    gamma, gamma_tilde, w, w_tilde = usadel_solver(**epsilon_vars)
                    
                    integrand_vals[j] = current_integrand_fun(
                        gamma[x_index],
                        gamma_tilde[x_index],
                        w[x_index],
                        w_tilde[x_index]
                    )
                    
                y_vals[i] = -simpson(integrand_vals, x = epsilon_vals)
                    
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(x_vals, y_vals)
    
    if variable_plot == 'x':
        ax.set_title(f"{observable} given {variable_plot}")
    else:
        ax.set_title(f"{observable} given {variable_plot} at x_index = {x_index}")
        
    ax.set_xlabel(variable_plot)
    ax.set_ylabel(observable)
    ax.grid()
    
     #codex
    base_dir = Path(__file__).resolve().parents[3]
    save_path = base_dir / "plots" / "usadel" / "observables" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    #codex
    
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()