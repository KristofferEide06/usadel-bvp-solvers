import numpy as np
from tma4320_project3.usadel.plot import(
    plot_usadel_sol,
    plot_observable,
)

def main() -> None:
    l = 1
    m = 101
    zeta = 3
    delta = 0.01
    
    x = np.linspace(0, l, m)
    y = np.zeros((32, m))
    
    eps_arr_2g = np.array([0, 1, 2])
    
    for epsilon in eps_arr_2g:
        plot_usadel_sol(
            x,
            y,
            epsilon,
            delta,
            zeta,
            l,
            phi_L = 0,
            phi_R = 0,
            superconductor = False,
            matrix_label = 'gamma',
            filename = f'gamma_component_eps_{epsilon}',
        )
        
if __name__ == '__main_':
    main()