import numpy as np
from tma4320_project3.usadel.plot import(
    plot_usadel_sol,
    plot_observable,
)

l = 1
m = 101
zeta = 3
delta = 0.01
    
x = np.linspace(0, l, m)
y = np.zeros((32, m))

def t2_g(x, y, l, zeta, delta) -> None:
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
            filename = f'2g/gamma_component_eps_{epsilon}',
        )
    
def t2_h(x, y, l, zeta, delta) -> None:
    eps_arr_2h = np.array([0, 1, 2])
    
     
    for epsilon in eps_arr_2h:
        plot_observable(
            observable = 'dos',
            variable_plot = 'x',
            x = x,
            x_index = -1,
            y = y,
            epsilon = epsilon,
            delta = delta,
            zeta = zeta,
            l = l,
            phi_L = 0,
            phi_R = 0,
            superconductor = False,
            filename = f'2h/dos_eps_{epsilon}'
        )

def t2_i(x, y, l, zeta, delta) -> None:
    eps_arr_2i = np.array([0, 1, 2])
    
    for epsilon in eps_arr_2i:
        plot_usadel_sol(
            x,
            y,
            epsilon,
            delta,
            zeta,
            l,
            phi_L = 0,
            phi_R = 0,
            superconductor = True,
            matrix_label = 'gamma',
            filename = f'2i/gamma_component_eps_{epsilon}',
        )

def t2_j(x, y, l, zeta, delta) -> None:
    eps_arr_2j = np.array([0, 1, 2])
    
     
    for epsilon in eps_arr_2j:
        plot_observable(
            observable = 'dos',
            variable_plot = 'x',
            x = x,
            x_index = -1,
            y = y,
            epsilon = epsilon,
            delta = delta,
            zeta = zeta,
            l = l,
            phi_L = 0,
            phi_R = 0,
            superconductor = True,
            filename = f'2j/dos_eps_{epsilon}'
        )
        
def t2_k(x, y, l,m, zeta, delta) -> None:
    l_arr_2k = np.array([0.5, 1, 2])
    eps_arr_2k = np.linspace(0, 2, 101)

    
    for l in l_arr_2k:
        x = np.linspace(0, l, m)
        x_index = len(x)//2
        
        plot_observable(
            observable = 'dos',
            variable_plot = 'epsilon',
            x = x,
            x_index = x_index,
            y = y,
            epsilon = eps_arr_2k,
            delta = delta,
            zeta = zeta,
            l = l,
            phi_L = 0,
            phi_R = 0,
            superconductor = True,
            filename = f'2k/dos_eps_l_{l}.png'
        )

def t2_l(x, y, l, zeta, delta) -> None:
    epsilon_arr_2l = np.array([2, 1.5, 1.0, 0.5, 0.0])
    
    for epsilon in epsilon_arr_2l:
        plot_observable(
            observable = 'current_integrand',
            variable_plot = 'x',
            x = x,
            x_index = -1,
            y = y,
            epsilon = epsilon,
            delta = delta,
            zeta = zeta,
            l = l,
            phi_L = 0,
            phi_R = 0,
            superconductor = True,
            filename = f'2l/current_int_eps_{epsilon}.png'
        )

def t2_m(x, y, l, zeta, delta) -> None:
    epsilon_arr = np.linspace(2, 0, 101)
    x_index = len(x)//2
    
    plot_observable(
            observable = 'current_integrand',
            variable_plot = 'epsilon',
            x = x,
            x_index = x_index,
            y = y,
            epsilon = epsilon_arr,
            delta = delta,
            zeta = zeta,
            l = l,
            phi_L = 1,
            phi_R = 0,
            superconductor = True,
            filename = f'2m/current_int_eps.png'
        )
    
def t2_n(x, y, l, zeta, delta) -> None:
    phi_L_arr_2n = np.linspace(0, 2 * np.pi, 15) #update to contain more pointsa after optimizing runtime
    x_index = len(x)//2
    
    plot_observable(
        observable = 'current',
        variable_plot = 'phi_L',
        x = x,
        x_index = x_index,
        y = y,
        epsilon = -1,
        delta = delta,
        zeta = zeta,
        l = l,
        phi_L = phi_L_arr_2n,
        phi_R = 0,
        superconductor = True,
        filename = f'2n/current_phi_L.png'
    )

def main(x, y, l, m, zeta, delta) -> None:
    t2_g(x, y, l, zeta, delta)
    t2_h(x, y, l, zeta, delta)
    t2_i(x, y, l, zeta, delta)
    t2_j(x, y, l, zeta, delta)
    t2_k(x, y, l,m, zeta, delta)
    t2_l(x, y, l, zeta, delta)
    t2_m(x, y, l, zeta, delta)
    t2_n(x, y, l, zeta, delta)
    
if __name__ == '__main__':
    main(x, y, l, m, zeta, delta)