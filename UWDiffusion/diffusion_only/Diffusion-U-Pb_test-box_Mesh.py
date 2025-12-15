#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: uw
#     language: python
#     name: python3
# ---

# %%
import os
import time


# %%
os.environ["UW_TIMING_ENABLE"] = "1"
from underworld3 import timing


# %%
import underworld3 as uw
import UWDiffusion as DIF


# %%
import numpy as np
import sympy as sp
from scipy.special import erf
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt


# %%
timing.reset()
timing.start()
time_start = time.time()


# %%
U_degree    = uw.options.getInt("U_degree", default=2)
continuous  = uw.options.getBool("continuous", default=True)

csize       = uw.options.getReal("csize", default=0.01)

tolerance   = uw.options.getReal("tolerance", default=1e-6)


save_figs    = uw.options.getBool("save_figs", default=True)

CFL_fac      = uw.options.getReal("CFL_fac", default=1.0)

# dt_max = uw.options.getInt("dt_max", default=1) ### Myr

if uw.mpi.rank == 0:
    print(f'csize = {csize}, CFL = {CFL_fac}, degree = {U_degree},')


# %%
T_start = uw.options.getReal("Temp_start", default=800) ### C
T_end   = uw.options.getReal("Temp_end", default=800) ### C
t_start = uw.options.getReal("time_start", default=0) ### Myr
t_end   = uw.options.getReal("time_end", default=500) ### Myr,


x0_c = uw.options.getReal("x0", default=0.35) ### start of heat pipe
x1_c = uw.options.getReal("x1", default=0.65) ### end of heat pipe

if uw.mpi.rank == 0:
    print(f'T_start = {T_start}, T_end = {T_end}, t_start = {t_start}, t_end = {t_end}')

duration = (t_end - t_start)

gradient = (T_start - T_end) / ( t_end - t_start)

if gradient == 0:
    outputPath  = uw.options.getString('outputPath', default=f'./output/Diffusion_U-Pb_test-boxMesh-T={T_start}C-t={t_end}Myr-degree={U_degree}-csize={csize}-CFL={CFL_fac}/')
else:
    outputPath  = uw.options.getString('outputPath', default=f'./output/Diffusion_U-Pb_test-boxMesh-Tstart={T_start}C-gradient={gradient}CMyr-t={t_end}Myr-degree={U_degree}-csize={csize}-CFL={CFL_fac}/')

if uw.mpi.rank == 0:
    os.makedirs(outputPath, exist_ok=True)




# %%
time_profile = np.linspace(t_start, t_end, int(t_end) ) ### every 1 Myr
temp_profile = np.linspace(T_start, T_end, int(t_end) ) ### corresponding temp

# Create the interpolator
temperature_interp = interp1d(
    time_profile, 
    temp_profile,
    kind='linear', 
    fill_value='extrapolate' 
)



# %% [markdown]
#
# Table of variables to determine the diffusion coefficient for each element
# | Variable | Symbol            | units | U | Pb | 
# | :---------- | :-------: | :-------: | :------: |  ------: | 
# | Pre-exponent| $D_0$   | $\text{m}^2\, \text{s}^{-1}$ |  1.63   | 0.11 |
# | Activation energy | $E_a$  | $\text{kJ}\, \text{mol}^{-1}$ |  726 $\pm$ 83    |  550 $\pm$ 30  |
# | Gas constant | $R$  | $\text{J}\, \text{mol}^{-1}\, \text{K}^{-1}$ |  8.314    | 8.314 | 
# | Reference | |  | [Cherniak and Watson, 1997](http://link.springer.com/10.1007/s004100050287) | [Cherniak and Watson, 2001](https://www.sciencedirect.com/science/article/pii/S0009254100002333) | [Cherniak and Watson, 2007](https://www.sciencedirect.com/science/article/pii/S0009254107002148) | 

# %%
D, E, R, T = sp.symbols('D E R T') # Temperature in Kelvin

D_sym = D * sp.exp(-E / (R * (T+ 273.15) ) )


# %%
def make_diffusivity_fn(D_sym, D0, Ea):
    D_exp = D_sym.subs({D: D0, E: Ea, R: 8.314}).simplify()
    return sp.lambdify(T, D_exp, 'numpy')

D_Pb_fn = make_diffusivity_fn(D_sym, D0=0.11, Ea=550e3)
D_U_fn = make_diffusivity_fn(D_sym, D0=10**0.212, Ea=726e3)


# %%
# import unit registry
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 


diffusive_rate    = D_Pb_fn(T_start) * u.meter**2 /u.second
model_length      = 100 * u.micrometer ### scale the mesh radius to the zircon radius


KL = model_length
Kt = model_length**2 / diffusive_rate


scaling_coefficients  = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt

scaling_coefficients


# ### Create analytical fn

# %%
v, t, x, x0, x1, ka = sp.symbols('u, t, x_c, x0, x1, kappa')

U_a_x = 0.5 * (
    sp.erf((x1 - x + (v*t)) / (2 * sp.sqrt(ka*t))) +
    sp.erf((-x0 + x - (v*t)) / (2 * sp.sqrt(ka*t)))
)


x_arr = np.linspace(0, 1, 100)
y_arr = np.linspace(0, 1, 100)
X_grid, Y_grid = np.meshgrid(x_arr, y_arr)




# %%
if save_figs:
    import matplotlib.pyplot as plt
    import numpy as np
    
    x_arr = np.linspace(0, 1, 100)
    y_arr = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.zeros_like(X) 
    Z[(X>=x0_c) & (X<=x1_c)] = 1
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = plt.cm.viridis  # Use the viridis colormap
    norm = plt.Normalize(vmin=0, vmax=1) 
    
    # Plot the data
    contour = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap=cmap, norm=norm)
    
    # Add the colorbar
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical')  # or 'horizontal'

    cbar.set_ticks(np.linspace(0, 1, 11) )
    
    # Add labels and title
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    ax.set_aspect('equal')
    
    
    # Show the plot
    # plt.tight_layout()

    plt.savefig(f'{outputPath}Initial_condition.pdf', bbox_inches='tight')
    
    # plt.show()

# %%
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1), cellSize=csize, qdegree=U_degree)

# %%
init_dist = sp.Piecewise( (0, mesh.X[0] < x0_c), (1, (mesh.X[0] >= x0_c) & (mesh.X[0] <= x1_c)), (0, mesh.X[0] > x1_c) )

# %%
Pb_diffusion = DIF.DiffusionModel(variable_name='Pb', initial_abundance=init_dist, degree=U_degree, order=1, mesh=mesh)

U_diffusion = DIF.DiffusionModel(variable_name='U', initial_abundance=init_dist, degree=U_degree, order=1, mesh=mesh)


# %%
T_initial = temperature_interp(0)


# %%
Pb_diffusion.diffusivity = D_Pb_fn(T_initial) * u.meter**2 / u.second

# %%
U_diffusion.diffusivity = D_U_fn(T_initial) * u.meter**2 / u.second

# %%
for _solver in [Pb_diffusion, U_diffusion]:
    _solver.diffusion_solver.petsc_options["snes_rtol"]   = tolerance*1e-4
    _solver.diffusion_solver.petsc_options["snes_atol"]   = tolerance
    _solver.diffusion_solver.petsc_options["ksp_atol"]    = tolerance
    _solver.diffusion_solver.petsc_options["snes_max_it"] = 100
    _solver.diffusion_solver.petsc_options["snes_monitor_short"] = None


# %%
def update_kappa(Model, D_fn, temperature_interp):
    current_temp = temperature_interp(dim(Model.current_time, u.megayear).m) #T_start - gradient*dim(Model.current_time, u.megayear).m
    kappa = D_fn(current_temp)
    Model.diffusivity = kappa * u.meter**2 / u.second




# %%
def update_kappa_wrapper():
    update_kappa(Pb_diffusion, D_Pb_fn, temperature_interp)


Pb_diffusion.register_pre_solve_hook('update kappa routine', update_kappa_wrapper)


# %%
dTdt = np.diff(temp_profile) / np.diff(time_profile)
max_allowed_temp_change = 5 ### C
dt_temp_change = np.min(max_allowed_temp_change / np.maximum(np.abs(dTdt), 1e-12))


# %%
Pb_diffusion.run_simulation(duration=t_end*u.megayear, max_dt=dt_temp_change*u.megayear, time_step_factor=CFL_fac)


# %%
def update_kappa_wrapper():
    update_kappa(U_diffusion, D_U_fn, temperature_interp)


U_diffusion.register_pre_solve_hook('update kappa routine', update_kappa_wrapper)


# %%
U_diffusion.run_simulation(duration=t_end*u.megayear, max_dt=dt_temp_change*u.megayear, time_step_factor=CFL_fac)


# %%
def get_analytical_profile(nsteps, D_fn):

    time_steps = np.linspace(t_start, t_end, nsteps)

    time_steps_nd = nd(time_steps * u.megayear)
    temp_profile = T_start - gradient * time_steps
    kappa_profile = nd(D_fn(temp_profile) * u.meter**2 / u.second)


    # Cumulative diffusivity integral
    S = cumulative_trapezoid(kappa_profile, time_steps_nd, initial=0)
    vel = 0

    # Profile calculation at desired time index (e.g., last)
    idx = -1
    S_t = S[idx]
    t = time_steps_nd[idx]

    diffusion_profile = 0.5 * (
        erf((x1_c - sample_x + vel * t) / (2 * np.sqrt(S_t))) +
        erf((-x0_c + sample_x - vel * t) / (2 * np.sqrt(S_t)))
    )

    return diffusion_profile


# %%
sample_x = np.linspace(0, 1, 100)
sample_y = np.zeros_like(sample_x) + 0.5
sample_coords = np.column_stack([sample_x, sample_y])

initial_profile = np.ones_like(sample_x)
initial_profile[(sample_x < x0_c) | (sample_x > x1_c)] = 0.


# %%
if gradient == 0:
    Pb_subbed = U_a_x.subs({v:0, t:nd(t_end *u.megayear), x:mesh.X[0], x0 : x0_c, x1 : x1_c, ka:nd(D_Pb_fn(T_start) * u.meter**2 /u.second )})
    Pb_diffusion_profile = uw.function.evaluate(Pb_subbed, sample_coords)
    
    U_subbed = U_a_x.subs({v:0, t:nd(t_end *u.megayear), x:mesh.X[0], x0 : x0_c, x1 : x1_c, ka:nd(D_U_fn(T_start) * u.meter**2 /u.second )})
    U_diffusion_profile = uw.function.evaluate(U_subbed, sample_coords)
else:
    Pb_diffusion_profile = get_analytical_profile(1000, D_Pb_fn)
    Pb_interpolated_values = np.interp(Pb_diffusion.mesh_var.coords[:,0], sample_x, Pb_diffusion_profile)
    
    U_diffusion_profile = get_analytical_profile(1000, D_U_fn)
    U_interpolated_values = np.interp(U_diffusion.mesh_var.coords[:,0], sample_x, U_diffusion_profile)

# %%
UW_Pb_profile = uw.function.evaluate(Pb_diffusion.mesh_var.sym[0], sample_coords)
UW_U_profile = uw.function.evaluate(U_diffusion.mesh_var.sym[0], sample_coords)

# %%
if save_figs:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1, figsize=(5, 4))
    ax.plot(sample_x, U_diffusion_profile, c='orange', label='U analytical')
    ax.plot(sample_x, UW_U_profile, ls=':', c='k', label='U numerical')
    ax.plot(sample_x, Pb_diffusion_profile, c='darkgray', label='Pb analytical')
    ax.plot(sample_x, UW_Pb_profile, ls='-.', c='k', label='Pb numerical')
    # ax.plot(sample_x, FD_profile, ls=':')
    
    ax.legend(loc='best')

    plt.grid()

    plt.savefig(f'{outputPath}profile_comparison.pdf', bbox_inches="tight")
    
    plt.show()


# %%
if save_figs:
    fig, ax = plt.subplots(figsize=(5, 4))

    with Pb_diffusion.mesh.access(Pb_diffusion.mesh_var):
        if gradient ==0.:
            Pb_error = Pb_diffusion.mesh_var.data[:, 0] - uw.function.evaluate(Pb_subbed, Pb_diffusion.mesh_var.coords)
        else:
            Pb_error = Pb_diffusion.mesh_var.data[:, 0] - Pb_interpolated_values

    with U_diffusion.mesh.access(U_diffusion.mesh_var):
        if gradient == 0.:
            U_error = U_diffusion.mesh_var.data[:, 0] - uw.function.evaluate(U_subbed, U_diffusion.mesh_var.coords)
        
        else:
            U_error = U_diffusion.mesh_var.data[:, 0] - U_interpolated_values



    x_coord = Pb_diffusion.mesh_var.coords[:, 0]
    y_coord = Pb_diffusion.mesh_var.coords[:, 1]

    contour0  = ax.tricontourf(x_coord, y_coord, np.abs(Pb_error), levels=100, cmap='RdBu_r')


    plt.colorbar(contour0, ax=ax, format='%.0e', label='|error|')

    contour0.set_edgecolors("face")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    ax.set_aspect('equal')

    plt.savefig(f'{outputPath}Pb_error_across_domain.pdf', bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5, 4))

    contour1 = ax.tricontourf(x_coord, y_coord, np.abs(U_error), levels=100, cmap='RdBu_r')
    plt.colorbar(contour1, ax=ax, label='|error|', format='%.0e')

    # # Make a new axis for the colorbar (to the right of ax)
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.2)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('L$_{2}$-error')
#     plt.tight_layout()

    ### for removing the anti-aliasing effects

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    ax.set_aspect('equal')

    contour1.set_edgecolors("face")

    plt.savefig(f'{outputPath}U_error_across_domain.pdf', bbox_inches="tight")
    plt.show()


# %%
if gradient == 0:
    Pb_l2_norm = uw.maths.L2_norm(Pb_diffusion.mesh_var.sym[0], Pb_subbed, mesh)
    U_l2_norm = uw.maths.L2_norm(U_diffusion.mesh_var.sym[0], U_subbed, mesh)
else:
    analytical_sol = uw.discretisation.MeshVariable("a", mesh, 1, degree=U_degree, continuous=continuous)
    with mesh.access(analytical_sol):
        analytical_sol.data[:,0] = Pb_interpolated_values
        Pb_l2_norm = uw.maths.L2_norm(Pb_diffusion.mesh_var.sym[0], analytical_sol.sym[0], mesh)

        analytical_sol.data[:,0] = U_interpolated_values
        U_l2_norm = uw.maths.L2_norm(U_diffusion.mesh_var.sym[0], analytical_sol.sym[0], mesh)


# %%
time_end = time.time()

total_time = time_end - time_start


# %%
if uw.mpi.rank == 0:
    import pandas as pd

    # --- File 1: Data in columns (standard CSV) ---
    timing_data = timing.get_data()
    rows = []
    for (location, idx), (count, time) in timing_data.items():
        rows.append({
            'Location': location,
            'Index': idx,
            'Count': count,
            'Time': time,
            'Average': time/count,
        })
    df = pd.DataFrame(rows)
    df.sort_values(by='Average', inplace=True, ascending=False)
    df.to_csv(f'{outputPath}/Diffusion_results_columns.csv', index=False)

    # --- File 2: Data in rows (single row, best for summary) ---
    summary = {
        'U_degree': U_degree,
        'n_unknowns': Pb_diffusion.mesh_var.coords.shape[0],
        'MinRadius': mesh.get_min_radius(),
        'Pb_L2_norm': Pb_l2_norm,
        'U_L2_norm' : U_l2_norm,
        'CFL_fac' : CFL_fac,
        'TotalTime': total_time,
        'NumProcessors': uw.mpi.size,
    }
    # Save as a single row (header + values)
    with open(f'{outputPath}/Diffusion_results_rows.csv', 'w') as f:
        f.write(','.join(summary.keys()) + '\n')
        f.write(','.join(str(v) for v in summary.values()) + '\n')


# %%
