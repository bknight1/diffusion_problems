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

import h5py


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

CFL_fac      = uw.options.getReal("CFL_fac", default=0.5)

# dt_max = uw.options.getInt("dt_max", default=1) ### Myr

if uw.mpi.rank == 0:
    print(f'csize = {csize}, CFL = {CFL_fac}, degree = {U_degree},')


# %%
T_initial = uw.options.getReal(name='temp_initial', default = 800) # Starting temperature

T_base = uw.options.getReal(name='temp_base', default = 750) # Final asymptotic temperature

profile = uw.options.getInt(name='profile', default = 4)

t_start = uw.options.getReal("time_start", default=0) ### Myr ( 1 - Ma, i.e. model start time)
t_end = uw.options.getReal("time_end", default=1500) ### Myr ( 1 - Ma, i.e. model end time)

t_pulse = uw.options.getReal(name='pulse_time', default = 1000) ### Myr, time after model start time (also the 2nd growth time)

duration = (t_end - t_start)

outputPath = uw.options.getString('outputPath', default= f'./output/Diffusion-Decay-ingrowth-profile={profile}_duration={int(duration)}_csize={csize}-cfl={CFL_fac}/')

if uw.mpi.rank == 0:
    os.makedirs(outputPath, exist_ok=True)

# %%
# Symbolic variable
t_sym = sp.symbols('t')

# Gaussian function
def gaussian(T_peak, T_base, t_peak, std, t_sym):
    return T_base + (T_peak - T_base) * sp.exp(-((t_sym - t_peak) ** 2) / (2 * std ** 2))

# Initial profile (always the same)
initial_T_profile = gaussian(T_initial, T_base, t_start, 40, t_sym)

# Select the temperature profile based on the 'profile' option
if profile == 1:
    temp_profile_expr = sp.Piecewise(
        (initial_T_profile, t_sym < 500),
        (gaussian(1000, T_base, t_pulse, 48, t_sym), t_sym >= 500)
    )
elif profile == 2:
    temp_profile_expr = sp.Piecewise(
        (initial_T_profile, t_sym < 500),
        (gaussian(950, T_base, t_pulse, 60, t_sym), t_sym >= 500)
    )
elif profile == 3:
    temp_profile_expr = sp.Piecewise(
        (initial_T_profile, t_sym < 500),
        (gaussian(900, T_base, t_pulse, 80, t_sym), t_sym >= 500)
    )
else:
    temp_profile_expr = sp.Piecewise(
        (initial_T_profile, t_sym < 500),
        (gaussian(850, T_base, t_pulse, 120, t_sym), t_sym >= 500)
    )

# Assume T is your sympy Piecewise function and t_sym is the symbol
T_func = sp.lambdify(t_sym, temp_profile_expr, modules='numpy')

    

# %%
time_profile = np.linspace(t_start, t_end, int(t_end) ) ### every 1 Myr
temp_profile = T_func(time_profile)

# Create the interpolator
temperature_interp = interp1d(
    time_profile, 
    temp_profile,
    kind='linear', 
    fill_value='extrapolate' 
)

if uw.is_notebook:
    plt.plot(time_profile, temp_profile)


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
import sympy as sp

# Symbolic variable for elapsed time (years)
t = sp.Symbol('t')

# Constants: half-lives in years
half_life_U235 = 703.8e6
half_life_U238 = 4.468e9

# Decay constants (yr^-1)
lambda_U235 = sp.log(2) / half_life_U235
lambda_U238 = sp.log(2) / half_life_U238

# Present-day U-238/U-235 ratio (dimensionless)
current_ratio_U238_to_U235 = 137.818

# Abundance at time t (years ago), normalized so present-day U-235 = 1
U235_fn = sp.Lambda(t, sp.exp(lambda_U235 * t))
U238_fn = sp.Lambda(t, current_ratio_U238_to_U235 * sp.exp(lambda_U238 * t))
ratio_fn = sp.Lambda(t, U238_fn(t) / U235_fn(t))

U235_amount = float(U235_fn(t_end * 1e6))
U238_amount = float(U238_fn(t_end * 1e6))
U238_U235_ratio = float(ratio_fn(t_end * 1e6))

if uw.mpi.rank == 0:
    print(f"U-235 abundance at {(t_end):1f} Ma (normalized): {U235_amount:.4f}")
    print(f"U-238 abundance at {(t_end):1f} Ma ago (normalized): {U238_amount:.4f}")
    print(f"U-238/U-235 ratio at {(t_end):1f} Ma ago: {U238_U235_ratio:.2f}")

# %%
# import unit registry
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 


diffusive_rate    = D_Pb_fn(T_initial) * u.meter**2 /u.second
model_length      = 100 * u.micrometer ### scale the mesh radius to the zircon radius


KL = model_length
Kt = model_length**2 / diffusive_rate


scaling_coefficients  = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt

scaling_coefficients


# ### Create analytical fn

# %%
### width (x) = 0.6 (60 micron), height (y) = 1. (100 micron)

points = [(-0.3, 0.35, 0), (-0.2, 0.5, 0), (0.2, 0.5, 0), (0.3, 0.35, 0), 
          (0.3, -0.35, 0), (0.2, -0.5, 0), (-0.2, -0.5, 0), (-0.3,-0.35, 0)]

points_array = np.array(points)

# %%
zircon_mesh = DIF.meshing.generate_2D_mesh_from_points(points_array, csize, U_degree)

# %%
# zircon_mesh.view(1)

# %%
U238_Pb206 = DIF.DiffusionDecayIngrowthModel(
        parent_name=r"{}^{238}\text{U}", daughter_name=r"{}^{206}\text{Pb}", half_life=half_life_U238*u.year, initial_parent=U238_amount, mesh=zircon_mesh
    )

U235_Pb207 = DIF.DiffusionDecayIngrowthModel (
        parent_name=r"{}^{235}\text{U}", daughter_name=r"{}^{207}\text{Pb}", half_life=half_life_U235*u.year, initial_parent=U235_amount, mesh=zircon_mesh
    )


# %%
T_initial = temperature_interp(0)

Pb_kappa = (D_Pb_fn(T_initial) * u.meter**2 / u.second)
U_kappa = (D_U_fn(T_initial) * u.meter**2 / u.second)


# %%
U238_Pb206.parent_diffusivity = D_U_fn(T_initial) * u.meter**2 / u.second

U238_Pb206.daughter_diffusivity = D_Pb_fn(T_initial) * u.meter**2 / u.second


# %%
U235_Pb207.parent_diffusivity = D_U_fn(T_initial) * u.meter**2 / u.second

U235_Pb207.daughter_diffusivity = D_Pb_fn(T_initial) * u.meter**2 / u.second


# %%
BC_value = sp.Float(0)

### Pb and U boundaries set to 0
for _solver in [U235_Pb207.parent_diffusion, U235_Pb207.daughter_diffusion, U238_Pb206.parent_diffusion, U238_Pb206.daughter_diffusion]:
    for _boundary in _solver.mesh.boundaries:
        if _boundary.name.lower().startswith('boundary'):
            _solver.add_dirichlet_bc([BC_value], _boundary.name)
    #     if 'innerBoundary' in _boundary.name:
    #         _solver.add_dirichlet_bc([BC_value], _boundary.name)


    _solver.petsc_options["snes_rtol"]   = tolerance*1e-4
    _solver.petsc_options["snes_atol"]   = tolerance
    _solver.petsc_options["ksp_atol"]    = tolerance
    _solver.petsc_options["snes_max_it"] = 100
    _solver.petsc_options["snes_monitor_short"] = None 

# %%
dTdt = np.diff(temp_profile) / np.diff(time_profile)
max_allowed_temp_change = 5 ### C
dt_temp_change = np.min(max_allowed_temp_change / np.maximum(np.abs(dTdt), 1e-12))


# %%
def update_kappa(Model, D_p_fn, D_d_fn, temperature_interp):
    current_temp = temperature_interp(dim(Model.current_time, u.megayear).m) #T_start - gradient*dim(Model.current_time, u.megayear).m
    kappa_p = D_p_fn(current_temp)
    kappa_d = D_d_fn(current_temp)
    Model.parent_diffusivity = kappa_p * u.meter**2 / u.second
    Model.daughter_diffusivity = kappa_d * u.meter**2 / u.second





# %%
def update_kappa_wrapper():
    update_kappa(U238_Pb206, D_U_fn, D_Pb_fn, temperature_interp)


U238_Pb206.register_pre_solve_hook('update kappa routine', update_kappa_wrapper)


# %%
U238_Pb206.run_simulation(duration=t_end*u.megayear, max_dt=dt_temp_change*u.megayear, min_dt=0.1*u.megayear, diffusion_time_step_factor=CFL_fac)


# %%
def update_kappa_wrapper():
    update_kappa(U235_Pb207, D_U_fn, D_Pb_fn, temperature_interp)


U235_Pb207.register_pre_solve_hook('update kappa routine', update_kappa_wrapper)


# %%
U235_Pb207.run_simulation(duration=t_end*u.megayear, max_dt=dt_temp_change*u.megayear, min_dt=0.1*u.megayear, diffusion_time_step_factor=CFL_fac)

# %%
with U238_Pb206.mesh.access(U238_Pb206.parent_mesh_var, U238_Pb206.daughter_mesh_var):
    ratio_U23Pb8206 = U238_Pb206.parent_mesh_var.data / U238_Pb206.daughter_mesh_var.data

# %%
with U235_Pb207.mesh.access(U235_Pb207.parent_mesh_var, U235_Pb207.daughter_mesh_var):
    ratio_U235_Pb207 = U235_Pb207.parent_mesh_var.data / U235_Pb207.daughter_mesh_var.data

# %%
with U235_Pb207.mesh.access(U235_Pb207.daughter_mesh_var), U238_Pb206.mesh.access(U238_Pb206.daughter_mesh_var):
    ratio_Pb207_Pb206 =  U235_Pb207.daughter_mesh_var.data / U238_Pb206.daughter_mesh_var.data

# %%
ratio_Pb207_Pb206[np.isnan(ratio_Pb207_Pb206)] = 0
ratio_U23Pb8206[np.isnan(ratio_U23Pb8206)] = 0

# %%
output_data = np.column_stack([U235_Pb207.daughter_mesh_var.coords[:,0], U235_Pb207.daughter_mesh_var.coords[:,1], ratio_Pb207_Pb206, ratio_U23Pb8206])

if uw.mpi.size == 1:

    np.savetxt(f"{outputPath}ratio_data.csv", output_data, delimiter=",", header="x,y,Pb207_Pb206,U238Pb206", comments='')
    np.savez_compressed(f"{outputPath}ratio_data.npz", data=output_data)

else:
    comm = uw.mpi.comm
    rank = uw.mpi.rank
    size = uw.mpi.size
    
    # output_data: shape (N_local, ncols) for each rank
    # You must know or compute N_local for each rank
    N_local = output_data.shape[0]
    ncols = output_data.shape[1]
    
    # Gather all N_local to compute offsets for each rank
    counts = comm.allgather(N_local)
    offset = sum(counts[:rank])
    total_rows = sum(counts)
    
    with h5py.File(f"{outputPath}ratio_data.h5", "w", driver="mpio", comm=comm) as f:
        dset = f.create_dataset("data", shape=(total_rows, ncols), dtype=output_data.dtype)
        dset[offset:offset+N_local, :] = output_data

# %%
# with h5py.File(f"{outputPath}ratio_data.h5", "r") as f:
#     data = f["data"][:]  # Read the entire dataset

# %%
np.random.seed(0)
sample_r = nd(23*u.micrometer) / 2  ### 23 micron radius spot
nsamples = 200

if uw.mpi.size == 1:
    data = np.load(f'{outputPath}/ratio_data.npz')['data']
else:
    with h5py.File(f"{outputPath}ratio_data.h5", "r") as f:
        data = f["data"][:]  # Read the entire dataset

# %%
# points: Nx2 array of all mesh points
points = np.column_stack([data[:,0], data[:,1]])

x_min, x_max = data[:,0].min(), data[:,0].max()
y_min, y_max = data[:,1].min(), data[:,1].max()
x_lower, x_upper = x_min + (sample_r ), x_max - (sample_r )
y_lower, y_upper = y_min + (sample_r ), y_max - (sample_r )

# Generate random sample centre points within the region
sample_centre_points = np.random.uniform(low=(x_lower, y_lower), 
                                        high=(x_upper, y_upper), 
                                        size=(nsamples, 2))

spot_data = np.zeros(shape=(sample_centre_points.shape[0], 2))

i = 0
for centre in sample_centre_points:
    result =  DIF.utilities.sample_spot(data[:,0:2], [data[:,2], data[:,3]], centre, sample_r)
    spot_data[i:,0], spot_data[i:,1] = np.average(result[1]), np.average(result[0])
    i += 1

# %%
if save_figs:
    import matplotlib.pyplot as plt
    import numpy as np
    data = np.load(f'{outputPath}/ratio_data.npz')['data']

    fig, ax = plt.subplots(1,2, figsize=(10, 8), sharey=True)
    
    
    data = np.load(f'{outputPath}/ratio_data.npz')['data']


    contour1 = DIF.utilities.plot_mesh_data(data[:,0:2]*100, data[:,2], ax=ax[0], vmin=np.quantile(data[:,2], 0.05), vmax=np.quantile(data[:,2], 0.95), levels=100)
    contour2 = DIF.utilities.plot_mesh_data(data[:,0:2]*100, data[:,3], ax=ax[1], vmin=np.quantile(data[:,3], 0.05), vmax=np.quantile(data[:,3], 0.95), levels=100)

    for spot in sample_centre_points:
        DIF.utilities.plot_spot_sample(spot[0:2]*100, radius=23/2, ax=ax[0], alpha=0.7, linewidth=1.)
        DIF.utilities.plot_spot_sample(spot[0:2]*100, radius=23/2, ax=ax[1], alpha=0.7, linewidth=1.)
   
    for _ax in ax:

        _ax.set_xlabel(r'x [$\mu$ m]')
        _ax.set_aspect('equal')

        _ax.set_xlim(-30, 30)
        _ax.set_ylim(-50, 50)

    ax[0].set_ylabel(r'y [$\mu$ m]')



    ax[0].set_title(r'$\frac{^{238}U}{^{206}Pb}$')
    ax[1].set_title(r'$\frac{^{207}Pb}{^{206}Pb}$')

    plt.colorbar(contour1, ax=ax[0])
    plt.colorbar(contour2, ax=ax[1])

    contour1.set_edgecolors("face")
    contour2.set_edgecolors("face")

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    plt.savefig(f'{outputPath}/zircon-U-Pb_isotope_ratio_mesh_plots.pdf', dpi=300)


# %%
if save_figs:
    fig, ax = plt.subplots(figsize=(10,6))
    DIF.utilities.plot_terra_wasserburg_plot(start_time=2000e6, end_time=100e6, marker_spacing=100e6, ax=ax)
    
    ax.scatter(ratio_U23Pb8206[~np.isnan(ratio_U23Pb8206)], ratio_Pb207_Pb206[~np.isnan(ratio_Pb207_Pb206)], zorder=0, s=1, label='All points')
    
    ax.set_xlim(3, 20)
    ax.set_ylim(0.055, 0.10)
    

    ax.scatter(spot_data[:,0], spot_data[:,1], marker='d', c='lightblue', s = 150, alpha=0.4, label='Spot data')

    ax.set_ylabel(r'$\frac{^{207}\mathrm{Pb}}{^{206}\mathrm{Pb}}$', rotation=0, labelpad=10)
    ax.set_xlabel(r'$\frac{^{238}\mathrm{U}}{^{206}\mathrm{Pb}}$', rotation=0, labelpad=10)
    
    ax.legend(fontsize=10, loc="best", frameon=True)

    plt.savefig(f'{outputPath}/concordia_plot+spot_data.pdf', bbox_inches='tight')


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
        'n_unknowns': U238_Pb206.daughter_mesh_var.coords.shape[0],
        'MinRadius': U238_Pb206.mesh.get_min_radius(),
        # 'Pb_L2_norm': Pb_l2_norm,
        # 'U_L2_norm' : U_l2_norm,
        'CFL_fac' : CFL_fac,
        'TotalTime': total_time,
        'NumProcessors': uw.mpi.size,
    }
    # Save as a single row (header + values)
    with open(f'{outputPath}/Diffusion_results_rows.csv', 'w') as f:
        f.write(','.join(summary.keys()) + '\n')
        f.write(','.join(str(v) for v in summary.values()) + '\n')


# %%
