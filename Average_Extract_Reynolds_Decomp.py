########################################################################################################################################################################
# Calculating Mean and Fluctuating Vorticity and Strain Rate Statistics from LES Solutions
# Based on: Welford Algorithm averaging framework from Compute_TKE_Transport.py
# Author: Patrick Deng (extended)
#
# Computes (full output — Vorticity_StrainRate_Stats.h5):
#   Mean velocity gradients:        du_dx_mean … dw_dz_mean  (all 9 components)
#   Mean vorticity vector:          omega_x_mean, omega_y_mean, omega_z_mean
#   Mean vorticity magnitude:       omega_mag_mean  = |<omega>|
#   Mean strain-rate tensor:        S_11_mean … S_23_mean  (6 unique components)
#   Mean strain-rate magnitude:     S_mag_mean  = sqrt(2 <Sij> <Sij>)
#   Mean enstrophy:                 enstrophy_mean  = 0.5 <omega_i><omega_i>
#   Fluctuating vorticity variance: omega_xx/yy/zz_fluct  = <omega_i' omega_i'>
#   Fluctuating vorticity cross:    omega_xy/xz/yz_fluct  = <omega_i' omega_j'>
#   Fluctuating vorticity RMS:      omega_x/y/z_rms
#   Fluctuating vorticity mag:      omega_fluct_mag  = sqrt(<omega_i' omega_i'>)
#   Enstrophy fluctuation variance: enstrophy_fluct_var
#   Fluctuating strain-rate prods:  Sf_11/22/33/12/13/23_sq_mean  = <Sij' Sij'>
#   Fluctuating strain-rate mag:    Sf_mag_mean  = sqrt(2 <Sij' Sij'>)
#   Pseudo-dissipation proxy:       epsilon_proxy  = 2 nu <Sij' Sij'>
#   Enstrophy production:           enstrophy_production  = <omega_i' Sij' omega_j'>
#
# Reduced output (Vorticity_StrainRate_Stats_Reduced.h5):
#   omega_x_mean, omega_y_mean, omega_z_mean   — mean vorticity vector
#   omega_x_rms,  omega_y_rms,  omega_z_rms   — fluctuating vorticity RMS
#   S_mag_mean                                 — mean strain-rate magnitude  sqrt(2<Sij><Sij>)
#   Sf_mag_mean                                — fluct. strain-rate magnitude sqrt(2<Sij'Sij'>)
#
# DO NOT MODIFY FUNCTIONS IN THIS FILE
########################################################################################################################################################################

import os
import numpy as np
import sys
from antares import *
import builtins

# -------------------------------------------------------------------------
# INPUT PARAMETERS  (mirror those used in TKE transport script)
# -------------------------------------------------------------------------
nstart   = 9        # solution dir count at which averaging begins

meshpath = '/project/rrg-moreaust-ac/denggua1/Bombardier_LES/B_10AOA_U50/MESH_Fine_Dec25/'
meshfile = 'Bombardier_10AOA_U50_Combine_Fine.mesh.h5'

sol_dirName = '/project/rrg-moreaust-ac/denggua1/Bombardier_LES/B_10AOA_U50/RUN_Fine/SOLUT/'

# Variables to strip from instantaneous solution files before assigning
# computed statistics (solver-specific fields not needed in output)
vars_base_clean = [
    'gamma_bar', 'hypvis_artif', 'hypvis_artif_y', 'mpi_rank', 'myzone', 'r_bar',
    'ss_bar', 'tau_turb_xy', 'tau_turb_xz', 'tau_turb_yz', 'vis_artif', 'vis_artif_y',
    'visco_mask', 'wall_EnergyFlux_normal', 'wall_EnergyFlux_x', 'wall_EnergyFlux_y',
    'wall_EnergyFlux_z', 'zeta_p', 'zeta_y', 'rhoE', 'rhou', 'rhov', 'rhow',
    'AIR', 'Q1', 'Q2',
    'wall_Stress_x', 'wall_Stress_y', 'wall_Stress_z',
    'wall_normal_Stress', 'wall_shear_Stress', 'wall_yplus',
    # raw instantaneous gradient fields — replaced by computed mean/fluct versions
    'du_dx', 'du_dy', 'du_dz',
    'dv_dx', 'dv_dy', 'dv_dz',
    'dw_dx', 'dw_dy', 'dw_dz',
    'vort_x', 'vort_y', 'vort_z',
    'pressure', 'rho',
    'div', 'Enstrophy', 'Strain', 'Pressure_Hessian', 'Strain_Rate',
]


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================

def print(text):
    """Print and flush immediately (mirrors original script)."""
    builtins.print(text)
    os.fsync(sys.stdout)


def sort_files(rand):
    """Return sorted, unique solution file stems (no collection/last_solution)."""
    rand_sub = os.listdir(rand)
    rand_arr = np.array([])
    for i in range(np.shape(rand_sub)[0]):
        file_split = os.path.splitext(rand_sub[i])[0]
        rand_arr = np.append(rand_arr, file_split)
    rand_arr = [*set(rand_arr)]
    rand_arr = [f for f in rand_arr if 'sol_collection' not in f]
    rand_arr = [f for f in rand_arr if 'last_solution'  not in f]
    rand_arr.sort()
    return rand_arr


def Calc_avg(mean, current, count):
    """Welford online mean update."""
    return mean + (1.0 / count) * (current - mean)


def extract_mesh(meshpath, meshfile):
    """Load the computational mesh."""
    text = 'Extracting the mesh'
    print(f'\n{text:.^80}\n')
    mesh_fileName = os.path.join(meshpath, meshfile)
    print(f'Mesh file: {mesh_fileName}')
    r = Reader('hdf_avbp')
    r['filename'] = mesh_fileName
    r['shared'] = True
    base = r.read()
    base.show()
    nodes = base[0].shared['x'].shape[0]
    return base, nodes


# =========================================================================
# HELPER: build the ordered list of solution files starting from nstart
# =========================================================================

def _build_file_list(ave_dirName):
    arr_dir = os.path.join(ave_dirName)
    arr = sorted(os.listdir(arr_dir))
    sol_dir = np.array([])
    for filename in arr:
        parts    = filename.split('_')
        sol_part = parts[-1].split('.')[0]
        sol_dir  = np.append(sol_dir, sol_part)
    sol_dir = np.unique(sol_dir)

    file_list = []
    for i in range(nstart, len(arr)):
        dir_path = os.path.join(arr_dir, arr[i])
        if os.path.isdir(dir_path):
            files = sort_files(dir_path)
            for f in files:
                file_list.append(os.path.join(dir_path, f + '.h5'))
    return file_list


# =========================================================================
# HELPER: assign a numpy array onto base and write an hdf_antares file
#         This is the canonical output pattern used for both full and reduced
#         outputs so both files are structurally identical to Paraview.
# =========================================================================

def _write_antares(base, fields_dict, filename):
    """
    Assign each (name -> numpy array) pair in fields_dict onto base[0][0],
    then write an hdf_antares file.  The base passed in should already have
    been loaded from a surrogate solution file and stripped of unwanted vars.
    """
    for name, arr in fields_dict.items():
        base[0][0][name] = arr.astype(np.float32)

    writer = Writer('hdf_antares')
    writer['filename'] = filename
    writer['base']     = base
    writer['dtype']    = 'float32'
    writer.dump()
    print(f'Written: {filename}.h5')


# =========================================================================
# MAIN COMPUTATION
# =========================================================================

def Extract_Vorticity_StrainRate(ave_dirName, base, nodes):
    """
    Two-pass Welford averaging for vorticity and strain-rate statistics.

    Pass 1 — Compute mean velocity gradients → mean vorticity vector,
              mean strain-rate tensor, and their scalar magnitudes.

    Pass 2 — Using first-pass means, accumulate fluctuating statistics:
              • Fluctuating vorticity variance and cross-correlations
              • Fluctuating strain-rate tensor products
              • Enstrophy production  <omega_i' Sij' omega_j'>

    Vorticity (right-hand rule, AVBP convention):
        omega_x =  dw/dy - dv/dz
        omega_y =  du/dz - dw/dx
        omega_z =  dv/dx - du/dy

    Strain-rate tensor (symmetric part of velocity gradient):
        S_ij = 0.5*(dUi/dxj + dUj/dxi)
        S_11 = du/dx
        S_22 = dv/dy
        S_33 = dw/dz
        S_12 = 0.5*(du/dy + dv/dx)
        S_13 = 0.5*(du/dz + dw/dx)
        S_23 = 0.5*(dv/dz + dw/dy)

    Scalar magnitudes:
        S_mag_mean  = sqrt(2 <Sij><Sij>)        — mean strain-rate magnitude
        Sf_mag_mean = sqrt(2 <Sij'Sij'>)        — fluctuating strain-rate magnitude
    """

    print(f'\n{"Building solution file list":.^80}\n')
    file_list   = _build_file_list(ave_dirName)
    total_files = len(file_list)
    print(f'Total files to process: {total_files}')

    # ------------------------------------------------------------------
    # FIRST PASS — mean velocity gradients → mean vorticity & strain rate
    # ------------------------------------------------------------------
    print(f'\n{"FIRST PASS: Mean Velocity Gradients":.^80}\n')

    du_dx_mean = np.zeros(nodes, dtype=np.float64)
    du_dy_mean = np.zeros(nodes, dtype=np.float64)
    du_dz_mean = np.zeros(nodes, dtype=np.float64)
    dv_dx_mean = np.zeros(nodes, dtype=np.float64)
    dv_dy_mean = np.zeros(nodes, dtype=np.float64)
    dv_dz_mean = np.zeros(nodes, dtype=np.float64)
    dw_dx_mean = np.zeros(nodes, dtype=np.float64)
    dw_dy_mean = np.zeros(nodes, dtype=np.float64)
    dw_dz_mean = np.zeros(nodes, dtype=np.float64)

    count = 0
    for sol_file in file_list:
        if count % 10 == 0:
            print(f'  Pass 1 — file {count+1}/{total_files}: {os.path.basename(sol_file)}')

        r = Reader('hdf_avbp')
        r['base']     = base
        r['filename'] = sol_file
        base          = r.read()

        du_dx_mean = Calc_avg(du_dx_mean, base[0][0]['du_dx'], count + 1)
        du_dy_mean = Calc_avg(du_dy_mean, base[0][0]['du_dy'], count + 1)
        du_dz_mean = Calc_avg(du_dz_mean, base[0][0]['du_dz'], count + 1)
        dv_dx_mean = Calc_avg(dv_dx_mean, base[0][0]['dv_dx'], count + 1)
        dv_dy_mean = Calc_avg(dv_dy_mean, base[0][0]['dv_dy'], count + 1)
        dv_dz_mean = Calc_avg(dv_dz_mean, base[0][0]['dv_dz'], count + 1)
        dw_dx_mean = Calc_avg(dw_dx_mean, base[0][0]['dw_dx'], count + 1)
        dw_dy_mean = Calc_avg(dw_dy_mean, base[0][0]['dw_dy'], count + 1)
        dw_dz_mean = Calc_avg(dw_dz_mean, base[0][0]['dw_dz'], count + 1)
        count += 1

    print(f'First pass complete ({count} files).')

    # Derived mean vorticity components
    omega_x_mean = dw_dy_mean - dv_dz_mean   # <omega_x> = <dw/dy> - <dv/dz>
    omega_y_mean = du_dz_mean - dw_dx_mean   # <omega_y> = <du/dz> - <dw/dx>
    omega_z_mean = dv_dx_mean - du_dy_mean   # <omega_z> = <dv/dx> - <du/dy>

    # Mean strain-rate tensor components (symmetric)
    S_11_mean = du_dx_mean                             # S11 = dU/dx
    S_22_mean = dv_dy_mean                             # S22 = dV/dy
    S_33_mean = dw_dz_mean                             # S33 = dW/dz
    S_12_mean = 0.5 * (du_dy_mean + dv_dx_mean)        # S12 = 0.5*(dU/dy + dV/dx)
    S_13_mean = 0.5 * (du_dz_mean + dw_dx_mean)        # S13 = 0.5*(dU/dz + dW/dx)
    S_23_mean = 0.5 * (dv_dz_mean + dw_dy_mean)        # S23 = 0.5*(dV/dz + dW/dy)

    # Scalar magnitudes from mean quantities
    omega_mag_mean = np.sqrt(omega_x_mean**2 + omega_y_mean**2 + omega_z_mean**2)
    # sqrt(2 Sij Sij) — factor of 2 accounts for off-diagonal symmetry
    S_mag_mean = np.sqrt(2.0 * (S_11_mean**2 + S_22_mean**2 + S_33_mean**2
                                + 2.0 * S_12_mean**2
                                + 2.0 * S_13_mean**2
                                + 2.0 * S_23_mean**2))
    # Mean enstrophy  0.5 <omega_i><omega_i>
    enstrophy_mean = 0.5 * (omega_x_mean**2 + omega_y_mean**2 + omega_z_mean**2)

    # ------------------------------------------------------------------
    # SECOND PASS — fluctuating vorticity and strain-rate statistics
    # ------------------------------------------------------------------
    print(f'\n{"SECOND PASS: Fluctuating Vorticity & Strain Rate":.^80}\n')

    # Fluctuating vorticity: variance (diagonal) and cross-correlations
    omf_xx_mean = np.zeros(nodes, dtype=np.float64)   # <omega_x' omega_x'>
    omf_yy_mean = np.zeros(nodes, dtype=np.float64)   # <omega_y' omega_y'>
    omf_zz_mean = np.zeros(nodes, dtype=np.float64)   # <omega_z' omega_z'>
    omf_xy_mean = np.zeros(nodes, dtype=np.float64)   # <omega_x' omega_y'>
    omf_xz_mean = np.zeros(nodes, dtype=np.float64)   # <omega_x' omega_z'>
    omf_yz_mean = np.zeros(nodes, dtype=np.float64)   # <omega_y' omega_z'>

    # Fluctuating enstrophy variance  <Omega'^2>
    enstrophy_fluct_var = np.zeros(nodes, dtype=np.float64)

    # Fluctuating strain-rate products  <Sij' Sij'>  (6 unique components)
    Sf_11_sq_mean = np.zeros(nodes, dtype=np.float64)  # <(du'/dx)^2>
    Sf_22_sq_mean = np.zeros(nodes, dtype=np.float64)  # <(dv'/dy)^2>
    Sf_33_sq_mean = np.zeros(nodes, dtype=np.float64)  # <(dw'/dz)^2>
    Sf_12_sq_mean = np.zeros(nodes, dtype=np.float64)  # <S12' S12'>
    Sf_13_sq_mean = np.zeros(nodes, dtype=np.float64)  # <S13' S13'>
    Sf_23_sq_mean = np.zeros(nodes, dtype=np.float64)  # <S23' S23'>

    # Enstrophy production  <omega_i' Sij' omega_j'>
    enstrophy_production = np.zeros(nodes, dtype=np.float64)

    count = 0
    for sol_file in file_list:
        if count % 10 == 0:
            print(f'  Pass 2 — file {count+1}/{total_files}: {os.path.basename(sol_file)}')

        r = Reader('hdf_avbp')
        r['base']     = base
        r['filename'] = sol_file
        base          = r.read()

        # Fluctuating velocity gradients  (instantaneous - mean)
        du_dx_f = base[0][0]['du_dx'] - du_dx_mean
        du_dy_f = base[0][0]['du_dy'] - du_dy_mean
        du_dz_f = base[0][0]['du_dz'] - du_dz_mean
        dv_dx_f = base[0][0]['dv_dx'] - dv_dx_mean
        dv_dy_f = base[0][0]['dv_dy'] - dv_dy_mean
        dv_dz_f = base[0][0]['dv_dz'] - dv_dz_mean
        dw_dx_f = base[0][0]['dw_dx'] - dw_dx_mean
        dw_dy_f = base[0][0]['dw_dy'] - dw_dy_mean
        dw_dz_f = base[0][0]['dw_dz'] - dw_dz_mean

        # Fluctuating vorticity components
        omf_x = dw_dy_f - dv_dz_f   # omega_x' = dw'/dy - dv'/dz
        omf_y = du_dz_f - dw_dx_f   # omega_y' = du'/dz - dw'/dx
        omf_z = dv_dx_f - du_dy_f   # omega_z' = dv'/dx - du'/dy

        # Fluctuating strain-rate tensor components
        Sf_11 = du_dx_f
        Sf_22 = dv_dy_f
        Sf_33 = dw_dz_f
        Sf_12 = 0.5 * (du_dy_f + dv_dx_f)
        Sf_13 = 0.5 * (du_dz_f + dw_dx_f)
        Sf_23 = 0.5 * (dv_dz_f + dw_dy_f)

        # Vorticity statistics
        omf_xx_mean = Calc_avg(omf_xx_mean, omf_x * omf_x, count + 1)
        omf_yy_mean = Calc_avg(omf_yy_mean, omf_y * omf_y, count + 1)
        omf_zz_mean = Calc_avg(omf_zz_mean, omf_z * omf_z, count + 1)
        omf_xy_mean = Calc_avg(omf_xy_mean, omf_x * omf_y, count + 1)
        omf_xz_mean = Calc_avg(omf_xz_mean, omf_x * omf_z, count + 1)
        omf_yz_mean = Calc_avg(omf_yz_mean, omf_y * omf_z, count + 1)

        # Fluctuating enstrophy variance
        Omega_f = 0.5 * (omf_x**2 + omf_y**2 + omf_z**2)
        enstrophy_fluct_var = Calc_avg(enstrophy_fluct_var, Omega_f * Omega_f, count + 1)

        # Fluctuating strain-rate products
        Sf_11_sq_mean = Calc_avg(Sf_11_sq_mean, Sf_11 * Sf_11, count + 1)
        Sf_22_sq_mean = Calc_avg(Sf_22_sq_mean, Sf_22 * Sf_22, count + 1)
        Sf_33_sq_mean = Calc_avg(Sf_33_sq_mean, Sf_33 * Sf_33, count + 1)
        Sf_12_sq_mean = Calc_avg(Sf_12_sq_mean, Sf_12 * Sf_12, count + 1)
        Sf_13_sq_mean = Calc_avg(Sf_13_sq_mean, Sf_13 * Sf_13, count + 1)
        Sf_23_sq_mean = Calc_avg(Sf_23_sq_mean, Sf_23 * Sf_23, count + 1)

        # Enstrophy production:  omega_i' Sij' omega_j'
        #   = S11*wx^2 + S22*wy^2 + S33*wz^2
        #     + 2*S12*wx*wy + 2*S13*wx*wz + 2*S23*wy*wz
        ep_inst = (Sf_11 * omf_x**2
                 + Sf_22 * omf_y**2
                 + Sf_33 * omf_z**2
                 + 2.0 * Sf_12 * omf_x * omf_y
                 + 2.0 * Sf_13 * omf_x * omf_z
                 + 2.0 * Sf_23 * omf_y * omf_z)
        enstrophy_production = Calc_avg(enstrophy_production, ep_inst, count + 1)

        count += 1

    print(f'Second pass complete ({count} files).')

    # Post-pass derived scalars
    omega_x_rms     = np.sqrt(np.maximum(omf_xx_mean, 0.0))
    omega_y_rms     = np.sqrt(np.maximum(omf_yy_mean, 0.0))
    omega_z_rms     = np.sqrt(np.maximum(omf_zz_mean, 0.0))
    omega_fluct_mag = np.sqrt(np.maximum(omf_xx_mean + omf_yy_mean + omf_zz_mean, 0.0))

    # Fluctuating strain-rate magnitude:  sqrt(2 <Sij' Sij'>)
    Sf_mag_mean = np.sqrt(2.0 * np.maximum(
        Sf_11_sq_mean + Sf_22_sq_mean + Sf_33_sq_mean
        + 2.0 * Sf_12_sq_mean
        + 2.0 * Sf_13_sq_mean
        + 2.0 * Sf_23_sq_mean, 0.0))

    # Pseudo-dissipation proxy:  2 nu <Sij' Sij'>
    nu = 1.46e-5
    epsilon_proxy = nu * Sf_mag_mean**2   # = 2 nu <Sij' Sij'>  (Sf_mag^2 = 2<Sij'Sij'>)

    # ------------------------------------------------------------------
    # LOAD SURROGATE BASE — read first file, strip solver-specific vars,
    # use as the mesh carrier for both output files
    # ------------------------------------------------------------------
    print(f'\n{"Loading surrogate base for output":.^80}\n')
    r = Reader('hdf_avbp')
    r['base']     = base
    r['filename'] = file_list[0]
    base_out = r.read()
    # Only delete vars that actually exist in this file to avoid Antares errors
    existing = {(v[0] if isinstance(v, tuple) else v) for v in base_out[0][0].keys()}
    to_delete = [v for v in vars_base_clean if v in existing]
    if to_delete:
        base_out.delete_variables(to_delete)

    # ------------------------------------------------------------------
    # FULL OUTPUT — all computed statistics
    # ------------------------------------------------------------------
    print(f'\n{"Writing full output":.^80}\n')

    full_fields = {
        # Mean velocity gradients
        'du_dx_mean': du_dx_mean, 'du_dy_mean': du_dy_mean, 'du_dz_mean': du_dz_mean,
        'dv_dx_mean': dv_dx_mean, 'dv_dy_mean': dv_dy_mean, 'dv_dz_mean': dv_dz_mean,
        'dw_dx_mean': dw_dx_mean, 'dw_dy_mean': dw_dy_mean, 'dw_dz_mean': dw_dz_mean,
        # Mean vorticity
        'omega_x_mean':   omega_x_mean,
        'omega_y_mean':   omega_y_mean,
        'omega_z_mean':   omega_z_mean,
        'omega_mag_mean': omega_mag_mean,
        # Mean strain-rate tensor
        'S_11_mean': S_11_mean, 'S_22_mean': S_22_mean, 'S_33_mean': S_33_mean,
        'S_12_mean': S_12_mean, 'S_13_mean': S_13_mean, 'S_23_mean': S_23_mean,
        'S_mag_mean': S_mag_mean,
        # Mean enstrophy
        'enstrophy_mean': enstrophy_mean,
        # Fluctuating vorticity statistics
        'omega_xx_fluct':      omf_xx_mean,
        'omega_yy_fluct':      omf_yy_mean,
        'omega_zz_fluct':      omf_zz_mean,
        'omega_xy_fluct':      omf_xy_mean,
        'omega_xz_fluct':      omf_xz_mean,
        'omega_yz_fluct':      omf_yz_mean,
        'omega_x_rms':         omega_x_rms,
        'omega_y_rms':         omega_y_rms,
        'omega_z_rms':         omega_z_rms,
        'omega_fluct_mag':     omega_fluct_mag,
        'enstrophy_fluct_var': enstrophy_fluct_var,
        # Fluctuating strain-rate products and derived scalars
        'Sf_11_sq_mean': Sf_11_sq_mean,
        'Sf_22_sq_mean': Sf_22_sq_mean,
        'Sf_33_sq_mean': Sf_33_sq_mean,
        'Sf_12_sq_mean': Sf_12_sq_mean,
        'Sf_13_sq_mean': Sf_13_sq_mean,
        'Sf_23_sq_mean': Sf_23_sq_mean,
        'Sf_mag_mean':   Sf_mag_mean,
        'epsilon_proxy': epsilon_proxy,
        # Enstrophy production
        'enstrophy_production': enstrophy_production,
    }

    import copy
    _write_antares(copy.deepcopy(base_out), full_fields, 'Averaged_Solution_Vorticity_StrainRate')

    # ------------------------------------------------------------------
    # REDUCED OUTPUT — load a fresh surrogate base independently,
    # assign only the reduced variable set, write in identical format
    # ------------------------------------------------------------------
    print(f'\n{"Writing reduced output":.^80}\n')

    r = Reader('hdf_avbp')
    r['base']     = base
    r['filename'] = file_list[0]
    base_red = r.read()
    existing_red = {(v[0] if isinstance(v, tuple) else v) for v in base_red[0][0].keys()}
    to_delete_red = [v for v in vars_base_clean if v in existing_red]
    if to_delete_red:
        base_red.delete_variables(to_delete_red)

    reduced_fields = {
        # Mean vorticity vector
        'omega_x_mean': omega_x_mean,
        'omega_y_mean': omega_y_mean,
        'omega_z_mean': omega_z_mean,
        # Fluctuating vorticity RMS (per component)
        'omega_x_rms':  omega_x_rms,
        'omega_y_rms':  omega_y_rms,
        'omega_z_rms':  omega_z_rms,
        # Mean strain-rate magnitude  sqrt(2 <Sij><Sij>)
        'S_mag_mean':   S_mag_mean,
        # Fluctuating strain-rate magnitude  sqrt(2 <Sij'Sij'>)
        'Sf_mag_mean':  Sf_mag_mean,
    }

    _write_antares(base_red, reduced_fields, 'Averaged_Solution_Vorticity_StrainRate_Reduced')

    print(f'\n{"All outputs written successfully!":.^80}\n')
    return base_out


# =========================================================================
# ENTRY POINT
# =========================================================================

def main():
    sys.stdout = open('log_Vorticity_StrainRate_Stats.txt', 'w', buffering=1)
    base, nodes = extract_mesh(meshpath, meshfile)
    Extract_Vorticity_StrainRate(sol_dirName, base, nodes)


if __name__ == '__main__':
    main()