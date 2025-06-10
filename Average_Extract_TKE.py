####################################################################################
# Appending mean data from AVE files and compute mean quantities from the instaneous field only
# Author: Patrick Deng
# Enhanced with complete TKE transport equation analysis
# DO NOT MODIFY FUNCTIONS IN THIS FILE
####################################################################################
# The required modules
import os
import numpy as np
import sys
from antares import *
import h5py
import copy
import glob
import builtins
import shutil
import csv
import matplotlib.pyplot as plt

## The requried input values
nstart = 20      # solution dir count to begin the extract
# The path of the source mesh
meshpath = '/project/p/plavoie/denggua1/BBDB_10AOA/MESH_ZONE_Apr24/'
# The name of the source mesh (IF MESH CONTAINS ZONES, REQUIRE ZONE MERGING A-PRIORI)
meshfile = 'Bombardier_10AOA_Combine_Apr24.mesh.h5'
# The path of the average solution directory
sol_dirName = '/project/p/plavoie/denggua1/BBDB_10AOA/RUN_ZONE_Apr24/SOLUT/'

# Variables to delete for final output
vars_ave = ['gamma_bar', 'hypvis_artif', 'hypvis_artif_y', 'mpi_rank', 'myzone', 'r_bar', 
        'ss_bar', 'tau_turb_xy', 'tau_turb_xz', 'tau_turb_yz', 'vis_artif', 'vis_artif_y', 
        'visco_mask', 'wall_EnergyFlux_normal', 'wall_EnergyFlux_x', 'wall_EnergyFlux_y', 
        'wall_EnergyFlux_z', 'zeta_p', 'zeta_y', 'rho', 'rhoE', 'rhou', 'rhov', 'rhow', 
        'AIR','Q1','Q2','Enstrophy','Strain','Pressure_Hessian','Strain_Rate','du_dx','du_dy','du_dz','dv_dx','dv_dy','dv_dz','dw_dx','dw_dy','dw_dz',
        'vort_y','vort_z','div']

# Variables to delete for intermediate processing
vars_inst = ['gamma_bar', 'hypvis_artif', 'hypvis_artif_y', 'mpi_rank', 'myzone', 'r_bar', 
        'ss_bar', 'tau_turb_xy', 'tau_turb_xz', 'tau_turb_yz', 'vis_artif', 'vis_artif_y', 
        'visco_mask', 'wall_EnergyFlux_normal', 'wall_EnergyFlux_x', 'wall_EnergyFlux_y', 
        'wall_EnergyFlux_z', 'zeta_p', 'zeta_y', 'rhoE', 'rhou', 'rhov', 'rhow', 
        'AIR','Q1','Q2','wall_Stress_x','wall_Stress_y','wall_Stress_z','wall_normal_Stress','wall_shear_Stress','wall_yplus',
        'du_dx','du_dy','du_dz','dv_dx','dv_dy','dv_dz','dw_dx','dw_dy','dw_dz',
        'vort_y','vort_z']

# Function to take out the unique file names in the folder
def sort_files(rand):  # Sorting the files in the subdirectory and filtering non-solution data and lists
    rand_sub = os.listdir(rand)
    rand_arr = np.array([]) 
    for i in range(0,np.shape(rand_sub)[0]):
        file_split = os.path.splitext(rand_sub[i])[0]
        rand_arr = np.append(rand_arr,file_split)
    rand_arr = [*set(rand_arr)] # removing the duplicates
    rand_arr = [ rand_arr for rand_arr in rand_arr if 'sol_collection' not in rand_arr]     # Removing the collection files
    rand_arr = [ rand_arr for rand_arr in rand_arr if 'last_solution' not in rand_arr]      # Removing the last solution file links
    rand_arr.sort()
    return rand_arr

def extract_mesh(meshpath,meshfile):    # function to extract the mesh
    ## Loading the mesh
    text = 'Extracting the mesh'
    print(f'\n{text:.^80}\n')  
    mesh_fileName = os.path.join(meshpath,meshfile)
    print(f'Mesh file: {mesh_fileName}')
    # # Reading the mesh
    r = Reader('hdf_avbp')
    r['filename'] = mesh_fileName
    r['shared'] = True # Same mesh for all sol
    base = r.read() # b is the Base object of the Antares API
    base.show()
    nodes = base[0].shared['x'].shape[0]            # The number of nodes in the mesh
    return base, nodes

def Welford_avg(mean, current, iter):    # Welford's algorithm for calculating the mean and variance
    """
    Welford's algorithm for calculating the mean
    """
    mean = mean + (1/iter)*(current-mean) if iter > 0 else current
    return mean

def Extract_data_TKE(ave_dirName, base, nodes, vars_delete):
    """
    Three-pass method for accurate turbulence statistics including velocity-TKE correlations:
    Pass 1: Calculate mean velocities, pressure, and velocity gradients
    Pass 2: Calculate mean TKE using first-pass means
    Pass 3: Calculate all fluctuating quantities and correlations using final means
    """
    loc = 'node'  # Location for gradient calculations
    ## Iterating the solution directory
    text = 'Iterating Solution'
    arr_dir = os.path.join(ave_dirName)
    arr = os.listdir(arr_dir)
    arr.sort()
    arr = list(arr)
    sol_dir = np.array([])
    for i in range(0,len(arr)):
        filename = arr[i]
        parts = filename.split('_')
        sol_dir_part = parts[len(parts)-1].split('.')[0]
        sol_dir = np.append(sol_dir, sol_dir_part)
    sol_dir = np.unique(sol_dir)        # Finding the unique files in the directory
    print(f'\n{text:.^80}\n') 
    
    # Collect all file paths for processing
    file_list = []
    for i in range(nstart, np.shape(sol_dir)[0]):
        dir_path = os.path.join(arr_dir, arr[i])
        if os.path.isdir(dir_path):
            files = sort_files(dir_path)
            for j in range(len(files)):
                sol_file = os.path.join(dir_path, files[j] + '.h5')
                file_list.append(sol_file)
    
    total_files = len(file_list)
    print(f'Total files to process: {total_files}')
    
    # =================================================================================
    # FIRST PASS: Calculate mean quantities
    # =================================================================================
    print(f'\n{"FIRST PASS: Calculating Mean Quantities":.^80}\n')
    
    # Initialize mean quantities
    P_mean = np.zeros((nodes,), dtype=np.float32)
    u_mean = np.zeros((nodes,), dtype=np.float32)
    v_mean = np.zeros((nodes,), dtype=np.float32)
    w_mean = np.zeros((nodes,), dtype=np.float32)
    TKE_mean = np.zeros((nodes,), dtype=np.float64)
    
    # Initialize the velocity gradient components
    du_dx = np.zeros((nodes,), dtype=np.float32)
    du_dy = np.zeros((nodes,), dtype=np.float32)
    du_dz = np.zeros((nodes,), dtype=np.float32)
    dv_dx = np.zeros((nodes,), dtype=np.float32)
    dv_dy = np.zeros((nodes,), dtype=np.float32)
    dv_dz = np.zeros((nodes,), dtype=np.float32)
    dw_dx = np.zeros((nodes,), dtype=np.float32)
    dw_dy = np.zeros((nodes,), dtype=np.float32)
    dw_dz = np.zeros((nodes,), dtype=np.float32)
    
    count = 0
    for sol_file in file_list:
        if count % 10 == 0:
            print(f'First pass - Processing file {count+1}/{total_files}: {os.path.basename(sol_file)}')
        
        r = Reader('hdf_avbp')
        r['base'] = base
        r['filename'] = sol_file
        base = r.read()
        
        # Extract instantaneous values
        P_inst = base[0][0]['pressure']
        base.compute('u = rhou / rho')
        base.compute('v = rhov / rho')
        base.compute('w = rhow / rho')
        u_inst = base[0][0]['u']
        v_inst = base[0][0]['v']
        w_inst = base[0][0]['w']
        
        # Update means using Welford's algorithm
        P_mean = Welford_avg(P_mean, P_inst, count)
        u_mean = Welford_avg(u_mean, u_inst, count)
        v_mean = Welford_avg(v_mean, v_inst, count)
        w_mean = Welford_avg(w_mean, w_inst, count)
        
        # The mean velocity gradients
        du_dx = Welford_avg(du_dx, base[0][0]['du_dx'], count)
        du_dy = Welford_avg(du_dy, base[0][0]['du_dy'], count)
        du_dz = Welford_avg(du_dz, base[0][0]['du_dz'], count)
        dv_dx = Welford_avg(dv_dx, base[0][0]['dv_dx'], count)
        dv_dy = Welford_avg(dv_dy, base[0][0]['dv_dy'], count)
        dv_dz = Welford_avg(dv_dz, base[0][0]['dv_dz'], count)
        dw_dx = Welford_avg(dw_dx, base[0][0]['dw_dx'], count)
        dw_dy = Welford_avg(dw_dy, base[0][0]['dw_dy'], count)
        dw_dz = Welford_avg(dw_dz, base[0][0]['dw_dz'], count)
        count += 1
    
    print(f'First pass complete. Mean values calculated from {count} files.')
    
    # =================================================================================
    # SECOND PASS: Calculate mean TKE
    # =================================================================================
    count = 0
    print(f'\n{"SECOND PASS: Calculating Mean TKE":.^80}\n')
    for sol_file in file_list:
        if count % 10 == 0:
            print(f'Second pass - Processing file {count+1}/{total_files}: {os.path.basename(sol_file)}')
        
        r = Reader('hdf_avbp')
        r['base'] = base
        r['filename'] = sol_file
        base = r.read()
        
        # Extract instantaneous values
        P_inst = base[0][0]['pressure']
        base.compute('u = rhou / rho')
        base.compute('v = rhov / rho')
        base.compute('w = rhow / rho')
        u_inst = base[0][0]['u']
        v_inst = base[0][0]['v']
        w_inst = base[0][0]['w']
        
        # Calculate instantaneous TKE from velocity fluctuations relative to final means
        TKE_inst = 0.5 * ((u_inst - u_mean)**2 + (v_inst - v_mean)**2 + (w_inst - w_mean)**2)
        TKE_mean = Welford_avg(TKE_mean, TKE_inst, count)
        count += 1
    
    print(f'Second pass complete. Mean TKE calculated from {count} files.')
    
    # =================================================================================
    # THIRD PASS: Calculate fluctuating quantities and correlations
    # =================================================================================
    print(f'\n{"THIRD PASS: Calculating Turbulence Statistics":.^80}\n')
    
    # Initialize Reynolds stress components
    uu_mean = np.zeros((nodes,), dtype=np.float64)
    vv_mean = np.zeros((nodes,), dtype=np.float64)
    ww_mean = np.zeros((nodes,), dtype=np.float64)
    uv_mean = np.zeros((nodes,), dtype=np.float64)
    uw_mean = np.zeros((nodes,), dtype=np.float64)
    vw_mean = np.zeros((nodes,), dtype=np.float64)
    pp_mean = np.zeros((nodes,), dtype=np.float64)
    
    # Initialize pressure-velocity correlations
    up_mean = np.zeros((nodes,), dtype=np.float64)
    vp_mean = np.zeros((nodes,), dtype=np.float64)
    wp_mean = np.zeros((nodes,), dtype=np.float64)
    
    # Initialize velocity-TKE correlations
    u_TKE_mean = np.zeros((nodes,), dtype=np.float64)
    v_TKE_mean = np.zeros((nodes,), dtype=np.float64)
    w_TKE_mean = np.zeros((nodes,), dtype=np.float64)
    
    # Initialize the fluctuating gradients
    du_dx_fluct = np.zeros((nodes,), dtype=np.float32)
    du_dy_fluct = np.zeros((nodes,), dtype=np.float32)
    du_dz_fluct = np.zeros((nodes,), dtype=np.float32)
    dv_dx_fluct = np.zeros((nodes,), dtype=np.float32)
    dv_dy_fluct = np.zeros((nodes,), dtype=np.float32)
    dv_dz_fluct = np.zeros((nodes,), dtype=np.float32)
    dw_dx_fluct = np.zeros((nodes,), dtype=np.float32)
    dw_dy_fluct = np.zeros((nodes,), dtype=np.float32)
    dw_dz_fluct = np.zeros((nodes,), dtype=np.float32)
    
    # Initialize TKE fluctuation statistics
    TKE_TKE_mean = np.zeros((nodes,), dtype=np.float64)  # TKE variance (TKE')^2
    
    count = 0
    for sol_file in file_list:
        if count % 10 == 0:
            print(f'Third pass - Processing file {count+1}/{total_files}: {os.path.basename(sol_file)}')
        
        r = Reader('hdf_avbp')
        r['base'] = base
        r['filename'] = sol_file
        base = r.read()
        
        # Extract instantaneous values
        P_inst = base[0][0]['pressure']
        base.compute('u = rhou / rho')
        base.compute('v = rhov / rho')
        base.compute('w = rhow / rho')
        u_inst = base[0][0]['u']
        v_inst = base[0][0]['v']
        w_inst = base[0][0]['w']
        
        # Calculate fluctuations relative to final means
        u_prime = u_inst - u_mean
        v_prime = v_inst - v_mean
        w_prime = w_inst - w_mean
        p_prime = P_inst - P_mean
        
        # Calculate instantaneous TKE from fluctuating velocities
        TKE_inst = 0.5 * (u_prime**2 + v_prime**2 + w_prime**2)
        # Calculate fluctuating TKE
        TKE_prime = TKE_inst - TKE_mean
        
        # Update Reynolds stress components
        uu_mean = Welford_avg(uu_mean, u_prime * u_prime, count)
        vv_mean = Welford_avg(vv_mean, v_prime * v_prime, count)
        ww_mean = Welford_avg(ww_mean, w_prime * w_prime, count)
        uv_mean = Welford_avg(uv_mean, u_prime * v_prime, count)
        uw_mean = Welford_avg(uw_mean, u_prime * w_prime, count)
        vw_mean = Welford_avg(vw_mean, v_prime * w_prime, count)
        pp_mean = Welford_avg(pp_mean, p_prime * p_prime, count)
        
        # Update pressure-velocity correlations
        up_mean = Welford_avg(up_mean, u_prime * p_prime, count)
        vp_mean = Welford_avg(vp_mean, v_prime * p_prime, count)
        wp_mean = Welford_avg(wp_mean, w_prime * p_prime, count)
        
        # Update velocity-TKE correlations (u' * TKE', v' * TKE', w' * TKE')
        u_TKE_mean = Welford_avg(u_TKE_mean, u_prime * TKE_prime, count)
        v_TKE_mean = Welford_avg(v_TKE_mean, v_prime * TKE_prime, count)
        w_TKE_mean = Welford_avg(w_TKE_mean, w_prime * TKE_prime, count)
        
        # Update fluctuating gradients
        du_dx_fluct = Welford_avg(du_dx_fluct, base[0][0]['du_dx'] - du_dx, count)
        du_dy_fluct = Welford_avg(du_dy_fluct, base[0][0]['du_dy'] - du_dy, count)
        du_dz_fluct = Welford_avg(du_dz_fluct, base[0][0]['du_dz'] - du_dz, count)
        dv_dx_fluct = Welford_avg(dv_dx_fluct, base[0][0]['dv_dx'] - dv_dx, count)
        dv_dy_fluct = Welford_avg(dv_dy_fluct, base[0][0]['dv_dy'] - dv_dy, count)
        dv_dz_fluct = Welford_avg(dv_dz_fluct, base[0][0]['dv_dz'] - dv_dz, count)
        dw_dx_fluct = Welford_avg(dw_dx_fluct, base[0][0]['dw_dx'] - dw_dx, count)
        dw_dy_fluct = Welford_avg(dw_dy_fluct, base[0][0]['dw_dy'] - dw_dy, count)
        dw_dz_fluct = Welford_avg(dw_dz_fluct, base[0][0]['dw_dz'] - dw_dz, count)
        
        # Update TKE variance
        TKE_TKE_mean = Welford_avg(TKE_TKE_mean, TKE_prime * TKE_prime, count)
        
        count += 1
    
    # Calculate final derived quantities
    TKE_final = 0.5 * (uu_mean + vv_mean + ww_mean)  # Alternative TKE calculation
    
    # RMS values
    u_rms = np.sqrt(uu_mean)
    v_rms = np.sqrt(vv_mean)
    w_rms = np.sqrt(ww_mean)
    P_rms = np.sqrt(pp_mean)
    TKE_rms = np.sqrt(TKE_TKE_mean)
    
    # Print summary statistics
    print(f'Third pass complete. Turbulence statistics calculated from {count} files.')
    
    # =================================================================================
    # Outputting the averaged results
    # =================================================================================
    r = Reader('hdf_avbp')
    r['base'] = base
    r['filename'] = file_list[0]
    base = r.read()
    base.delete_variables(vars_delete)
    
    # The mean flow quantities
    base[0][0]['P'] = P_mean
    base[0][0]['u'] = u_mean
    base[0][0]['v'] = v_mean
    base[0][0]['w'] = w_mean
    
    # The mean flow gradients
    base[0][0]['du_dx'] = du_dx
    base[0][0]['du_dy'] = du_dy
    base[0][0]['du_dz'] = du_dz
    base[0][0]['dv_dx'] = dv_dx
    base[0][0]['dv_dy'] = dv_dy
    base[0][0]['dv_dz'] = dv_dz
    base[0][0]['dw_dx'] = dw_dx
    base[0][0]['dw_dy'] = dw_dy
    base[0][0]['dw_dz'] = dw_dz
    
    # The fluctuating gradients
    base[0][0]['du_dx_fluct'] = du_dx_fluct
    base[0][0]['du_dy_fluct'] = du_dy_fluct
    base[0][0]['du_dz_fluct'] = du_dz_fluct
    base[0][0]['dv_dx_fluct'] = dv_dx_fluct
    base[0][0]['dv_dy_fluct'] = dv_dy_fluct
    base[0][0]['dv_dz_fluct'] = dv_dz_fluct
    base[0][0]['dw_dx_fluct'] = dw_dx_fluct
    base[0][0]['dw_dy_fluct'] = dw_dy_fluct
    base[0][0]['dw_dz_fluct'] = dw_dz_fluct
    
    # The turbulence statistics
    base[0][0]['TKE'] = TKE_mean
    base[0][0]['up_mean'] = up_mean
    base[0][0]['vp_mean'] = vp_mean
    base[0][0]['wp_mean'] = wp_mean
    base[0][0]['u_TKE_mean'] = u_TKE_mean
    base[0][0]['v_TKE_mean'] = v_TKE_mean
    base[0][0]['w_TKE_mean'] = w_TKE_mean
    
    # The Reynolds stress components
    base[0][0]['Reynolds_uu'] = uu_mean
    base[0][0]['Reynolds_vv'] = vv_mean
    base[0][0]['Reynolds_ww'] = ww_mean
    base[0][0]['Reynolds_uv'] = uv_mean
    base[0][0]['Reynolds_uw'] = uw_mean
    base[0][0]['Reynolds_vw'] = vw_mean
    base[0][0]['Reynolds_pp'] = pp_mean
    
    # The RMS components
    base[0][0]['TKE_rms'] = TKE_rms
    base[0][0]['u_rms'] = u_rms
    base[0][0]['v_rms'] = v_rms
    base[0][0]['w_rms'] = w_rms
    
    writer = Writer('hdf_antares')
    file_name = 'Intermediate_Turbulence_Stats'
    writer['filename'] = file_name
    writer['base'] = base
    writer['dtype'] = 'float32'
    writer.dump()
    return file_name

def compute_gradients(filename):
    """
    Compute gradients of TKE, pressure-velocity correlations, and velocity-TKE correlations
    with enhanced error handling and comprehensive gradient calculation.
    """
    # =================================================================================
    # Computing gradients and preparing for transport analysis
    # =================================================================================
    print(f'\n{"Calculating the gradients":.^80}\n')
    r = Reader('hdf_antares')
    r['filename'] = 'Intermediate_Turbulence_Stats.h5'
    base = r.read()
    
    try:
        print('Computing gradients of TKE...')
        treatment = Treatment('gradient')
        treatment['base'] = base
        treatment['coordinates'] = ['x', 'y', 'z']
        treatment['variables'] = ['TKE']
        base = treatment.execute()
        base.cell_to_node()
        print('✓ TKE gradients computed successfully')
        
        # Compute all pressure-velocity correlation gradients at once
        print('Computing gradients of pressure-velocity correlations...')
        treatment = Treatment('gradient')
        treatment['base'] = base
        treatment['coordinates'] = ['x', 'y', 'z']
        treatment['variables'] = ['up_mean', 'vp_mean', 'wp_mean']
        base = treatment.execute()
        base.cell_to_node()
        print('✓ Pressure-velocity correlation gradients computed successfully')
        
        # Compute all velocity-TKE correlation gradients at once
        print('Computing gradients of velocity-TKE correlations...')
        treatment = Treatment('gradient')
        treatment['base'] = base
        treatment['coordinates'] = ['x', 'y', 'z']
        treatment['variables'] = ['u_TKE_mean', 'v_TKE_mean', 'w_TKE_mean']
        base = treatment.execute()
        base.cell_to_node()
        print('✓ Velocity-TKE correlation gradients computed successfully')
        
        # Compute second order gradients of TKE (Laplacian components)
        print('Computing second order gradients of TKE...')
        # Check if the first-order gradients exist before computing second-order
        treatment = Treatment('gradient')
        treatment['base'] = base
        treatment['coordinates'] = ['x', 'y', 'z']
        treatment['variables'] = ['grad_TKE_x', 'grad_TKE_y', 'grad_TKE_z']
        base = treatment.execute()
        base.cell_to_node()
        print('✓ Second order TKE gradients computed successfully')
        
    except Exception as e:
        print(f'✗ Error during gradient computation: {e}')
        print('Continuing with available gradients...')
        
        # Try a simpler gradient computation approach
        try:
            print('Attempting simplified gradient computation...')
            treatment = Treatment('gradient')
            treatment['base'] = base
            treatment['coordinates'] = ['x', 'y', 'z']
            treatment['variables'] = ['TKE', 'up_mean', 'vp_mean', 'wp_mean', 'u_TKE_mean', 'v_TKE_mean', 'w_TKE_mean']
            base = treatment.execute()
            base.cell_to_node()
            print('✓ Simplified gradient computation successful')
        except Exception as e2:
            print(f'✗ Simplified gradient computation also failed: {e2}')
            print('Proceeding without gradient fields...')
    
    # =================================================================================
    # OUTPUT INTERMEDIATE RESULTS
    # =================================================================================
    print(f'\n{"WRITING GRADIENT OUTPUT FILES":.^80}\n')
    
    try:
        # Write comprehensive turbulence statistics with gradients
        writer = Writer('hdf_antares')
        writer['filename'] = 'Averaged_Solution_Complete_Turbulence_Stats'
        writer['base'] = base
        writer['dtype'] = 'float32'
        writer.dump()
        
        print('✓ Complete turbulence statistics with gradients written')
        
    except Exception as e:
        print(f'✗ Error writing gradient output: {e}')
    
    print(f'\n{"Gradient computation complete!":.^80}\n')
    
    return base

# Main function to extract the TKE transport statistics
def compute_TKE_transport():
    """
    Calculate all components of the TKE transport equation:
    ∂TKE/∂t + U_j ∂TKE/∂x_j = P - ε - ∂/∂x_j(u'_j TKE' + u'_j p'/ρ) + ν ∇²TKE
    
    Where:
    P = Production term = -u'_i u'_j ∂U_i/∂x_j
    ε = Dissipation term
    U_j ∂TKE/∂x_j = Convection term
    ∂/∂x_j(u'_j TKE') = Turbulent transport
    ∂/∂x_j(u'_j p'/ρ) = Pressure transport  
    ν ∇²TKE = Viscous diffusion
    """
    
    print(f'\n{"CALCULATING TKE TRANSPORT EQUATION COMPONENTS":.^80}\n')
    loc = 'node'  # Location for gradient calculations
    # Read the file with computed gradients and statistics
    r = Reader('hdf_antares')
    r['filename'] = 'Averaged_Solution_Complete_Turbulence_Stats.h5'
    base = r.read()
    base.show()
    # try:
    # =================================================================================
    # 1. TKE PRODUCTION TERMS (P_ij = -u'_i u'_j ∂U_i/∂x_j)
    # =================================================================================
    print('Computing TKE production terms...')
    
    # Production components from Reynolds stress tensor and mean velocity gradients
    base.compute('P_11 = -Reynolds_uu * du_dx')  # -u'u' × ∂U/∂x
    base.compute('P_12 = -Reynolds_uv * du_dy')  # -u'v' × ∂U/∂y  
    base.compute('P_13 = -Reynolds_uw * du_dz')  # -u'w' × ∂U/∂z
    base.compute('P_21 = -Reynolds_uv * dv_dx')  # -u'v' × ∂V/∂x
    base.compute('P_22 = -Reynolds_vv * dv_dy')  # -v'v' × ∂V/∂y
    base.compute('P_23 = -Reynolds_vw * dv_dz')  # -v'w' × ∂V/∂z
    base.compute('P_31 = -Reynolds_uw * dw_dx')  # -u'w' × ∂W/∂x
    base.compute('P_32 = -Reynolds_vw * dw_dy')  # -v'w' × ∂W/∂y
    base.compute('P_33 = -Reynolds_ww * dw_dz')  # -w'w' × ∂W/∂z
    
    # Total TKE Production (sum of all production components from the Reynolds stress tensor and mean velocity gradients)
    base.compute('TKE_Production = P_11 + P_12 + P_13 + P_21 + P_22 + P_23 + P_31 + P_32 + P_33')
    
    # Separate production by velocity components (for detailed analysis)
    base.compute('Production_from_U = P_11 + P_12 + P_13')  # Production from U-gradients
    base.compute('Production_from_V = P_21 + P_22 + P_23')  # Production from V-gradients  
    base.compute('Production_from_W = P_31 + P_32 + P_33')  # Production from W-gradients
    
    # Normal and shear production components
    base.compute('Normal_Production = P_11 + P_22 + P_33')      # Normal stress production
    base.compute('Shear_Production = P_12 + P_13 + P_21 + P_23 + P_31 + P_32')  # Shear stress production
    
    print('✓ TKE production terms computed successfully')
    
    # =================================================================================
    # 2. CONVECTION TERMS (U_j ∂TKE/∂x_j)
    # =================================================================================
    print('Computing convection terms...')
    
    if (('grad_TKE_x',loc) in base[0][0].keys() and 
        ('grad_TKE_y',loc) in base[0][0].keys() and 
        ('grad_TKE_z',loc) in base[0][0].keys()):
        
        base.compute('Convection_x = u * grad_TKE_x', location=loc)  # U × ∂TKE/∂x
        base.compute('Convection_y = v * grad_TKE_y', location=loc)  # V × ∂TKE/∂y
        base.compute('Convection_z = w * grad_TKE_z', location=loc)  # W × ∂TKE/∂z
        base.compute('Total_Convection = Convection_x + Convection_y + Convection_z')
        
        print('✓ Convection terms computed successfully')
    else:
        print('⚠ Warning: TKE gradients not available for convection calculation')
    
    # =================================================================================
    # 3. TURBULENT TRANSPORT DIVERGENCE (∇ · (u'TKE'))
    # =================================================================================
    print('Computing turbulent transport divergence...')
    
    if (('grad_u_TKE_mean_x',loc) in base[0][0].keys() and 
        ('grad_v_TKE_mean_y',loc) in base[0][0].keys() and 
        ('grad_w_TKE_mean_z',loc) in base[0][0].keys()):
        
        base.compute('Turbulent_Transport_Div = grad_u_TKE_mean_x + grad_v_TKE_mean_y + grad_w_TKE_mean_z')
        
        # Individual components for analysis
        base.compute('Turbulent_Transport_x = grad_u_TKE_mean_x', location=loc)
        base.compute('Turbulent_Transport_y = grad_v_TKE_mean_y', location=loc)
        base.compute('Turbulent_Transport_z = grad_w_TKE_mean_z', location=loc)
        
        print('✓ Turbulent transport divergence computed successfully')
    else:
        print('⚠ Warning: Velocity-TKE correlation gradients not available')
    
    # =================================================================================
    # 4. PRESSURE TRANSPORT DIVERGENCE (∇ · (u'p'/ρ))
    # =================================================================================
    print('Computing pressure transport divergence...')
    
    if (('grad_up_mean_x',loc) in base[0][0].keys() and 
        ('grad_vp_mean_y',loc) in base[0][0].keys() and 
        ('grad_wp_mean_z',loc) in base[0][0].keys()):
        
        # Assuming constant density (ρ = 1 for normalized pressure)
        base.compute('Pressure_Transport_Div = grad_up_mean_x + grad_vp_mean_y + grad_wp_mean_z')
        
        # Individual components for analysis
        base.compute('Pressure_Transport_x = grad_up_mean_x', location=loc)
        base.compute('Pressure_Transport_y = grad_vp_mean_y', location=loc)
        base.compute('Pressure_Transport_z = grad_wp_mean_z', location=loc)
        
        print('✓ Pressure transport divergence computed successfully')
    else:
        print('⚠ Warning: Pressure-velocity correlation gradients not available')
    
    # =================================================================================
    # 5. VISCOUS DISSIPATION APPROXIMATION
    # =================================================================================
    print('Computing viscous dissipation approximation...')
    
    if ('du_dx_fluct',loc) in base[0][0].keys():
        # Pseudo-dissipation from fluctuating velocity gradients
        # ε ≈ 2ν ⟨S'_{ij} S'_{ij}⟩ where S'_{ij} is the fluctuating strain rate tensor
        base.compute('Strain_Rate_Fluct_11 = du_dx_fluct', location=loc)
        base.compute('Strain_Rate_Fluct_22 = dv_dy_fluct', location=loc)
        base.compute('Strain_Rate_Fluct_33 = dw_dz_fluct', location=loc)
        base.compute('Strain_Rate_Fluct_12 = 0.5 * (du_dy_fluct + dv_dx_fluct)', location=loc)
        base.compute('Strain_Rate_Fluct_13 = 0.5 * (du_dz_fluct + dw_dx_fluct)', location=loc)
        base.compute('Strain_Rate_Fluct_23 = 0.5 * (dv_dz_fluct + dw_dy_fluct)', location=loc)
        
        # Pseudo-dissipation (without viscosity coefficient)
        base.compute('Pseudo_Dissipation = 2.0 * (Strain_Rate_Fluct_11**2 + Strain_Rate_Fluct_22**2 + Strain_Rate_Fluct_33**2 + ' +
                    '2.0 * (Strain_Rate_Fluct_12**2 + Strain_Rate_Fluct_13**2 + Strain_Rate_Fluct_23**2))')
        
        print('✓ Pseudo-dissipation computed successfully')
    else:
        print('⚠ Warning: Fluctuating gradients not available for dissipation calculation')
    
    # =================================================================================
    # 6. VISCOUS DIFFUSION (ν ∇²TKE)
    # =================================================================================
    print('Computing viscous diffusion term...')
    
    if (('grad_grad_TKE_x_x',loc) in base[0][0].keys() and 
        ('grad_grad_TKE_y_y',loc) in base[0][0].keys() and 
        ('grad_grad_TKE_z_z',loc) in base[0][0].keys()):
        
        base.compute('TKE_Laplacian = grad_grad_TKE_x_x + grad_grad_TKE_y_y + grad_grad_TKE_z_z', location=loc)
        # Viscous diffusion = ν × ∇²TKE (without viscosity coefficient)
        base.compute('Viscous_Diffusion = TKE_Laplacian')
        
        print('✓ Viscous diffusion computed successfully')
    else:
        print('⚠ Warning: Second-order TKE gradients not available for viscous diffusion')
    
    # =================================================================================
    # 7. ADDITIONAL TRANSPORT ANALYSIS QUANTITIES
    # =================================================================================
    print('Computing additional transport analysis quantities...')
    
    # TKE budget residual (if all terms are available)
    # budget_terms = [('TKE_Production',loc), ('Total_Convection',loc), ('Turbulent_Transport_Div',loc), 
    #                 ('Pressure_Transport_Div',loc), ('Pseudo_Dissipation',loc), ('Viscous_Diffusion',loc)]
    
    # available_terms = [term for term in budget_terms if term in base[0][0].keys()]
    
    # if len(available_terms) >= 4:  # Need at least 4 terms for meaningful budget
    #     budget_equation = ' + '.join([f'({term})' if 'Transport' in term or 'Dissipation' in term 
    #                                 else term for term in available_terms])
    #     base.compute(f'TKE_Budget_Residual = {budget_equation}')
    #     print(f'✓ TKE budget computed with {len(available_terms)} terms')
    
    # Anisotropy measures
    base.compute('Anisotropy_b11 = Reynolds_uu / (2.0 * TKE + 0.00000001) - 1/3', location=loc)
    base.compute('Anisotropy_b22 = Reynolds_vv / (2.0 * TKE + 0.00000001) - 1/3', location=loc)
    base.compute('Anisotropy_b33 = Reynolds_ww / (2.0 * TKE + 0.00000001) - 1/3', location=loc)
    base.compute('Anisotropy_b12 = Reynolds_uv / (2.0 * TKE + 0.00000001)', location=loc)
    base.compute('Anisotropy_b13 = Reynolds_uw / (2.0 * TKE + 0.00000001)', location=loc)
    base.compute('Anisotropy_b23 = Reynolds_vw / (2.0 * TKE + 0.00000001)', location=loc)
    
    # Turbulent time scale
    base.compute('Turbulent_Time_Scale = TKE / (abs(Pseudo_Dissipation) + 0.00000001)', location=loc)
    
    # Production-to-dissipation ratio
    base.compute('P_to_Epsilon_Ratio = abs(TKE_Production) / (abs(Pseudo_Dissipation) + 0.00000001)', location=loc)
    
    print('✓ Additional transport quantities computed successfully')
    print(f'✓ TKE transport equation components calculation complete!')

    # except Exception as e:
    #     print(f'✗ Error during TKE transport calculation: {e}')
    #     print('Some transport quantities may not be available')
    #     import traceback
    #     traceback.print_exc()
    
    # =================================================================================
    # 9. WRITE FINAL OUTPUT
    # =================================================================================
    print(f'\n{"WRITING FINAL TKE TRANSPORT RESULTS":.^80}\n')
    
    try:
        # Also update the original complete file
        writer = Writer('hdf_antares')
        writer['filename'] = 'Averaged_Solution_Complete_Turbulence_Stats'
        writer['base'] = base
        writer['dtype'] = 'float32'
        writer.dump()
        
        print('✓ Final TKE transport file written: Averaged_Solution_Complete_Turbulence_Stats.h5')
        
    except Exception as e:
        print(f'Error writing output files: {e}')
    
    print(f'\n{"TKE TRANSPORT ANALYSIS COMPLETE!":.^80}\n')
    
    return base

def main():
    #base, nodes = extract_mesh(meshpath,meshfile)                   # Extract the mesh
    #intermediate_file = Extract_data_TKE(sol_dirName, base, nodes, vars_inst)  # Extract the data from the solution directory
    #compute_gradients(intermediate_file)                            # Compute the gradients and turbulence statistics
    compute_TKE_transport()                                         # Compute TKE transport equation components

if __name__ ==  '__main__':
    main()