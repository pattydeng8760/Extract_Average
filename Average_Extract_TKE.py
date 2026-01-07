########################################################################################################################################################################
# Calculating the average from instantaneous solutions via the Welford Algorithm and Extracting TKE Transport Equation Components
# Author: Patrick Deng
# Enhanced with complete TKE transport equation analysis
# DO NOT MODIFY FUNCTIONS IN THIS FILE
########################################################################################################################################################################
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
#nstart = 37
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


vars_tke = ['gamma_bar', 'hypvis_artif', 'hypvis_artif_y', 'mpi_rank', 'myzone', 'r_bar',
    'ss_bar', 'tau_turb_xy', 'tau_turb_xz', 'tau_turb_yz', 'vis_artif', 'vis_artif_y', 
    'visco_mask', 'wall_EnergyFlux_normal', 'wall_EnergyFlux_x', 'wall_EnergyFlux_y', 
    'wall_EnergyFlux_z', 'zeta_p', 'zeta_y', 'rhoE', 'rhou', 'rhov', 'rhow', 
    'AIR','Q1','Q2','wall_Stress_x','wall_Stress_y','wall_Stress_z','wall_normal_Stress','wall_shear_Stress','wall_yplus',
    'du_dx','du_dy','du_dz','dv_dx','dv_dy','dv_dz','dw_dx','dw_dy','dw_dz',
    'vort_y','vort_z', 'VD_volume', 'temperature', 'vis_turb', 'dv_dy_fluct', 'dw_dz_fluct', 'du_dy_fluct', 'du_dz_fluct', 'dv_dx_fluct', 'dv_dz_fluct', 'dw_dx_fluct', 'dw_dy_fluct',
    'du_dx_fluct', 'up_mean', 'vp_mean', 'wp_mean', 'u_TKE_mean', 'v_TKE_mean', 'w_TKE_mean', 'Reynolds_uu', 'Reynolds_vv', 'Reynolds_ww', 'Reynolds_uv', 'Reynolds_uw','Reynolds_vw', 'Reynolds_pp',
    'grad_TKE_x', 'grad_TKE_y', 'grad_TKE_z', 'cell_volume', 'grad_up_mean_x', 'grad_vp_mean_x', 'grad_wp_mean_x',  'grad_up_mean_y', 'grad_vp_mean_y', 'grad_wp_mean_y', 
    'grad_up_mean_z', 'grad_vp_mean_z', 'grad_wp_mean_z', 'grad_u_TKE_mean_x', 'grad_v_TKE_mean_x', 'grad_w_TKE_mean_x', 
    'grad_u_TKE_mean_y', 'grad_v_TKE_mean_y', 'grad_w_TKE_mean_y', 'grad_u_TKE_mean_z', 'grad_v_TKE_mean_z', 'grad_w_TKE_mean_z',
    'grad_grad_TKE_x_x', 'grad_grad_TKE_x_y', 'grad_grad_TKE_x_z', 'grad_grad_TKE_y_x', 'grad_grad_TKE_y_y',
    'grad_grad_TKE_y_z', 'grad_grad_TKE_z_x', 'grad_grad_TKE_z_y', 'grad_grad_TKE_z_z',
    'P_11', 'P_12', 'P_13', 'P_21', 'P_22', 'P_23', 'P_31', 'P_32', 'P_33']

vars_tke2 = ['gamma_bar', 'hypvis_artif', 'hypvis_artif_y', 'mpi_rank', 'myzone', 'r_bar',
    'ss_bar', 'tau_turb_xy', 'tau_turb_xz', 'tau_turb_yz', 'vis_artif', 'vis_artif_y', 
    'visco_mask', 'wall_EnergyFlux_normal', 'wall_EnergyFlux_x', 'wall_EnergyFlux_y', 
    'wall_EnergyFlux_z', 'zeta_p', 'zeta_y', 'rhoE', 'rhou', 'rhov', 'rhow', 
    'AIR','Q1','Q2','wall_Stress_x','wall_Stress_y','wall_Stress_z','wall_normal_Stress','wall_shear_Stress','wall_yplus',
    'du_dx_mean', 'du_dy_mean', 'du_dz_mean', 'dv_dx_mean', 'dv_dy_mean', 'dv_dz_mean', 'dw_dx_mean', 'dw_dy_mean', 'dw_dz_mean',
    'Strain_dudx2', 'Strain_dvdy2', 'Strain_dwdz2', 'Strain_dudy_dvdx', 'Strain_dudz_dwdx', 'Strain_dvdz_dwdy',
    'TKE_Reynolds_uu', 'TKE_Reynolds_vv', 'TKE_Reynolds_ww', 'Reynolds_pp',
    'grad_Tdx_mean_x', 'grad_Tdx_mean_y', 'grad_Tdx_mean_z', 'grad_Tdy_mean_x', 'grad_Tdy_mean_y', 'grad_Tdy_mean_z','grad_Tdz_mean_x', 'grad_Tdz_mean_y', 'grad_Tdz_mean_z',
    'grad_TKE_Reynolds_uu_x', 'grad_TKE_Reynolds_uu_y', 'grad_TKE_Reynolds_uu_z', 'grad_TKE_Reynolds_vv_x', 'grad_TKE_Reynolds_vv_y', 'grad_TKE_Reynolds_vv_z',
    'grad_TKE_Reynolds_ww_x', 'grad_TKE_Reynolds_ww_y', 'grad_TKE_Reynolds_ww_z', 
    'grad_TKE_x', 'grad_TKE_y', 'grad_TKE_z',
    'grad_Reynolds_uv_x', 'grad_Reynolds_uv_y', 'grad_Reynolds_uv_z', 'grad_Reynolds_uw_x', 'grad_Reynolds_uw_y', 'grad_Reynolds_uw_z',
    'grad_Reynolds_vw_x', 'grad_Reynolds_vw_y', 'grad_Reynolds_vw_z', 'grad_grad_TKE_Reynolds_uu_x_x', 'grad_grad_TKE_Reynolds_vv_y_x', 'grad_grad_TKE_Reynolds_ww_z_x',
    'grad_grad_TKE_Reynolds_uu_x_y', 'grad_grad_TKE_Reynolds_vv_y_y', 'grad_grad_TKE_Reynolds_ww_z_y', 'grad_grad_TKE_Reynolds_uu_x_z', 'grad_grad_TKE_Reynolds_vv_y_z', 'grad_grad_TKE_Reynolds_ww_z_z',
    'grad_grad_Reynolds_uv_x_x', 'grad_grad_Reynolds_vw_y_x', 'grad_grad_Reynolds_uw_z_x', 'grad_grad_Reynolds_uv_x_y', 'grad_grad_Reynolds_vw_y_y', 'grad_grad_Reynolds_uw_z_y',
    'grad_grad_Reynolds_uv_x_z', 'grad_grad_Reynolds_vw_y_z', 'grad_grad_Reynolds_uw_z_z', 
    'Strain_Rate_11', 'Strain_Rate_22', 'Strain_Rate_33', 'Strain_Rate_12', 'Strain_Rate_13', 'Strain_Rate_23','grad_P_x', 'grad_P_y', 'grad_P_z',
    'P_11', 'P_12', 'P_13', 'P_21', 'P_22', 'P_23', 'P_31', 'P_32', 'P_33', 
    'grad_P_x', 'grad_P_y', 'grad_P_z', 'cell_volume', 'VD_volume', 'temperature', 'vis_turb',]


def print(text):
    """ Function to print the text and flush the output"""
    builtins.print(text)
    os.fsync(sys.stdout)
    
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

def Calc_avg(mean, current, count): 
    """
    calculate the mean
    """
    mean = mean + (1/count)*(current-mean)
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
    print(f'Total files to process from list: {total_files}')
    
    total_files = 0
    for sol_file in file_list:
        total_files += 1
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
    vort_x_mean = np.zeros((nodes,), dtype=np.float32)
    TKE = np.zeros((nodes,), dtype=np.float64)
    rho_mean = np.zeros((nodes,), dtype=np.float32)
    
    # Initialize the velocity gradient components
    du_dx_mean = np.zeros((nodes,), dtype=np.float32)
    du_dy_mean = np.zeros((nodes,), dtype=np.float32)
    du_dz_mean = np.zeros((nodes,), dtype=np.float32)
    dv_dx_mean = np.zeros((nodes,), dtype=np.float32)
    dv_dy_mean = np.zeros((nodes,), dtype=np.float32)
    dv_dz_mean = np.zeros((nodes,), dtype=np.float32)
    dw_dx_mean = np.zeros((nodes,), dtype=np.float32)
    dw_dy_mean = np.zeros((nodes,), dtype=np.float32)
    dw_dz_mean = np.zeros((nodes,), dtype=np.float32)
    
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
        rho_inst = base[0][0]['rho']
        
        # Update means 
        P_mean = Calc_avg(P_mean, P_inst, count+1)
        u_mean = Calc_avg(u_mean, u_inst, count+1)
        v_mean = Calc_avg(v_mean, v_inst, count+1)
        w_mean = Calc_avg(w_mean, w_inst, count+1)
        rho_mean = Calc_avg(rho_mean, rho_inst, count+1)
        vort_x_mean = Calc_avg(vort_x_mean, base[0][0]['vort_x'], count+1)
        # The mean velocity gradients
        du_dx_mean = Calc_avg(du_dx_mean, base[0][0]['du_dx'], count+1)
        du_dy_mean = Calc_avg(du_dy_mean, base[0][0]['du_dy'], count+1)
        du_dz_mean = Calc_avg(du_dz_mean, base[0][0]['du_dz'], count+1)
        dv_dx_mean = Calc_avg(dv_dx_mean, base[0][0]['dv_dx'], count+1)
        dv_dy_mean = Calc_avg(dv_dy_mean, base[0][0]['dv_dy'], count+1)
        dv_dz_mean = Calc_avg(dv_dz_mean, base[0][0]['dv_dz'], count+1)
        dw_dx_mean = Calc_avg(dw_dx_mean, base[0][0]['dw_dx'], count+1)
        dw_dy_mean = Calc_avg(dw_dy_mean, base[0][0]['dw_dy'], count+1)
        dw_dz_mean = Calc_avg(dw_dz_mean, base[0][0]['dw_dz'], count+1)
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
        TKE = Calc_avg(TKE, TKE_inst, count+1)
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
    
    # Initialize the turbulent diffusion terms
    Tdx_mean = np.zeros((nodes,), dtype=np.float64)
    Tdy_mean = np.zeros((nodes,), dtype=np.float64)
    Tdz_mean = np.zeros((nodes,), dtype=np.float64)

    # Initialize the fluctuating gradients for dissipation
    Strain_dudx2 = np.zeros((nodes,), dtype=np.float64)
    Strain_dvdy2 = np.zeros((nodes,), dtype=np.float64)
    Strain_dwdz2 = np.zeros((nodes,), dtype=np.float64)
    Strain_dudy_dvdx = np.zeros((nodes,), dtype=np.float64)
    Strain_dudz_dwdx = np.zeros((nodes,), dtype=np.float64)
    Strain_dvdz_dwdy = np.zeros((nodes,), dtype=np.float64)
    
    # Initialize TKE fluctuation statistics
    TKE_TKE_mean = np.zeros((nodes,), dtype=np.float64)  # TKE variance (TKE')^2
    
    # Initialize the pressure dilation terms
    pressure_dilation = np.zeros((nodes,), dtype=np.float64)
    
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
        rho_inst = base[0][0]['rho']
        du_dx_inst = base[0][0]['du_dx'] - du_dx_mean
        du_dy_inst = base[0][0]['du_dy'] - du_dy_mean
        du_dz_inst = base[0][0]['du_dz'] - du_dz_mean
        dv_dx_inst = base[0][0]['dv_dx'] - dv_dx_mean
        dv_dy_inst = base[0][0]['dv_dy'] - dv_dy_mean
        dv_dz_inst = base[0][0]['dv_dz'] - dv_dz_mean
        dw_dx_inst = base[0][0]['dw_dx'] - dw_dx_mean
        dw_dy_inst = base[0][0]['dw_dy'] - dw_dy_mean
        dw_dz_inst = base[0][0]['dw_dz'] - dw_dz_mean
        
        # Calculate fluctuations relative to final means
        u_prime = u_inst - u_mean
        v_prime = v_inst - v_mean
        w_prime = w_inst - w_mean
        p_prime = P_inst - P_mean
        
        # Calculate instantaneous TKE from fluctuating velocities (q^2)
        TKE_inst = 0.5 * (u_prime**2 + v_prime**2 + w_prime**2)
        
        # Update Reynolds stress components
        uu_mean = Calc_avg(uu_mean, u_prime * u_prime, count+1)
        vv_mean = Calc_avg(vv_mean, v_prime * v_prime, count+1)
        ww_mean = Calc_avg(ww_mean, w_prime * w_prime, count+1)
        uv_mean = Calc_avg(uv_mean, u_prime * v_prime, count+1)
        uw_mean = Calc_avg(uw_mean, u_prime * w_prime, count+1)
        vw_mean = Calc_avg(vw_mean, v_prime * w_prime, count+1)
        pp_mean = Calc_avg(pp_mean, p_prime * p_prime, count+1)
        
        # Update pressure-velocity correlations = div(u'_j (p'+ rho*TKE'))
        Tdx_mean = Calc_avg(Tdx_mean, u_prime * (p_prime + rho_inst*TKE_inst), count+1)
        Tdy_mean = Calc_avg(Tdy_mean, v_prime * (p_prime + rho_inst*TKE_inst), count+1)
        Tdz_mean = Calc_avg(Tdz_mean, w_prime * (p_prime + rho_inst*TKE_inst), count+1)
        
        # Update fluctuating gradients for dissipation
        Strain_dudx2 = Calc_avg(Strain_dudx2, du_dx_inst * du_dx_inst, count+1)
        Strain_dvdy2 = Calc_avg(Strain_dvdy2, dv_dy_inst * dv_dy_inst, count+1)
        Strain_dwdz2 = Calc_avg(Strain_dwdz2, dw_dz_inst * dw_dz_inst, count+1)
        Strain_dudy_dvdx = Calc_avg(Strain_dudy_dvdx, (du_dy_inst + dv_dx_inst)**2, count+1)
        Strain_dudz_dwdx = Calc_avg(Strain_dudz_dwdx, (du_dz_inst + dw_dx_inst)**2, count+1)
        Strain_dvdz_dwdy = Calc_avg(Strain_dvdz_dwdy, (dv_dz_inst + dw_dy_inst)**2, count+1)
        
        # pressure dilation term
        pressure_dilation = Calc_avg(pressure_dilation, p_prime * (du_dx_inst + dv_dy_inst + dw_dz_inst), count+1)
        
        count += 1
    
    # Print summary statistics
    print(f'Third pass complete. Turbulence statistics calculated from {count} files.')
    
    # =================================================================================
    # Outputting the averaged results
    # =================================================================================
    r = Reader('hdf_avbp')
    r['base'] = base
    r['filename'] = file_list[0]        # Use the first file as a surrogate for structure
    base = r.read()
    base.delete_variables(vars_delete)
    
    # The mean flow quantities
    base[0][0]['P'] = P_mean
    base[0][0]['u'] = u_mean
    base[0][0]['v'] = v_mean
    base[0][0]['w'] = w_mean
    base[0][0]['TKE'] = TKE
    base[0][0]['rho'] = rho_mean
    base[0][0]['vort_x'] = vort_x_mean
    
    # The mean flow gradients
    base[0][0]['du_dx_mean'] = du_dx_mean
    base[0][0]['du_dy_mean'] = du_dy_mean
    base[0][0]['du_dz_mean'] = du_dz_mean
    base[0][0]['dv_dx_mean'] = dv_dx_mean
    base[0][0]['dv_dy_mean'] = dv_dy_mean
    base[0][0]['dv_dz_mean'] = dv_dz_mean
    base[0][0]['dw_dx_mean'] = dw_dx_mean
    base[0][0]['dw_dy_mean'] = dw_dy_mean
    base[0][0]['dw_dz_mean'] = dw_dz_mean
    
    # The Turbulent Diffusion terms
    base[0][0]['Tdx_mean'] = Tdx_mean
    base[0][0]['Tdy_mean'] = Tdy_mean
    base[0][0]['Tdz_mean'] = Tdz_mean
    
    # The Reynolds stress components
    base[0][0]['Reynolds_uu'] = uu_mean
    base[0][0]['Reynolds_vv'] = vv_mean
    base[0][0]['Reynolds_ww'] = ww_mean
    base[0][0]['Reynolds_uv'] = uv_mean
    base[0][0]['Reynolds_uw'] = uw_mean
    base[0][0]['Reynolds_vw'] = vw_mean
    base[0][0]['Reynolds_pp'] = pp_mean
    
    # The Dissipation terms
    base[0][0]['Strain_dudx2'] = Strain_dudx2
    base[0][0]['Strain_dvdy2'] = Strain_dvdy2
    base[0][0]['Strain_dwdz2'] = Strain_dwdz2
    base[0][0]['Strain_dudy_dvdx'] = Strain_dudy_dvdx
    base[0][0]['Strain_dudz_dwdx'] = Strain_dudz_dwdx
    base[0][0]['Strain_dvdz_dwdy'] = Strain_dvdz_dwdy
    
    
    # The Diffusion terms pre-gradient
    base[0][0]['TKE_Reynolds_uu'] = uu_mean + TKE
    base[0][0]['TKE_Reynolds_vv'] = vv_mean + TKE
    base[0][0]['TKE_Reynolds_ww'] = ww_mean + TKE
    
    # The pressure dilation term
    base[0][0]['pressure_dilation'] = 1/rho_mean*pressure_dilation
    
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
        treatment['variables'] = ['TKE', 'P']
        base = treatment.execute()
        base.cell_to_node()
        print('✓ TKE gradients computed successfully')
        
        # Compute all pressure-velocity correlation gradients at once
        print('Computing gradients of pressure/TKE-velocity correlations for turbulent diffusion...')
        treatment = Treatment('gradient')
        treatment['base'] = base
        treatment['coordinates'] = ['x', 'y', 'z']
        treatment['variables'] = ['Tdx_mean', 'Tdy_mean', 'Tdz_mean']
        base = treatment.execute()
        base.cell_to_node()
        print('✓ Pressure-velocity correlation gradients computed successfully')
        
        # Compute The first order gradient for TKE-Reynolds Stress flucuations
        print('Computing first order gradients of TKE-Reynold Stress for Diffuision...')
        treatment = Treatment('gradient')
        treatment['base'] = base
        treatment['coordinates'] = ['x', 'y', 'z']
        treatment['variables'] = ['TKE_Reynolds_uu', 'TKE_Reynolds_vv', 'TKE_Reynolds_ww', 'Reynolds_uv', 'Reynolds_uw', 'Reynolds_vw']
        base = treatment.execute()
        base.cell_to_node()
        print('✓ TKE-Reynolds stress gradients computed successfully')
        
        # Compute the second order gradients of TKE-Reynolds stress fluctuations
        print('Computing second order gradients of TKE-Reynolds stress for Diffusion...')
        treatment = Treatment('gradient')
        treatment['base'] = base
        treatment['coordinates'] = ['x', 'y', 'z']
        treatment['variables'] = ['grad_TKE_Reynolds_uu_x', 'grad_TKE_Reynolds_vv_y', 'grad_TKE_Reynolds_ww_z',
                                  'grad_Reynolds_uv_x', 'grad_Reynolds_vw_y', 'grad_Reynolds_uw_z']
        base = treatment.execute()
        base.cell_to_node()
        print('✓ Second order TKE-Reynolds stress gradients computed successfully')
        
                
    except Exception as e:
        print(f'✗ Error during gradient computation: {e}')
        print('Continuing with available gradients...')
        
        # Try a simpler gradient computation approach
        try:
            print('Attempting simplified gradient computation...')
            treatment = Treatment('gradient')
            treatment['base'] = base
            treatment['coordinates'] = ['x', 'y', 'z']
            treatment['variables'] = ['TKE', 'Tdx_mean', 'Tdy_mean', 'Tdz_mean',
                                      'TKE_Reynolds_uu', 'TKE_Reynolds_vv', 'TKE_Reynolds_ww',
                                      'Reynolds_uv', 'Reynolds_uw', 'Reynolds_vw']
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
def compute_TKE_transport(vars_del):
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
    if (('Reynolds_uu', loc) in base[0][0].keys() and 
        ('Reynolds_vv', loc) in base[0][0].keys() and
        ('Reynolds_ww', loc) in base[0][0].keys()):
        
        base.compute('P_11 = -Reynolds_uu * du_dx_mean', location=loc)  # P_11 = -uu ∂U/∂x
        base.compute('P_12 = -Reynolds_uv * du_dy_mean', location=loc)  # P_12 = -uv ∂U/∂y
        base.compute('P_13 = -Reynolds_uw * du_dz_mean', location=loc)  # P_13 = -uw ∂U/∂z
        base.compute('P_21 = -Reynolds_uv * dv_dx_mean', location=loc)  # P_21 = -uv ∂V/∂x
        base.compute('P_22 = -Reynolds_vv * dv_dy_mean', location=loc)  # P_22 = -vv ∂V/∂y
        base.compute('P_23 = -Reynolds_vw * dv_dz_mean', location=loc)  # P_23 = -vw ∂V/∂z
        base.compute('P_31 = -Reynolds_uw * dw_dx_mean', location=loc)  # P_31 = -uw ∂W/∂x
        base.compute('P_32 = -Reynolds_vw * dw_dy_mean', location=loc)  # P_32 = -vw ∂W/∂y
        base.compute('P_33 = -Reynolds_ww * dw_dz_mean', location=loc)  # P_33 = -ww ∂W/∂z
        
        # Separate production by velocity components (for detailed analysis)
        base.compute('Production_from_U = rho * (P_11 + P_12 + P_13)')  # Production from U-gradients
        base.compute('Production_from_V = rho * (P_21 + P_22 + P_23)')  # Production from V-gradients  
        base.compute('Production_from_W = rho * (P_31 + P_32 + P_33)')  # Production from W-gradients
        
        # Separate mean gradients for detailed analysis
        base.compute('grad_U' + ' = du_dx_mean  + du_dy_mean + du_dz_mean', location=loc)  # Mean gradient of U
        base.compute('grad_V' + ' = dv_dx_mean  + dv_dy_mean + dv_dz_mean', location=loc)  # Mean gradient of V
        base.compute('grad_W' + ' = dw_dx_mean  + dw_dy_mean + dw_dz_mean', location=loc)  # Mean gradient of W
        
        # Normal and shear production components
        base.compute('Normal_Production = rho * (P_11 + P_22 + P_33)')      # Normal stress production
        base.compute('Shear_Production = rho * (P_12 + P_13 + P_21 + P_23 + P_31 + P_32)')  # Shear stress production
        
        # Total TKE Production (sum of all production components from the Reynolds stress tensor and mean velocity gradients)
        base.compute('TKE_Production = rho * (P_11 + P_12 + P_13 + P_21 + P_22 + P_23 + P_31 + P_32 + P_33)')
        print('✓ TKE production terms computed successfully')
    else:
        print('⚠ Warning: Reynolds stress components not available for TKE production calculation')
    
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
        base.compute('TKE_Convection = rho * (Convection_x + Convection_y + Convection_z)')
        
        print('✓ Convection terms computed successfully')
    else:
        print('⚠ Warning: TKE gradients not available for convection calculation')
    
    # =================================================================================
    # 3. TURBULENT DIFFUSION DIVERGENCE (∇ · (u'_j TKE'))
    # =================================================================================
    print('Computing turbulent transport divergence...')
    
    if (('grad_Tdx_mean_x',loc) in base[0][0].keys() and 
        ('grad_Tdy_mean_y',loc) in base[0][0].keys() and 
        ('grad_Tdz_mean_z',loc) in base[0][0].keys()):
        
        base.compute('TKE_Turbulent_Diffusion = grad_Tdx_mean_x + grad_Tdy_mean_y + grad_Tdz_mean_z')
        
        # Individual components for analysis
        base.compute('Turbulent_Transport_x = grad_Tdx_mean_x', location=loc)
        base.compute('Turbulent_Transport_y = grad_Tdy_mean_y', location=loc)
        base.compute('Turbulent_Transport_z = grad_Tdz_mean_z', location=loc)
        
        print('✓ Turbulent transport divergence computed successfully')
    else:
        print('⚠ Warning: Velocity-TKE correlation gradients not available')
    
    # =================================================================================
    # 4. VISCOUS DIFFUSION APPROXIMATION (ν ∇²TKE)
    # =================================================================================
    print('Computing viscous diffusion approximation...')
    if (('grad_grad_TKE_Reynolds_uu_x_x',loc) in base[0][0].keys() and
        ('grad_grad_TKE_Reynolds_vv_y_y',loc) in base[0][0].keys() and
        ('grad_grad_TKE_Reynolds_ww_z_z',loc) in base[0][0].keys() and
        ('grad_grad_Reynolds_uv_x_y',loc) in base[0][0].keys() and
        ('grad_grad_Reynolds_vw_y_z',loc) in base[0][0].keys() and
        ('grad_grad_Reynolds_uw_z_x',loc) in base[0][0].keys()):
        nu = 1.46e-5  # Kinematic viscosity of air
        # Viscous diffusion approximation using second-order gradients
        base.compute('TKE_Laplacian = grad_grad_TKE_Reynolds_uu_x_x + grad_grad_TKE_Reynolds_vv_y_y + grad_grad_TKE_Reynolds_ww_z_z', location=loc)
        base.compute('TKE_cross = 2 * (grad_grad_Reynolds_uv_x_y + grad_grad_Reynolds_vw_y_z + grad_grad_Reynolds_uw_z_x)', location=loc)
        base.compute('TKE_Viscous_Diffusion = 0.0000146 * (TKE_Laplacian + TKE_cross)', location=loc)
        print('✓ Viscous diffusion approximation computed successfully')
    else:
        print('⚠ Warning: Second-order TKE-Reynolds stress gradients not available for viscous diffusion approximation')
    # =================================================================================
    # 5. VISCOUS DISSIPATION APPROXIMATION (ε) 
    # =================================================================================
    print('Computing dissipation...')
    if (('Strain_dudx2',loc) in base[0][0].keys() and
        ('Strain_dvdy2',loc) in base[0][0].keys() and
        ('Strain_dwdz2',loc) in base[0][0].keys() and
        ('Strain_dudy_dvdx',loc) in base[0][0].keys() and
        ('Strain_dudz_dwdx',loc) in base[0][0].keys() and
        ('Strain_dvdz_dwdy',loc) in base[0][0].keys()):
        # Dissipation approximation using fluctuating strain rates
        nu = 1.46e-5  # Kinematic viscosity of air
        # ε ≈ 2ν ⟨S_{ij} S_{ij}⟩ where S_{ij} is the strain rate tensor
        base.compute('Strain_Rate_11 = 2*Strain_dudx2', location=loc)
        base.compute('Strain_Rate_22 = 2*Strain_dvdy2', location=loc)
        base.compute('Strain_Rate_33 = 2*Strain_dwdz2', location=loc)
        base.compute('Strain_Rate_12 = Strain_dudy_dvdx', location=loc)
        base.compute('Strain_Rate_13 = Strain_dudz_dwdx', location=loc)
        base.compute('Strain_Rate_23 = Strain_dvdz_dwdy', location=loc)
        # Pseudo-dissipation (without viscosity coefficient)
        base.compute('TKE_Dissipation = 0.0000146 * (Strain_Rate_11 + Strain_Rate_22 + Strain_Rate_33 + Strain_Rate_12 + Strain_Rate_13 + Strain_Rate_23)', location=loc)
    else:
        print('⚠ Warning: Fluctuating strain rates not available for dissipation calculation')
    # =================================================================================
    # 6. PRESSURE DILATION TERM (∂/∂x_j(u'_j p'/ρ))
    # =================================================================================
    if ('pressure_dilation', loc) in base[0][0].keys():
        print('Computing pressure dilation term...')
        # Pressure dilation term
        base.compute('Pressure_Dilation = pressure_dilation', location=loc)
        print('✓ Pressure dilation term computed successfully')
    else:
        print('⚠ Warning: Pressure dilation term not available')
    
    # =================================================================================
    # 7. PRESSURE WORK TERM (∂/∂x_j(u'_j p'/ρ))
    # =================================================================================
    if ('grad_P_x', loc) in base[0][0].keys() and \
        ('grad_P_y', loc) in base[0][0].keys() and \
        ('grad_P_z', loc) in base[0][0].keys():
        print('Computing pressure work term...')
        # Pressure work term
        base.compute('Pressure_Work = 1/rho*(grad_P_x * u + grad_P_y * v + grad_P_z * w)', location=loc)
        print('✓ Pressure work term computed successfully')
    else: 
        print('⚠ Warning: Pressure work term gradients not available')
    
    # =================================================================================
    # 8. FINAL TKE TRANSPORT EQUATION
    # =================================================================================
    
    print(f'\n{"WRITING FINAL TKE TRANSPORT RESULTS":.^80}\n')
    base.delete_variables(vars_del)
    print(vars_del)
    try:
        # Also update the original complete file
        writer = Writer('hdf_antares')
        writer['filename'] = 'Final_TKE_Transport_Complete'
        writer['base'] = base
        writer['dtype'] = 'float32'
        writer.dump()
        
        print('✓ Final TKE transport file written: Final_TKE_Transport_Complete.h5')
        
    except Exception as e:
        print(f'Error writing output files: {e}')
    
    print(f'\n{"TKE TRANSPORT ANALYSIS COMPLETE!":.^80}\n')
    
    return base

def main():
    # The following lines syncs the print function and sync the flushed file also on the operating system side
    sys.stdout = open(os.path.join('log_Compute_TKE_Transport.txt'), "w", buffering=1)
    base, nodes = extract_mesh(meshpath,meshfile)                   # Extract the mesh
    intermediate_file = Extract_data_TKE(sol_dirName, base, nodes, vars_inst)  # Extract the data from the solution directory
    compute_gradients(intermediate_file)                            # Compute the gradients and turbulence statistics
    compute_TKE_transport(vars_tke2)                                         # Compute TKE transport equation components

if __name__ ==  '__main__':
    main()