####################################################################################
# Appending mean data from AVE files and compute mean quantities from the instaneous field only
# Author: Patrick Deng
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
nstart = 37      # solution dir count to begin the extract
# The path of the source mesh
meshpath = '/project/p/plavoie/denggua1/BBDB_10AOA/MESH_ZONE_Apr24/'
# The name of the source mesh (IF MESH CONTAINS ZONES, REQUIRE ZONE MERGING A-PRIORI)
meshfile = 'Bombardier_10AOA_Combine_Apr24.mesh.h5'
# The path of the average solution directory
sol_dirName = '/project/p/plavoie/denggua1/BBDB_10AOA/RUN_ZONE_Apr24/SOLUT/'


# Variables to delete
vars_ave = ['gamma_bar', 'hypvis_artif', 'hypvis_artif_y', 'mpi_rank', 'myzone', 'r_bar', 
        'ss_bar', 'tau_turb_xy', 'tau_turb_xz', 'tau_turb_yz', 'vis_artif', 'vis_artif_y', 
        'visco_mask', 'wall_EnergyFlux_normal', 'wall_EnergyFlux_x', 'wall_EnergyFlux_y', 
        'wall_EnergyFlux_z', 'zeta_p', 'zeta_y', 'rho', 'rhoE', 'rhou', 'rhov', 'rhow', 
        'AIR','Q1','Q2','Enstrophy','Strain','Pressure_Hessian','Strain_Rate','du_dx','du_dy','du_dz','dv_dx','dv_dy','dv_dz','dw_dx','dw_dy','dw_dz',
        'vort_y','vort_z','div']

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
    base.show()
    return base, nodes

def Welford_avg(mean, current, iter):    # Welford's algorithm for calculating the mean and variance
    """
    Welford's algorithm for calculating the mean
    """
    mean = mean + (1/iter)*(current-mean) if iter > 0 else current
    return mean

def Extract_data_TKE(ave_dirName, base, nodes):
    """
    Two-pass method for accurate turbulence statistics including velocity-TKE correlations:
    Pass 1: Calculate mean velocities, pressure, and mean TKE
    Pass 2: Calculate all fluctuating quantities and correlations
    """
    
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
    P_mean = np.zeros((nodes,), dtype=np.float64)
    u_mean = np.zeros((nodes,), dtype=np.float64)
    v_mean = np.zeros((nodes,), dtype=np.float64)
    w_mean = np.zeros((nodes,), dtype=np.float64)
    Q_mean = np.zeros((nodes,), dtype=np.float64)
    TKE_mean = np.zeros((nodes,), dtype=np.float64)
    
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
        count += 1
    
    print(f'First pass complete. Mean values calculated from {count} files.')
    
    # =================================================================================
    # SECOND PASS: Calculate fluctuating quantities and correlations
    # =================================================================================
    print(f'\n{"SECOND PASS: Calculating TKE":.^80}\n')
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
        # Calculate instantaneous TKE from current velocity fluctuations
        # Note: This is an approximation for first pass
        TKE_inst = 0.5 * ((u_inst - u_mean)**2 + (v_inst - v_mean)**2 + (w_inst - w_mean)**2)
        TKE_mean = Welford_avg(TKE_mean, TKE_inst, count)
        count += 1
    
    print(f'Second pass complete. Mean values calculated from {count} files.')
    
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
    
    # Initialize TKE fluctuation statistics
    TKE_TKE_mean = np.zeros((nodes,), dtype=np.float64)  # TKE variance (TKE')^2
    
    count = 0
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
    # Computing the Gradients
    # =================================================================================
    r = Reader('hdf_avbp')
    r['base'] = base
    r['filename'] = file_list[0]
    base = r.read()
    base[0][0]['TKE'] = TKE_mean
    base[0][0]['up_mean'] = up_mean
    base[0][0]['vp_mean'] = vp_mean
    base[0][0]['wp_mean'] = wp_mean
    base[0][0]['u_TKE_mean'] = u_TKE_mean
    base[0][0]['v_TKE_mean'] = v_TKE_mean
    base[0][0]['w_TKE_mean'] = w_TKE_mean
    
    # The gradient of TKE
    treatment = Treatment('gradient')
    treatment['base'] = base
    treatment['coordinates'] = ['x', 'y', 'z']
    treatment['variables'] = ['TKE']
    base = treatment.execute()
    
    # The gradient of velocity - pressure correlations
    treatment = Treatment('gradient')
    treatment['base'] = base
    treatment['coordinates'] = ['x']
    treatment['variables'] = ['up_mean', 'u_TKE_mean']
    base = treatment.execute()
    
    # The gradient of velocity - pressure correlations
    treatment = Treatment('gradient')
    treatment['base'] = base
    treatment['coordinates'] = ['y']
    treatment['variables'] = ['vp_mean', 'v_TKE_mean']
    base = treatment.execute()
    
    treatment = Treatment('gradient')
    treatment['base'] = base
    treatment['coordinates'] = ['z']
    treatment['variables'] = ['wp_mean', 'w_TKE_mean']
    base = treatment.execute()
    
    treatment = Treatment('gradient')
    treatment['base'] = base
    treatment['coordinates'] = ['x','y','z']
    treatment['variables'] = ['grad_TKE_x', 'grad_TKE_y','grad_TKE_z']

    base = treatment.execute()

    # =================================================================================
    # OUTPUT RESULTS
    # =================================================================================
    print(f'\n{"WRITING OUTPUT FILES":.^80}\n')
    
    # Create output base with all quantities
    output_base = Base()
    output_base['0'] = Zone()
    output_base[0].shared.connectivity = base[0][0].connectivity
    output_base[0].shared["x"] = base[0][0]["x"]
    output_base[0].shared["y"] = base[0][0]["y"]
    output_base[0].shared["z"] = base[0][0]["z"]
    output_base[0][str(0)] = Instant()
    
    # Mean flow quantities
    output_base[0][str(0)]["P_mean"] = P_mean.astype(np.float32)
    output_base[0][str(0)]["u_mean"] = u_mean.astype(np.float32)
    output_base[0][str(0)]["v_mean"] = v_mean.astype(np.float32)
    output_base[0][str(0)]["w_mean"] = w_mean.astype(np.float32)
    output_base[0][str(0)]["Q_mean"] = Q_mean.astype(np.float32)
    
    # Turbulence quantities
    output_base[0][str(0)]["TKE"] = TKE_mean.astype(np.float32)
    output_base[0][str(0)]["TKE_from_Reynolds"] = TKE_final.astype(np.float32)
    output_base[0][str(0)]["TKE_rms"] = TKE_rms.astype(np.float32)
    output_base[0][str(0)]["u_rms"] = u_rms.astype(np.float32)
    output_base[0][str(0)]["v_rms"] = v_rms.astype(np.float32)
    output_base[0][str(0)]["w_rms"] = w_rms.astype(np.float32)
    output_base[0][str(0)]["P_rms"] = P_rms.astype(np.float32)
    
    # Reynolds stress components
    output_base[0][str(0)]["Reynolds_uu"] = uu_mean.astype(np.float32)
    output_base[0][str(0)]["Reynolds_vv"] = vv_mean.astype(np.float32)
    output_base[0][str(0)]["Reynolds_ww"] = ww_mean.astype(np.float32)
    output_base[0][str(0)]["Reynolds_uv"] = uv_mean.astype(np.float32)
    output_base[0][str(0)]["Reynolds_uw"] = uw_mean.astype(np.float32)
    output_base[0][str(0)]["Reynolds_vw"] = vw_mean.astype(np.float32)
    output_base[0][str(0)]["Reynolds_pp"] = pp_mean.astype(np.float32)
    
    # Pressure-velocity correlations
    output_base[0][str(0)]["up_correlation"] = up_mean.astype(np.float32)
    output_base[0][str(0)]["vp_correlation"] = vp_mean.astype(np.float32)
    output_base[0][str(0)]["wp_correlation"] = wp_mean.astype(np.float32)
    
    # Velocity-TKE correlations
    output_base[0][str(0)]["u_TKE_correlation"] = u_TKE_mean.astype(np.float32)
    output_base[0][str(0)]["v_TKE_correlation"] = v_TKE_mean.astype(np.float32)
    output_base[0][str(0)]["w_TKE_correlation"] = w_TKE_mean.astype(np.float32)
    
    # Write comprehensive turbulence statistics
    writer = Writer('hdf_antares')
    writer['filename'] = os.path.join('Averaged_Solution_Complete_Turbulence_Stats')
    writer['base'] = output_base
    writer['dtype'] = 'float32'
    writer.dump()
    
    # Also update the original base for compatibility with existing code
    base[0][0]['P'] = P_mean.astype(np.float32)
    base[0][0]['u'] = u_mean.astype(np.float32)
    base[0][0]['v'] = v_mean.astype(np.float32)
    base[0][0]['w'] = w_mean.astype(np.float32)
    base[0][0]['Q'] = Q_mean.astype(np.float32)
    base[0][0]['TKE'] = TKE_mean.astype(np.float32)
    base[0][0]['u_rms'] = u_rms.astype(np.float32)
    base[0][0]['v_rms'] = v_rms.astype(np.float32)
    base[0][0]['w_rms'] = w_rms.astype(np.float32)
    base[0][0]['P_rms'] = P_rms.astype(np.float32)
    
    # Write standard output (for compatibility)
    writer = Writer('hdf_antares')
    writer['filename'] = os.path.join('Averaged_Solution')
    writer['base'] = base
    writer['dtype'] = 'float32'
    writer.dump()
    
    # Extract surface data
    patch = base[base.families['Patches']]
    writer = Writer('hdf_antares')
    writer['base'] = patch['Airfoil_Surface','Airfoil_Trailing_Edge','Airfoil_Side_LE','Airfoil_Side_Mid','Airfoil_Side_TE']
    writer['dtype'] = 'float32'
    writer['filename'] = os.path.join('Averaged_Solution_Surface')
    writer.dump()
    
    # Write reduced variables version
    writer = Writer('hdf_antares')
    writer['filename'] = os.path.join('Averaged_Solution_Reduced_Variables')
    base.delete_variables(vars_ave)
    writer['base'] = base
    writer['dtype'] = 'float32'
    writer.dump()
    
    text = 'Complete Turbulence Statistics Calculation Finished!'
    print(f'\n{text:.^80}\n')
    
    return output_base


def main():
    base, nodes = extract_mesh(meshpath,meshfile)                   # Extract the mesh
    Extract_data_TKE(sol_dirName, base, nodes)                          # Extract the data from the solution directory


if __name__ ==  '__main__':
    main()