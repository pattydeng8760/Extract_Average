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
nstart = 6       # solution dir count to begin the extract
# The path of the source mesh
meshpath = '/scratch/p/plavoie/denggua1/Bombardier_LES/B_10AOA_LES/MESH_ZONE_Nov24/'
#meshpath = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/MESH_ZONE2/'
# The name of the source mesh (IF MESH CONTAINS ZONES, REQUIRE ZONE MERGING A-PRIORI)
meshfile = 'Bombardier_10AOA_Combine_Nov24.mesh.h5'
#meshfile = 'Bombardier_10AOA_Combine2.mesh.h5'
# The path of the average solution directory
ave_dirName = '/scratch/p/plavoie/denggua1/Bombardier_LES/B_10AOA_LES/RUN_ZONE_Nov24/AVE/'
#ave_dirName = '/scratch/p/plavoie/denggua1/Bombardier_LES/B_10AOA_LES/RUN_ZONE_Nov24/SOLUT/'


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

def Extract_data(ave_dirName, base, nodes):    # main function to exctact the mean data and append solutions

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
    
    #looping over all the directories to count the number of time steps
    timestep = 0
    for i in range(nstart,np.shape(sol_dir)[0]):
        dir = os.path.join(arr_dir,arr[i])
        files = sort_files(dir)
        for j in range(0,np.shape(files)[0]):
            timestep+= 1
    
    # initializing the variables for averaging
    field_names = ['P', 'u', 'v', 'w', 'Q', 'vort_x', 'Strain_Rate', 'P_rms', 'u_rms', 'v_rms', 'w_rms', 'TKE']
    field_arrays = [np.zeros((nodes,), dtype=np.float32) for _ in field_names]
    P, u, v, w, Q, vort_x, Strain_Rate, P_rms, u_rms, v_rms, w_rms, TKE = field_arrays
    count = 0
    # looping over all main solver directories
    for i in range(nstart,np.shape(sol_dir)[0]):
        dir = os.path.join(arr_dir,arr[i])
        text = 'Processing the directory: '; print(f'\n{text}{dir}') 
        files = sort_files(dir)
        for j in range(0,np.shape(files)[0]):
            text = 'Iteration: '; print(f'\n{text}{count}') 
            text = 'Reading full solution file '; print(f'{text}:{files[j]}') 
            sol_file = os.path.join(arr_dir,arr[i],files[j]+'.h5') # The full solution file
            r = Reader('hdf_avbp')
            r['base'] = base
            r['filename'] = sol_file
            base = r.read()
            base.compute('P_rms = abs(P2 - P*P)**0.5')
            base.compute('u_rms = abs(u2 - u*u)**0.5')
            base.compute('v_rms = abs(v2 - v*v)**0.5')
            base.compute('w_rms = abs(w2 - w*w)**0.5')
            base.compute('Q = Q1+Q2')
            base.compute('TKE = 0.5*((u2 - u*u) + (v2 - v*v) + (w2 - w*w))')
            #base.compute('Enstrophy=1/2*((dw_dy-dv_dz)**2+(du_dz-dw_dx)**2+(dv_dx-du_dy)**2)')
            #base.compute('Strain=(du_dx)**2+(dv_dy)**2+(dw_dz)**2+1/2*((du_dy+dv_dx)**2+(du_dz+dw_dx)**2+(dv_dz+dw_dy)**2)')
            #base.compute('Pressure_Hessian=rho*(Enstrophy-Strain)')
            base.compute('Strain_Rate=0.5*((du_dx**2 + dv_dy**2 + dw_dz**2 + 2*(du_dy + dv_dx)**2 + 2*(du_dz + dw_dx)**2 + 2*(dv_dz + dw_dy)**2)**0.5)')
            P = Welford_avg(P, base[0][0]['P'], count)
            u = Welford_avg(u, base[0][0]['u'], count)
            v = Welford_avg(v, base[0][0]['v'], count)
            w = Welford_avg(w, base[0][0]['w'], count)
            Q = Welford_avg(Q, base[0][0]['Q'], count)
            P_rms = Welford_avg(P_rms, base[0][0]['P_rms'], count)
            u_rms = Welford_avg(u_rms, base[0][0]['u_rms'], count)
            v_rms = Welford_avg(v_rms, base[0][0]['v_rms'], count)
            w_rms = Welford_avg(w_rms, base[0][0]['w_rms'], count)
            TKE = Welford_avg(TKE, base[0][0]['TKE'], count)
            vort_x = Welford_avg(vort_x, base[0][0]['vort_x'], count)
            Strain_Rate = Welford_avg(Strain_Rate, base[0][0]['Strain_Rate'], count)
            count+=1    
    text = 'Average Field Data'
    print(f'\n{text:.^80}\n')
    base.show()
    
    #outputting the mean field of select variables
    output_base = Base()
    output_base['0'] = Zone()
    output_base[0].shared.connectivity = base[0][0].connectivity
    output_base[0].shared["x"] = base[0][0]["x"]
    output_base[0].shared["y"] = base[0][0]["y"]
    output_base[0].shared["z"] = base[0][0]["z"]
    output_base[0][str(0)] = Instant()
    output_base[0][str(0)]["P"] = P
    output_base[0][str(0)]["u"] = u
    output_base[0][str(0)]["v"] = v
    output_base[0][str(0)]["w"] = w
    output_base[0][str(0)]["Q"] = Q
    output_base[0][str(0)]["vort_x"] = vort_x
    output_base[0][str(0)]["Strain_Rate"] = Strain_Rate
    output_base[0][str(0)]["P_rms"] = P_rms
    output_base[0][str(0)]["u_rms"] = u_rms
    output_base[0][str(0)]["v_rms"] = v_rms
    output_base[0][str(0)]["w_rms"] = w_rms
    output_base[0][str(0)]["TKE"] = TKE
    
    # Extracting the average solution of the domain
    writer = Writer('hdf_antares')
    writer['filename'] = os.path.join('Averaged_Solution_Limited_Ave')
    writer['base'] = output_base
    writer['dtype'] = 'float32'
    writer.dump()
    
    writer = Writer('hdf_antares')
    writer['filename'] = os.path.join('Averaged_Solution')
    writer['base'] = base
    # Overwriting the variables to the actual mean values instead of the first solution
    base[0][0]['P'] = P
    base[0][0]['u'] = u
    base[0][0]['v'] = v
    base[0][0]['w'] = w
    base[0][0]['Q'] = Q
    base[0][0]['vort_x'] = vort_x
    base[0][0]['Strain_Rate'] = Strain_Rate
    base[0][0]['P_rms'] = P_rms
    base[0][0]['u_rms'] = u_rms
    base[0][0]['v_rms'] = v_rms
    base[0][0]['w_rms'] = w_rms
    base[0][0]['TKE'] = TKE
    writer['dtype'] = 'float32'
    writer.dump()
    
    # Extracting the average solution of the airfoil only 
    patch = base[base.families['Patches']]
    # The surface mesh
    writer = Writer('hdf_antares')
    #w['base'] = patch[(bnd_name,)]
    writer['base'] = patch['Airfoil_Surface','Airfoil_Trailing_Edge','Airfoil_Side_LE','Airfoil_Side_Mid','Airfoil_Side_TE']
    writer['dtype'] = 'float32'
    writer['filename'] = os.path.join('Averaged_Solution_Surface')
    writer.dump()
    
    writer = Writer('hdf_antares')
    writer['filename'] = os.path.join('Averaged_Solution_Reduced_Variables')
    base.delete_variables(vars_ave)
    writer['base'] = base
    writer['dtype'] = 'float32'
    writer.dump()
    

    text = 'Output Field Data'
    print(f'\n{text:.^80}\n')
    output_base.show()
    text = 'Calculation Complete!'
    print(f'\n{text:.^80}\n')  


def main():
    base, nodes = extract_mesh(meshpath,meshfile)                   # Extract the mesh
    Extract_data(ave_dirName, base, nodes)                          # Extract the data from the solution directory


if __name__ ==  '__main__':
    main()