####################################################################################
# Appending mean data from AVE files and compute mean quantities from the instaneous field only
# Extracting the surface data from the AVE files for cp and cf distribution
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
from datetime import datetime
import matplotlib.pyplot as plt

## The requried input values
nstart = 6       # solution count to begin the extract
Uinf = 16       # The free stream velocity
alpha = 8       # The angle of attack
# The path of the source mesh
meshpath = '/home/p/plavoie/denggua1/scratch/CD_RimeIce_LES/MESH/'
#meshpath = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/MESH_ZONE2/'
# The name of the source mesh (IF MESH CONTAINS ZONES, REQUIRE ZONE MERGING A-PRIORI)
meshfile = 'CD_Airfoil_Combine_Feb25.mesh.h5'
#meshfile = 'Bombardier_10AOA_Combine2.mesh.h5'
# The path of the average solution directory
#ave_dirName = '/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/RUN_ZONE/AVE/'
ave_dirName = '/home/p/plavoie/denggua1/scratch/CD_RimeIce_LES/RUN/SOLUT/'
sol_dirName = '/home/p/plavoie/denggua1/scratch/CD_RimeIce_LES/RUN/SOLUT/13/CD_RimeIce_LES.sol_04060000.h5'
# Function to normalize the airfoil coordinate
def Normalize(coordinate: np.array):
    if coordinate is None:
        raise ValueError("The coordinate is empty")
    normalize = (coordinate - np.min(coordinate))/(np.max(coordinate) -  np.min(coordinate))
    return normalize


# Function to take out the unique file names in the folder
def sort_files(rand):  # Sorting the files in the subdirectory and filtering non-solution data and lists
    rand_sub = os.listdir(rand)
    rand_arr = np.array([]) 
    for i in range(0,np.shape(rand_sub)[0]):
        file_split = os.path.splitext(rand_sub[i])[0]
        rand_arr = np.append(rand_arr,file_split)
    rand_arr = [*set(rand_arr)] # removing the duplicates
    rand_arr = [ rand_arr for rand_arr in rand_arr if 'sol_collection' not in rand_arr]
    rand_arr = [ rand_arr for rand_arr in rand_arr if 'last_solution' not in rand_arr]
    rand_arr.sort()
    return rand_arr

def extract_mesh(meshpath,meshfile, solut_path):    # function to extract the mesh
    ## Loading the mesh
    text = 'Extracting the mesh'
    print(f'\n{text:.^80}\n')  
    mesh_fileName = os.path.join(meshpath,meshfile)
    # # Reading the mesh
    r = Reader('hdf_avbp')
    r['filename'] = mesh_fileName
    r['shared'] = True # Same mesh for all sol
    base = r.read() # b is the Base object of the Antares API
    
    r = Reader('hdf_avbp')
    r['filename'] = solut_path
    r['base'] = base
    base = r.read() # b is the Base object of the Antares API
    
    # Extracting the airfoil surface zone
    #airfoil = b[('Airfoil_Surface'),('Airfoil_Side_LE'),('Airfoil_Side_Mid'),('Airfoil_Side_TE'),('Airfoil_Trailing_Edge'),]
    #airfoil = base['Airfoil']
    airfoil = base[base.families['Patches']]
    t= Treatment('cut')
    t['base'] = airfoil
    t['type'] = 'plane'
    t['origin'] = [1.225,0.,0.0]
    t['normal'] = [0.,0.,1.]
    cut = t.execute() 
    
    t = Treatment('unwrapline')
    t['base'] = cut
    line = t.execute() # This a new object with only mid sections
    nodes = line['Airfoil'].shared['x'].shape[0]            # The number of nodes in the mesh
    return base, nodes

def Welford_avg(mean, current, iter):    # Welford's algorithm for calculating the mean and variance
    """
    Welford's algorithm for calculating the mean
    """
    mean = mean + (1/iter)*(current-mean) if iter > 0 else current
    return mean

def Extract_data(ave_dirName,sol_dirName, base, nodes):    # main function to exctact the mean data and append solutions

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
    field_names = ['P', 'wall_Stress_x']
    field_arrays = [np.zeros((nodes,timestep), dtype=np.float32) for _ in field_names]
    P, wall_Stress_x = field_arrays
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
            
            # Extracting the line 
            airfoil = base[base.families['Patches']]
            t = Treatment('cut')
            t['base'] = airfoil
            t['type'] = 'plane'
            t['origin'] = [1.225,0.,0.0]
            t['normal'] = [0.,0.,1.]
            cut = t.execute() 

            t = Treatment('unwrapline')
            t['base'] = cut
            line = t.execute() # This a new object with only mid sections
            
            P[:, count] = line['Airfoil'][0]['pressure']
            wall_Stress_x[:, count] = line['Airfoil'][0]['wall_Stress_x']
            count+=1    
    text = 'Average Field Data'
    print(f'\n{text:.^80}\n')
    base.show()
    
    #outputting the mean field of select variables

    cf = wall_Stress_x/(0.5*1.225*Uinf**2) # wall shear stress
    Cp = (P - 101325)/(0.5*1.225*Uinf**2) # pressure coefficient
    
    filename = 'Data_Surface_unsteady_CD_Rime_U'+str(Uinf)+'.h5'
    with h5py.File(filename, 'w') as f:
        f.attrs['velocity'] = str(Uinf)+'m/s'
        f.attrs['angle_of_attack'] = str(alpha)+'deg'
        current_date = datetime.now()
        f.attrs['Extracted_Date'] = current_date.strftime("%Y-%m-%d")
        f.create_group('Data_Midspan')
        f['Data_Midspan'].create_dataset('P', data=P)
        f['Data_Midspan'].create_dataset('Cf', data=cf)
        f['Data_Midspan'].create_dataset('Cp', data=Cp)
        f['Data_Midspan'].create_dataset('x', data=line['Airfoil'].shared['x'])
    text = 'Calculation Complete!'
    print(f'\n{text:.^80}\n')  


def main():
    base, nodes = extract_mesh(meshpath,meshfile,sol_dirName)                   # Extract the mesh
    Extract_data(ave_dirName,sol_dirName, base, nodes)          # Extract the data from the solution directory

# Function to read an instaneous solution
if __name__ ==  '__main__':
    main()
    