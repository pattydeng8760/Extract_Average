""" Extract the surface data from post-averaged AVBP surface solution files 
"""
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
from datetime import datetime
import csv
import matplotlib.pyplot as plt

# The path to the average solution on the airfoil surface
alpha = 10 # The angle of attack
U = 30  # The free stream velocity
# The path to the average solution on the airfoil surface
ave_solut = './Averaged_Solution_Surface.h5'
location = ['midspan_exp', 'midspan_act', 25,5, 'tip']


def compute_surface_data(ave_solut, location, alpha: int=10, U: int=30):
    
    # Reading the post-extracted surface datat
    print('\n----> Reading post-processed surface data')
    r = Reader('hdf_antares')
    r['filename'] = ave_solut
    b = r.read() # b is the Base object of the Antares API
    b.show()
    
    filename = 'Data_Surface_B_'+str(alpha)+'AOA_LES_U'+str(U)+'.h5'
    # Creating a hdf5 file to store the output surface data
    with h5py.File(filename, 'w') as f:
        f.attrs['velocity'] = str(U)+'m/s'
        f.attrs['angle_of_attack'] = str(alpha)+'deg'
        current_date = datetime.now()
        f.attrs['Extracted_Date'] = current_date.strftime("%Y-%m-%d")
        for i, loc in enumerate(location):
            loc_string = str(loc) if type(loc) != int else str(loc)+'mm_to_tip'
            f.create_group(loc_string)
            print('\n----> Extracting the data at {0}'.format(loc))
            line = extract_line(b, loc)
            extract_vars(line, loc_string, f,U)

def extract_vars(line:Base,location:str,f,U):
    x,y,z = line[0].shared['x'], line[0].shared['y'], line[0].shared['z']
    chord = Normalize(x)
    p = line[0][0]['P']
    prms = line[0][0]['P_rms']
    yplus = line[0][0]['wall_Yplus']
    wall_Stress_x = line[0][0]['wall_Stress_x']
    wall_Stress_y = line[0][0]['wall_Stress_y']
    wall_Stress_z = line[0][0]['wall_Stress_z']
    wall_shear_Stress = line[0][0]['wall_shear_Stress']
    wall_normal_Stress = line[0][0]['wall_normal_Stress']
    Cf = wall_shear_Stress / (0.5 * 1.225 * U**2)
    f[location].create_dataset('x', data=x)
    f[location].create_dataset('y', data=y)
    f[location].create_dataset('z', data=z)
    f[location].create_dataset('chord', data=chord)
    f[location].create_dataset('P', data=p)
    f[location].create_dataset('Prms', data=prms)
    f[location].create_dataset('wall_yplus', data=yplus)
    f[location].create_dataset('wall_Stress_x', data=wall_Stress_x)
    f[location].create_dataset('wall_Stress_y', data=wall_Stress_y)
    f[location].create_dataset('wall_Stress_z', data=wall_Stress_z)
    f[location].create_dataset('wall_shear_Stress', data=wall_shear_Stress)
    f[location].create_dataset('wall_normal_Stress', data=wall_normal_Stress)
    f[location].create_dataset('Cf', data=Cf)

def extract_line(b:Base, location: str):
    # Merge the zones
    print('   Merging the airfoil zones')
    airfoil = b[('Airfoil_Surface'),('Airfoil_Side_LE'),('Airfoil_Side_Mid'),('Airfoil_Side_TE'),('Airfoil_Trailing_Edge'),]
    myt = Treatment('merge')
    myt['base'] = airfoil
    myt['duplicates_detection'] = True
    myt['tolerance_decimals'] = 13
    merged = myt.execute()
    
    print('   Applying the cut at {0}'.format(location))
    t= Treatment('cut')
    t['base'] = merged
    t['type'] = 'plane'
    if location == 'tip':
        location = Geometry(location)
        a,n= location.a, location.n
        t['origin'] = [a[0],a[1],a[2]]
        t['normal'] = [n[0],n[1],n[2]]
    else:
        t['origin'] = [1.225,0.,Geometry(location).z]
        t['normal'] = [0.,0.,1.]
    cut = t.execute() # Merging the airfoil zones
    t = Treatment('unwrapline')
    t['base'] = cut
    line = t.execute() # This a new object with only mid sections
    return line

# Worker functions for intermediate processingclass geometry:
class Geometry:
    def __init__(self, location):
        self.tip_gap = -0.1034
        self.span = -0.2286
        # Handle location
        if location == 'midspan_exp':
            self.z = self.tip_gap - 0.1651
        elif location == 'midspan_act':
            self.z = self.tip_gap - abs(self.span) / 2
        elif isinstance(location, int):  # Dimensions in mm, reference at the tip
            self.z = self.tip_gap - location * 0.001
            location = f"{location}mm"
        elif location == 'tip':  # Dimensions in mm, reference at the tip
            self.a = np.array([1.29260396, -0.00172957, -0.1034])
            b = np.array([1.36223972, -0.00448763, -0.1035])
            c = np.array([1.46186891, -0.00772926, -0.1036])
            ab = b - self.a
            self.n = np.array([-ab[1], ab[0], 0])
        elif location == 'midspan_CD':
            self.z = 0.0
        else:
            raise ValueError("The location is not valid")
        
        self.location = location

# Function to normalize the airfoil coordinate
def Normalize(coordinate: np.array):
    if coordinate is None:
        raise ValueError("The coordinate is empty")
    normalize = (coordinate - np.min(coordinate))/(np.max(coordinate) -  np.min(coordinate))
    return normalize

def main():
    compute_surface_data(ave_solut, location, alpha=alpha, U=U)

if __name__ == '__main__':
    main()

