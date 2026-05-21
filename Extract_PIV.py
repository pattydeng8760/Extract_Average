####################################################################################
# Extract cutplane data for PIV
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
import numpy as np
import matplotlib.pyplot as plt

# The file directory
dir = './'
file = 'Averaged_Solution_Vorticity_StrainRate_Reduced.h5'

# PIV Location x
#PIV = [1.27005,1.31511,1.375183,1.42222035, 1.48172998, 1.5641908]
#PIV = [1.48172998]
#Loc = [0.75]
#Loc = [0.58,0.78,1.05]
Loc = [0.30,0.58,0.77, 0.85, 0.95, 1.05, 1.15]
#Loc = [0.15,0.3,0.5,0.58,0.6,0.75,0.78,0.8,0.85,0.95,1.05,1.15,1.25]
#Loc = np.linspace(0.01,1,100)
AoA = 10 # Angle of Attack in degree
U_inf = 50 #Freestream velocity m/s
PIV = 1.245 + np.array(Loc)*0.3048*np.cos(AoA*np.pi/180)
PIV = list(PIV)
print(PIV)
# Reading the solution
r = Reader('hdf_antares')
r['filename'] = os.path.join(dir,file)
solut = r.read() # b is the Base object of the Antares API
solut.show()

# PIV Location
for i in range(0,np.shape(PIV)[0]):
    t= Treatment('cut')
    t['base'] = solut
    t['type'] = 'plane'
    tip_gap = -0.1034
    span = -0.2286
    # Midspan
    z_mid_span = tip_gap - 0.1651
    t['origin'] = [PIV[i],0.,z_mid_span]
    t['normal'] = [1.,0.,0.]
    inter = t.execute()
    #inter.show()

    writer = Writer('hdf_antares')
    #writer['base'] = inter[:,:,['x','y','z','u','v','w']]
    if 'StrainRate' in file: 
        writer['base'] = inter[:,:,['x','y','z','omega_x_mean','omega_y_mean','omega_z_mean','omega_x_rms','omega_y_rms','omega_z_rms',"S_mag_mean","Sf_mag_mean"]]
        print("Strain Rate and vorticity is included in the file, extracting Strain Rate data")
    else:
        writer['base'] = inter[:,:,['x','y','z','u','v','w','vort_x','vort_y','vort_z',"u_rms","v_rms","w_rms","TKE","Strain_Rate"]]
        print("Extracting avergae data")
    writer['filename'] = 'PIV_StrainVort_{:.2f}_U{:d}_A{:d}'.format(np.round(Loc[i]*100)/100,int(U_inf),int(AoA))
    writer.dump()
    del t, inter, writer


