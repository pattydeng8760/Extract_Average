import os
import numpy as np
import sys
import h5py
from glob import glob
from antares import *
import builtins



meshpath = '/project/rrg-moreaust-ac/denggua1/Bombardier_LES/B_10AOA_U50/MESH_Fine_Dec25/'
meshfile = 'Bombardier_10AOA_U50_Combine_Fine.mesh.h5'
sol_dirName =  '/project/rrg-moreaust-ac/denggua1/Bombardier_LES/B_10AOA_U50/RUN_Fine/SOLUT/'
nfolder = 18
Alpha = 10
Uinf = 50

text = 'Extracting the mesh'
print(f'\n{text:.^80}\n')
mesh_fileName = os.path.join(meshpath, meshfile)
print(f'Mesh file: {mesh_fileName}')
r = Reader('hdf_avbp')
r['filename'] = mesh_fileName
r['shared'] = True
base = r.read()
nodes = base[0].shared['x'].shape[0]

text="Loading the solution"
print(f'\n{text:.^80}\n')
sol_files = glob(os.path.join(sol_dirName,str(nfolder), '*.h5'))
sol_fileName = sol_files[-1]
print(f'Solution file: {sol_fileName}')
r = Reader('hdf_avbp')
r['base'] = base
r['filename'] = sol_fileName
base_i = r.read()
turb_vis  = base_i[0][0]['vis_turb']
lam_vis = base_i[0][0]['vis_lam']
IQV = (1+0.05*(turb_vis/lam_vis)**(0.5))**(-1)
base_i.compute('IQv = (1+0.05*(vis_turb/vis_lam)**(0.5))**(-1)')

text = 'Writing the output'
print(f'\n{text:.^80}\n')
output_base = Base()
output_base['0'] = Zone()
output_base[0].shared.connectivity = base[0][0].connectivity
output_base[0].shared["x"] = base[0][0]["x"]
output_base[0].shared["y"] = base[0][0]["y"]
output_base[0].shared["z"] = base[0][0]["z"]
output_base[0][str(0)] = Instant()
output_base[0][str(0)]["IQv"] = IQV.astype('float32')
file_name = "IQv_AOA{}_U{}.h5".format(Alpha, Uinf)
output_fileName = os.path.join('./', file_name)

writer = Writer('hdf_antares')
writer['filename'] = output_fileName
writer['dtype'] = 'float32'
writer['base'] = output_base
writer.dump()

