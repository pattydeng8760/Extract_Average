########################################################################################################
# Pope's criterion only: Q_pope = k_res / (k_res + k_sgs)
# k_res from velocity fluctuations, k_sgs from vis_turb and Delta=(VD_volume)^(1/3)
########################################################################################################
import os
import numpy as np
import sys
from antares import *
import builtins

# -----------------------------
# User inputs
# -----------------------------
nstart = 20
meshpath = '/project/p/plavoie/denggua1/BBDB_10AOA/MESH_ZONE_Apr24/'
meshfile = 'Bombardier_10AOA_Combine_Apr24.mesh.h5'
sol_dirName = '/project/p/plavoie/denggua1/BBDB_10AOA/RUN_ZONE_Apr24/SOLUT/'

# Pope/SGS model constants (CHECK your solver)
Ck = 0.094
eps = 1e-30

# Keep only these in output (everything else deleted)
vars_keep = ['Pope_Q', 'TKE_res', 'TKE_sgs', 'vis_turb_mean', 'Delta']


def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)

def sort_files(rand):
    rand_sub = os.listdir(rand)
    rand_arr = np.array([])
    for i in range(0, np.shape(rand_sub)[0]):
        file_split = os.path.splitext(rand_sub[i])[0]
        rand_arr = np.append(rand_arr, file_split)
    rand_arr = [*set(rand_arr)]
    rand_arr = [f for f in rand_arr if 'sol_collection' not in f]
    rand_arr = [f for f in rand_arr if 'last_solution' not in f]
    rand_arr.sort()
    return rand_arr

def extract_mesh(meshpath, meshfile):
    text = 'Extracting the mesh'
    print(f'\n{text:.^80}\n')
    mesh_fileName = os.path.join(meshpath, meshfile)
    print(f'Mesh file: {mesh_fileName}')

    r = Reader('hdf_avbp')
    r['filename'] = mesh_fileName
    r['shared'] = True
    base = r.read()
    nodes = base[0].shared['x'].shape[0]
    return base, nodes

def Calc_avg(mean, current, count):
    return mean + (1.0 / count) * (current - mean)

def collect_files(sol_dirName):
    arr_dir = os.path.join(sol_dirName)
    arr = os.listdir(arr_dir)
    arr.sort()

    file_list = []
    for i in range(nstart, len(arr)):
        dir_path = os.path.join(arr_dir, arr[i])
        if os.path.isdir(dir_path):
            files = sort_files(dir_path)
            for f in files:
                file_list.append(os.path.join(dir_path, f + '.h5'))

    print(f'Total files to process: {len(file_list)}')
    return file_list

def compute_pope(sol_dirName, base, nodes):
    file_list = collect_files(sol_dirName)
    total_files = len(file_list)
    if total_files == 0:
        raise RuntimeError("No solution files found. Check sol_dirName and nstart.")

    # -----------------------------
    # PASS 1: mean velocities + mean vis_turb + mean Delta
    # -----------------------------
    print(f'\n{"PASS 1: Mean velocities + mean vis_turb + mean Delta":.^80}\n')

    u_mean = np.zeros((nodes,), dtype=np.float64)
    v_mean = np.zeros((nodes,), dtype=np.float64)
    w_mean = np.zeros((nodes,), dtype=np.float64)
    vis_turb_mean = np.zeros((nodes,), dtype=np.float64)
    Delta_mean = np.zeros((nodes,), dtype=np.float64)

    count = 0
    for sol_file in file_list:
        if count % 10 == 0:
            print(f'Pass 1 - {count+1}/{total_files}: {os.path.basename(sol_file)}')

        r = Reader('hdf_avbp')
        r['base'] = base
        r['filename'] = sol_file
        base_i = r.read()

        # velocities
        base_i.compute('u = rhou / rho')
        base_i.compute('v = rhov / rho')
        base_i.compute('w = rhow / rho')

        u_inst = base_i[0][0]['u']
        v_inst = base_i[0][0]['v']
        w_inst = base_i[0][0]['w']

        # eddy viscosity and Delta
        nu_t = base_i[0][0]['vis_turb']              # assumed kinematic eddy viscosity
        VDvol = base_i[0][0]['VD_volume']            # node-centered
        Delta = np.power(np.maximum(VDvol, 0.0), 1.0/3.0)

        # update means
        count += 1
        u_mean = Calc_avg(u_mean, u_inst, count)
        v_mean = Calc_avg(v_mean, v_inst, count)
        w_mean = Calc_avg(w_mean, w_inst, count)
        vis_turb_mean = Calc_avg(vis_turb_mean, nu_t, count)
        Delta_mean = Calc_avg(Delta_mean, Delta, count)

    # -----------------------------
    # PASS 2: resolved TKE from fluctuations
    # -----------------------------
    print(f'\n{"PASS 2: Resolved TKE (k_res)":.^80}\n')

    TKE_res = np.zeros((nodes,), dtype=np.float64)
    count = 0
    for sol_file in file_list:
        if count % 10 == 0:
            print(f'Pass 2 - {count+1}/{total_files}: {os.path.basename(sol_file)}')

        r = Reader('hdf_avbp')
        r['base'] = base
        r['filename'] = sol_file
        base_i = r.read()

        base_i.compute('u = rhou / rho')
        base_i.compute('v = rhov / rho')
        base_i.compute('w = rhow / rho')

        u_inst = base_i[0][0]['u']
        v_inst = base_i[0][0]['v']
        w_inst = base_i[0][0]['w']

        u_p = u_inst - u_mean
        v_p = v_inst - v_mean
        w_p = w_inst - w_mean

        k_inst = 0.5 * (u_p*u_p + v_p*v_p + w_p*w_p)

        count += 1
        TKE_res = Calc_avg(TKE_res, k_inst, count)

    # -----------------------------
    # SGS TKE from nu_t and Delta
    # -----------------------------
    print(f'\n{"Computing k_sgs and Pope_Q":.^80}\n')

    Delta_safe = np.maximum(Delta_mean, eps)
    TKE_sgs = (vis_turb_mean / (Ck * Delta_safe))**2

    Pope_Q = TKE_res / (TKE_res + TKE_sgs + eps)
    
    # =============================================================================
    # Pope_Q statistics (unweighted and volume-weighted)
    # =============================================================================
    P = Pope_Q.astype(np.float64)

    # Remove non-finite values
    mask = np.isfinite(P)
    P = P[mask]

    # ---------- Unweighted statistics ----------
    mean_q   = float(np.mean(P))
    median_q = float(np.median(P))
    std_q    = float(np.std(P, ddof=0))
    min_q    = float(np.min(P))
    max_q    = float(np.max(P))

    print("\n" + "Pope_Q UNWEIGHTED statistics".center(80, "="))
    print(f"Count  : {P.size}")
    print(f"Mean   : {mean_q:.6f}")
    print(f"Median : {median_q:.6f}")
    print(f"StdDev : {std_q:.6f}")
    print(f"Min    : {min_q:.6f}")
    print(f"Max    : {max_q:.6f}")
    print("="*80 + "\n")

    # ---------- Volume-weighted statistics ----------
    # VD_volume ≈ Delta^3
    w = Delta_mean**3
    w = w.astype(np.float64)

    mask_w = np.isfinite(Pope_Q) & np.isfinite(w) & (w > 0.0)
    Pw = Pope_Q[mask_w].astype(np.float64)
    w  = w[mask_w]

    wmean_q = float(np.sum(w * Pw) / np.sum(w))
    wvar_q  = float(np.sum(w * (Pw - wmean_q)**2) / np.sum(w))
    wstd_q  = float(np.sqrt(wvar_q))

    print("\n" + "Pope_Q VOLUME-WEIGHTED statistics".center(80, "="))
    print(f"Weighted Mean   : {wmean_q:.6f}")
    print(f"Weighted StdDev : {wstd_q:.6f}")
    print("="*80 + "\n")



    # -----------------------------
    # Write output
    # -----------------------------
    r = Reader('hdf_avbp')
    r['base'] = base
    r['filename'] = file_list[0]  # structure donor
    out = r.read()

    # Delete everything except a minimal set (keep coords/connectivity automatically)
    # Safer: delete variables not in vars_keep if they exist
    existing = [k[0] for k in out[0][0].keys()]
    to_delete = [v for v in existing if v not in vars_keep]
    if len(to_delete) > 0:
        out.delete_variables(to_delete)

    # Attach outputs
    out[0][0]['TKE_res'] = TKE_res.astype(np.float32)
    out[0][0]['vis_turb_mean'] = vis_turb_mean.astype(np.float32)
    out[0][0]['Delta'] = Delta_mean.astype(np.float32)
    out[0][0]['TKE_sgs'] = TKE_sgs.astype(np.float32)
    out[0][0]['Pope_Q'] = Pope_Q.astype(np.float32)
    
    # =============================================================================
    # Store statistics as constant nodal fields
    # =============================================================================
    N = nodes

    out[0][0]['Pope_Q_mean']        = np.full(N, mean_q,   dtype=np.float32)
    out[0][0]['Pope_Q_median']      = np.full(N, median_q, dtype=np.float32)
    out[0][0]['Pope_Q_std']         = np.full(N, std_q,    dtype=np.float32)
    out[0][0]['Pope_Q_min']         = np.full(N, min_q,    dtype=np.float32)
    out[0][0]['Pope_Q_max']         = np.full(N, max_q,    dtype=np.float32)
    out[0][0]['Pope_Q_wmean']       = np.full(N, wmean_q,  dtype=np.float32)
    out[0][0]['Pope_Q_wstd']        = np.full(N, wstd_q,   dtype=np.float32)

    writer = Writer('hdf_antares')
    writer['filename'] = 'Pope_Criterion'
    writer['base'] = out
    writer['dtype'] = 'float32'
    writer.dump()

    print('\n' + 'Pope criterion written to Pope_Criterion.h5'.center(80, '=') + '\n')


def main():
    sys.stdout = open(os.path.join('log_Pope_Criterion.txt'), "w", buffering=1)
    base, nodes = extract_mesh(meshpath, meshfile)
    compute_pope(sol_dirName, base, nodes)

if __name__ == '__main__':
    main()
