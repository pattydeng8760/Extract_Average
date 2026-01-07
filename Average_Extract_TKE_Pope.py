########################################################################################################
# Pope's criterion: Q_pope = k_res / (k_res + k_sgs)
#
# k_res from velocity fluctuations
# k_sgs from eddy viscosity and Delta:
#   k_sgs ≈ (nu_t / (Ck * Delta))^2
#
# Filtering:
#  - Compute time-average of vort_x
#  - Build a mask using |vort_x_mean| >= VORTX_ABS_THRESHOLD
#  - Compute Pope_Q everywhere (finite)
#  - After vorticity filter, remove bottom n percentile of Pope_Q (default 10%)
#  - Stats/brackets reported for:
#      (1) all finite nodes
#      (2) vorticity-filtered nodes
#      (3) vorticity + Pope_Q percentile-filtered nodes
#
# Outputs:
#  - Antares H5: Pope_Criterion.h5 with fields kept in vars_keep
#      Pope_mask_vortx         : finite(Pope_Q) & vorticity filter
#      Pope_mask_vortx_popeq   : finite(Pope_Q) & vorticity filter & percentile filter
#  - Distribution export (for later plotting):
#      Pope_Q_filtered.npy  : 1D array of Pope_Q after BOTH filters
#      Pope_Q_filtered.h5   : dataset "Pope_Q_filtered" after BOTH filters
########################################################################################################

import os
import numpy as np
import sys
import h5py
from antares import *
import builtins

# -----------------------------
# User inputs
# -----------------------------
nstart = 12

meshpath = '/project/rrg-plavoie/denggua1/BBDB_10AOA/MESH_Fine_Jul25/'
meshfile = 'Bombardier_10AOA_U30_Combine_Fine.mesh.h5'
sol_dirName = '/project/rrg-plavoie/denggua1/BBDB_10AOA/RUN_Fine_Jul25/SOLUT/'

# Pope constants
Ck = 0.094
eps = 1e-30

# If vis_turb is dynamic mu_t (Pa*s), set True so nu_t = mu_t / rho
VIS_TURB_IS_DYNAMIC_MU = True

# -----------------------------
# Vorticity-based filtering
# -----------------------------
# Any node with |vort_x_mean| < VORTX_ABS_THRESHOLD is excluded from Pope stats/brackets
VORTX_ABS_THRESHOLD = 1500   # <-- SET THIS (units: 1/s)

# Optional: also print/use time-mean vort_x (computed and reported, not required for the mask)
REPORT_VORTX_MEAN_STATS = True

# Pope_Q bracket printing
BRACKET_STEP_PERCENT = 10   # 0-10-...-100 bins over [0,1]

# -----------------------------
# Pope_Q percentile filtering (applied AFTER vorticity filtering)
# -----------------------------
POPEQ_BOTTOM_PERCENTILE = 10.0  # remove bottom 10% among vorticity-filtered nodes
EXPORT_PDFILES = True
POPEQ_NPY_NAME = "Pope_Q_filtered.npy"
POPEQ_H5_NAME  = "Pope_Q_filtered.h5"

# Keep only these in Antares output
vars_keep = [
    'Pope_Q',
    'TKE_res',
    'TKE_sgs',
    'vis_turb_mean',
    'Delta',
    'vort_x_mean',
    'Pope_mask_vortx',
    'Pope_mask_vortx_popeq'
]

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

def compute_delta_once(base, donor_file):
    print(f'\n{"Computing Delta once (mesh quantity)":.^80}\n')

    r = Reader('hdf_avbp')
    r['base'] = base
    r['filename'] = donor_file
    b0 = r.read()

    VDvol = b0[0][0]['VD_volume']  # node-centered in your setup
    Delta = np.power(np.maximum(VDvol, 0.0), 1.0/3.0).astype(np.float64)
    Delta_safe = np.maximum(Delta, eps)

    dmin = float(np.min(Delta_safe[np.isfinite(Delta_safe)]))
    dmax = float(np.max(Delta_safe[np.isfinite(Delta_safe)]))
    print(f"Delta(min,max) = ({dmin:.6e}, {dmax:.6e})")

    return Delta, Delta_safe

def print_brackets(Q, step=10):
    step = int(step)
    if step <= 0 or 100 % step != 0:
        raise ValueError("BRACKET_STEP_PERCENT must be a positive divisor of 100 (e.g., 1,2,5,10,20,25,50).")

    edges = np.linspace(0.0, 1.0, int(100/step) + 1)
    counts, _ = np.histogram(Q, bins=edges)
    total = Q.size

    print("\n" + f"Pope_Q brackets ({step}% bins)".center(80, "="))
    for i in range(len(counts)):
        lo = edges[i]
        hi = edges[i+1]
        c = int(counts[i])
        frac = (c/total*100.0) if total > 0 else 0.0
        bracket = f"[{lo:0.2f}, {hi:0.2f}{')' if i < len(counts)-1 else ']'}"
        print(f"{int(lo*100):3d}%–{int(hi*100):3d}%  {bracket:14s} : {c:10d}  ({frac:6.2f}%)")
    print("="*80 + "\n")

def stats_block(name, Qvals):
    mean_q   = float(np.mean(Qvals))
    median_q = float(np.median(Qvals))
    std_q    = float(np.std(Qvals, ddof=0))
    min_q    = float(np.min(Qvals))
    max_q    = float(np.max(Qvals))
    p10, p90 = [float(x) for x in np.percentile(Qvals, [10, 90])]

    print("\n" + f"{name}".center(80, "="))
    print(f"Count   : {Qvals.size}")
    print(f"Mean    : {mean_q:.6f}")
    print(f"Median  : {median_q:.6f}")
    print(f"StdDev  : {std_q:.6f}")
    print(f"P10/P90 : {p10:.6f} / {p90:.6f}")
    print(f"Min/Max : {min_q:.6f} / {max_q:.6f}")
    print("="*80 + "\n")

    return mean_q, median_q, std_q, min_q, max_q, p10, p90

def compute_pope(sol_dirName, base, nodes):
    file_list = collect_files(sol_dirName)
    total_files = len(file_list)
    if total_files == 0:
        raise RuntimeError("No solution files found. Check sol_dirName and nstart.")
    mesh_base = base.copy()
    # Compute Delta once
    Delta, Delta_safe = compute_delta_once(base, file_list[0])

    # -----------------------------
    # PASS 1: mean velocities + mean nu_t + mean vort_x
    # -----------------------------
    print(f'\n{"PASS 1: Mean velocities + mean vis_turb + mean vort_x":.^80}\n')

    u_mean = np.zeros((nodes,), dtype=np.float64)
    v_mean = np.zeros((nodes,), dtype=np.float64)
    w_mean = np.zeros((nodes,), dtype=np.float64)
    vis_turb_mean = np.zeros((nodes,), dtype=np.float64)
    vort_x_mean = np.zeros((nodes,), dtype=np.float64)

    count = 0
    for sol_file in file_list:
        if count % 10 == 0:
            print(f'Pass 1 - {count+1}/{total_files}: {os.path.basename(sol_file)}')

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

        # vorticity_x (assumes variable name in files is 'vort_x')
        vort_x_inst = base_i[0][0]['vort_x']

        # Eddy viscosity -> nu_t
        vis_turb = base_i[0][0]['vis_turb']
        if VIS_TURB_IS_DYNAMIC_MU:
            rho = base_i[0][0]['rho']
            nu_t = vis_turb / np.maximum(rho, eps)
        else:
            nu_t = vis_turb

        count += 1
        u_mean = Calc_avg(u_mean, u_inst, count)
        v_mean = Calc_avg(v_mean, v_inst, count)
        w_mean = Calc_avg(w_mean, w_inst, count)
        vis_turb_mean = Calc_avg(vis_turb_mean, nu_t, count)
        vort_x_mean = Calc_avg(vort_x_mean, vort_x_inst, count)

    if REPORT_VORTX_MEAN_STATS:
        vx = vort_x_mean[np.isfinite(vort_x_mean)]
        print("\n" + "vort_x_mean statistics".center(80, "="))
        print(f"Mean(vort_x_mean)   : {float(np.mean(vx)):.6e}")
        print(f"Min/Max(vort_x_mean): {float(np.min(vx)):.6e} / {float(np.max(vx)):.6e}")
        print("="*80 + "\n")

    # Build the vorticity-based mask ONCE from vort_x_mean
    Pope_mask_vortx = np.isfinite(vort_x_mean) & (np.abs(vort_x_mean) >= float(VORTX_ABS_THRESHOLD))

    kept = int(np.sum(Pope_mask_vortx))
    print(f"Vorticity filter: keep nodes with |vort_x_mean| >= {VORTX_ABS_THRESHOLD:.6e}")
    print(f"Kept nodes: {kept} / {nodes} ({kept/nodes*100:.2f}%)")

    # -----------------------------
    # PASS 2: resolved TKE + SGS TKE (instantaneous then average)
    # -----------------------------
    print(f'\n{"PASS 2: Resolved TKE (k_res) + SGS TKE (k_sgs)":.^80}\n')

    TKE_res = np.zeros((nodes,), dtype=np.float64)
    TKE_sgs = np.zeros((nodes,), dtype=np.float64)

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

        k_res_inst = 0.5 * (u_p*u_p + v_p*v_p + w_p*w_p)

        # instantaneous k_sgs from nu_t and Delta
        vis_turb = base_i[0][0]['vis_turb']
        if VIS_TURB_IS_DYNAMIC_MU:
            rho = base_i[0][0]['rho']
            nu_t = vis_turb / np.maximum(rho, eps)
        else:
            nu_t = vis_turb

        k_sgs_inst = (nu_t / (Ck * Delta_safe))**2

        count += 1
        TKE_res = Calc_avg(TKE_res, k_res_inst, count)
        TKE_sgs = Calc_avg(TKE_sgs, k_sgs_inst, count)

    # -----------------------------
    # Pope_Q field
    # -----------------------------
    print(f'\n{"Computing Pope_Q":.^80}\n')
    Pope_Q = TKE_res / (TKE_res + TKE_sgs + eps)

    # -----------------------------
    # Statistics / brackets with filtering
    # -----------------------------
    print(f'\n{"Pope_Q statistics (filtered)":.^80}\n')

    finite_all = np.isfinite(Pope_Q)
    Q_all = Pope_Q[finite_all].astype(np.float64)

    # Mask 1: finite Pope_Q + vorticity filter
    mask_vortx = Pope_mask_vortx & finite_all
    Q_f = Pope_Q[mask_vortx].astype(np.float64)

    print(f"Stats mask #1: finite(Pope_Q) & (|vort_x_mean| >= {VORTX_ABS_THRESHOLD:.6e})")
    print(f"Kept nodes for stats #1: {Q_f.size} / {Q_all.size} ({(Q_f.size/Q_all.size*100.0 if Q_all.size else 0.0):.2f}%)\n")

    # Percentile filter applied AFTER vorticity filter
    if Q_f.size == 0:
        raise RuntimeError("No nodes remain after vorticity filtering; cannot apply Pope_Q percentile filter.")

    pct = float(POPEQ_BOTTOM_PERCENTILE)
    if pct < 0.0 or pct >= 100.0:
        raise ValueError("POPEQ_BOTTOM_PERCENTILE must be in [0, 100).")

    popeq_cut = float(np.percentile(Q_f, pct))
    print(f"Pope_Q percentile filter: removing bottom {pct:.2f}% of Pope_Q (on vorticity-filtered nodes)")
    print(f"Cutoff value (P{pct:.2f}) = {popeq_cut:.6f}")

    mask_popeq = (Pope_Q >= popeq_cut) & finite_all

    # Mask 2: finite Pope_Q + vorticity filter + percentile filter
    mask_vortx_popeq = mask_vortx & mask_popeq
    Q_f2 = Pope_Q[mask_vortx_popeq].astype(np.float64)

    print(f"Kept nodes after BOTH filters: {Q_f2.size} / {Q_f.size} ({(Q_f2.size/Q_f.size*100.0):.2f}%)\n")

    # Brackets
    print_brackets(Q_all, step=BRACKET_STEP_PERCENT)
    print_brackets(Q_f, step=BRACKET_STEP_PERCENT)
    print_brackets(Q_f2, step=BRACKET_STEP_PERCENT)

    # Stats blocks
    stats_block("Pope_Q UNWEIGHTED (all finite nodes)", Q_all)
    stats_block("Pope_Q UNWEIGHTED (vorticity-filtered nodes)", Q_f)
    stats_block("Pope_Q UNWEIGHTED (vortx + Pope_Q percentile filtered nodes)", Q_f2)

    # -----------------------------
    # Export Pope_Q values for distribution plotting later
    # -----------------------------
    if EXPORT_PDFILES:
        np.save(POPEQ_NPY_NAME, Q_f2.astype(np.float64))
        print(f"Saved filtered Pope_Q values to: {POPEQ_NPY_NAME}  (count={Q_f2.size})")

        with h5py.File(POPEQ_H5_NAME, "w") as hf:
            hf.create_dataset("Pope_Q_filtered", data=Q_f2.astype(np.float64), compression="gzip")
            hf.attrs["description"] = "Pope_Q values after vorticity filter and bottom-percentile removal"
            hf.attrs["VORTX_ABS_THRESHOLD"] = float(VORTX_ABS_THRESHOLD)
            hf.attrs["POPEQ_BOTTOM_PERCENTILE"] = float(POPEQ_BOTTOM_PERCENTILE)
            hf.attrs["PopeQ_cutoff_value"] = float(popeq_cut)

        print(f"Saved filtered Pope_Q values to: {POPEQ_H5_NAME}  (dataset='Pope_Q_filtered')")

    # -----------------------------
    # Write Antares output
    # -----------------------------
    r = Reader('hdf_avbp')
    r['base'] = base
    r['filename'] = file_list[0]  # structure donor
    out = r.read()

    existing = [k[0] for k in out[0][0].keys()]
    to_delete = [v for v in existing if v not in vars_keep]
    if len(to_delete) > 0:
        out.delete_variables(to_delete)

    output_base = Base()
    output_base['0'] = Zone()
    output_base[0].shared.connectivity = mesh_base[0][0].connectivity
    output_base[0].shared["x"] = mesh_base[0][0]["x"]
    output_base[0].shared["y"] = mesh_base[0][0]["y"]
    output_base[0].shared["z"] = mesh_base[0][0]["z"]
    output_base[0][str(0)] = Instant()
    output_base[0][str(0)]["Pope_Q"] = Pope_Q.astype(np.float32)
    output_base[0][str(0)]["TKE_res"] = TKE_res.astype(np.float32)
    output_base[0][str(0)]["TKE_sgs"] = TKE_sgs.astype(np.float32)

    writer = Writer('hdf_antares')
    writer['filename'] = 'Pope_Criterion'
    writer['dtype'] = 'float32'
    writer['base'] = output_base
    writer.dump()

    print('\n' + 'Pope criterion written to Pope_Criterion.h5'.center(80, '=') + '\n')

def main():
    sys.stdout = open(os.path.join('log_Pope_Criterion.txt'), "w", buffering=1)
    base, nodes = extract_mesh(meshpath, meshfile)
    compute_pope(sol_dirName, base, nodes)

if __name__ == '__main__':
    main()
