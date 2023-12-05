"""
Investigating the host-tracer pairs that are not considered
parent-subhalo pairs as well.

Tasks:
    1. get pure number of pairs / total pairs at each snapshot,
        and in different mass bins. Also plot for
        tracers that have already had a pericenter.
    2. Do these pairs exist in the raw catalogs? What time does
        ROCKSTAR call these subhalos?
    3. Create mask for tracers that belong to this population for
        later analyses.

THE FOLLOWING IS FOR EXAMINING WHY THESE ARE CONSIDERED PERICENTERS
IN THE FIRST PLACE:
    3. Plot the phi values until the pericenter is detected. Test
        with different r_ini values (going to further back snapshots).
        


"""
import sys
import os
sys.path.append('/Users/cosinga/code/sparta_dev/analysis/build/lib/')
from sparta_tools import moria, sparta
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm

# paths
BASE = '/Users/cosinga/code/'
SIMPATH = '/Users/cosinga/sims/sparta_output/'
MORIA_FILE = 'moria_tree_cbol_l0063_n0256_no_cuts.hdf5'
SIMNAME = 'L0063_N0256_CBol'
SPARTA_FILE = 'sparta_no_cuts.hdf5'

# verbosity - 1 includes print statemtents, 
# 2 breaks after one halo (with at least one tracer)
V = 2
V_COUNT = 50
# desired quantities
RADIUS_DEF = 'R200m_all_spa_internal'
MASS_DEF = 'M' + RADIUS_DEF[1:]
mdata = moria.load(
    SIMPATH + '%s/%s'%(SIMNAME, MORIA_FILE), 
    [RADIUS_DEF, MASS_DEF, 'mask_alive', 'x', 'id',
     'phantom', 'parent_id_cat', 'parent_id_orbinf',
     'scale_of_last_MM']
)

sdata = sparta.load(
    SIMPATH + '%s/%s'%(SIMNAME, SPARTA_FILE), load_halo_data = True,
    anl_match = ['hps', 'rsp'], anl_pad_unmatched = True,
    res_match = ['ifl', 'oct', 'tjy'], res_pad_unmatched = True
)

def create_htXps(mdata, sdata):
    # renaming symbols
    tjy = sdata['tcr_sho']['res_tjy']
    ifl = sdata['tcr_sho']['res_ifl']
    oct = sdata['tcr_sho']['res_oct']
    hdata = sdata['halos']
    boxsize = sdata['simulation']['box_size']
    nhalos = len(hdata['sho_oct_first'])
    nsnaps = sdata['simulation']['n_snaps']
    snap_z = sdata['simulation']['snap_z']
    snap_a = sdata['simulation']['snap_a']
    res_lim = 200 * sdata['simulation']['particle_mass']
    snaps = np.arange(nsnaps)
    ntcrs = tjy.shape[0]

    # desired quantities
    empty_arr = np.zeros((ntcrs, nsnaps)) - 1
    htXps = dict(
        tcr_inR = np.zeros_like(empty_arr, dtype = bool),
        sub_inR = np.zeros_like(empty_arr, dtype = bool),
        host_is_sub = empty_arr.copy(),
        htXps = np.zeros(ntcrs, dtype = bool),
        parent_id = empty_arr.copy(),
        M_host = empty_arr.copy(),
        M_sub = empty_arr.copy(),
        alive = np.zeros_like(empty_arr, dtype = bool),
        is_tcr = np.zeros_like(empty_arr, dtype = bool),
        is_tcr_interp = np.zeros_like(empty_arr, dtype = bool)
    )
    count = 0


    for i in tqdm(range(nhalos), desc = "creating htXps dict..."):
        # for each tracer that belongs to halo
        ftcr = hdata['sho_ifl_first'][i]
        ltcr = ftcr + hdata['sho_ifl_n'][i]

        # if this halo has no tracers, skip
        if ftcr < 0: continue

        # get host's moria idx
        host_alive = hdata['id'][i] >= 0
        host_moria_mask = np.isin(mdata['id'], hdata['id'][i, host_alive], assume_unique = True)
        host_moria_idx = np.unique(np.where(host_moria_mask)[1])
        if len(host_moria_idx) > 2:
            print("too many host indices found in moria")
            continue
    
        try:
            host_moria_idx = host_moria_idx[0]
        except Exception:
            print("moria index of host not found")
            continue


        for itcr in range(ftcr, ltcr):
            tracer_id = tjy['tracer_id'][itcr]
            tcr_moria_mask = np.isin(mdata['id'], tracer_id, assume_unique=True)
            tcr_moria_idx = np.where(tcr_moria_mask)
            try:
                if len(tcr_moria_idx[0]) > 2:
                    print("too many tracer-subhalo matches found in moria")
                    continue
                tcr_moria_idx = tcr_moria_idx[1][0]
            except IndexError:
                print("moria index of tracer not found")

            # now find if tracer-host is also parent-subhalo
            
            # check if host is in tracer's parent IDs
            sub_alive = mdata['mask_alive'][:, tcr_moria_idx]
            tcr_pids = mdata['parent_id_cat'][sub_alive, tcr_moria_idx]
            in_mask = np.isin(hdata['id'][i, host_alive], tcr_pids)

            if not np.any(in_mask):
                
                # saving parent IDs for the tracer
                htXps['parent_id'][itcr, sub_alive] = tcr_pids

                htXps['M_sub'][itcr, :] = mdata[MASS_DEF][:, tcr_moria_idx]
                htXps['M_host'][itcr, :] = mdata[MASS_DEF][:, host_moria_idx]
                htXps['alive'][itcr, :] = sub_alive
                fsnap = tjy['first_snap'][itcr]
                lsnap = tjy['last_snap'][itcr]
                htXps['is_tcr'][itcr, fsnap:lsnap] = True
                htXps['is_tcr_interp'][itcr, tjy['r'][itcr] > 0] = True
                htXps['htXps'][itcr] = True

                ppids = mdata['parent_id_cat'][:, host_moria_idx]
                htXps['host_is_sub'][itcr, :] = ppids > 0

                host_rad = mdata[RADIUS_DEF][:, host_moria_idx]
                rad_in_mask = np.zeros(nsnaps, dtype = bool)
                host_rad_mask = host_rad > 0
                tcr_rad_mask = tjy['r'][itcr, :] > 0
                both_mask = host_rad_mask & tcr_rad_mask
                rad_in_mask[both_mask] = tjy['r'][itcr, both_mask] <= host_rad[both_mask]
                htXps['tcr_inR'][itcr] = rad_in_mask

                sub_pos = mdata['x'][:, tcr_moria_idx]
                host_pos = mdata['x'][:, host_moria_idx]
                both_mask = sub_alive & host_alive
                sub_host_dist = np.zeros(nsnaps) - 1
                sub_host_dist[both_mask] = np.linalg.norm(sub_pos - host_pos)
                htXps['sub_inR'][itcr, both_mask] = \
                        sub_host_dist[both_mask] <= host_rad[both_mask]
                count += 1

    pkl.dump(htXps, open("htXps.pkl", 'wb'), pkl.HIGHEST_PROTOCOL)
    return


if __name__ == '__main__':
    if not os.path.exists("htXps.pkl"):
        create_htXps(mdata, sdata)
