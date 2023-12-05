import sys
sys.path.append('/Users/cosinga/code/sparta_dev/analysis/build/lib/')
from sparta_tools import moria, sparta
import numpy as np
import matplotlib.pyplot as plt

# paths
BASE = '/Users/cosinga/code/'
SIMPATH = '/Users/cosinga/sims/sparta_output/'
MORIA_FILE = 'moria_tree_cbol_l0063_n0256_no_cuts.hdf5'
SIMNAME = 'L0063_N0256_CBol'
SPARTA_FILE = 'sparta_no_cuts.hdf5'

# plot parameters
print(plt.rcParams['lines.markersize'])
title = "tracer ID: %d, sub moria idx: %d\nhost sparta idx: %d, host moria idx: %d"
tcr_line_thick = 3
def unwrapPositions(positions, alives, boxsize):
        """
        Since the positions of the trackers are logged in a periodic
        box, the positions are wrapped when they cross the boundary
        of the box. This function unwraps their positions, allowing
        for their positions to be displayed continuously

        Args:
            
        """
        # first determine if unwrapping is necessary

        # if the range of positions is greater than L/2, then
        # unwrapping is needed
        pos_range = np.zeros((3, 2))
        pos_range[:, 0] = boxsize
        for i in range(len(positions)):
            pos = positions[i][alives[i], :]
            for j in range(pos_range.shape[0]):
                posmin = np.min(pos[:, j])
                posmax = np.max(pos[:, j])
                pos_range[j, 0] = min(posmin, pos_range[j, 0])
                pos_range[j, 1] = max(posmax, pos_range[j, 1])
        
        sub_pos = positions[0].copy()
        host_pos = positions[1].copy()
        did_unwrapping = False
        for a in range(pos_range.shape[0]):
            if pos_range[a, 1] - pos_range[a, 0] >= boxsize / 2:
                did_unwrapping = True
                unwrap_mask = sub_pos[:, a] <= boxsize / 2
                sub_pos[unwrap_mask & alives[0], a] += boxsize
                unwrap_mask = host_pos[:, a] <= boxsize / 2
                host_pos[unwrap_mask & alives[1], a] += boxsize

        # if did_unwrapping:
        #     print(pos_range)
        #     print("subhalo")
        #     print(positions[0])
        #     print(sub_pos)
        #     print('host')
        #     print(positions[1])
        #     print(host_pos)

        return sub_pos, host_pos

def find_transition_indices(arr):
    if len(arr) < 2 or np.all(arr == 0):
        return [], [], []

    zero_to_nonzero = np.where((arr[:-1] == 0) & (arr[1:] != 0))
    if len(zero_to_nonzero) > 0:
        zero_to_nonzero = zero_to_nonzero[0] + 1
    
    nonzero_to_zero = np.where((arr[:-1] != 0) & (arr[1:] == 0))

    if len(nonzero_to_zero) > 0:
        nonzero_to_zero = nonzero_to_zero[0] + 1
    
    
    nonzero_to_different_nonzero = np.where((arr[:-1] != 0) & (arr[:-1] != arr[1:]))
    if len(nonzero_to_different_nonzero) > 0:
        nonzero_to_different_nonzero = nonzero_to_different_nonzero[0] + 1

    return zero_to_nonzero, nonzero_to_zero, nonzero_to_different_nonzero
# desired quantities
RADIUS_DEF = 'R200m_bnd_spa_internal'
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
count = 0
snaps = np.arange(nsnaps)
# for each halo 
for i in range(nhalos):
    # for each tracer that belongs to halo
    ftcr = hdata['sho_ifl_first'][i]
    ltcr = ftcr + hdata['sho_ifl_n'][i]
    # if this halo has no tracers, skip
    if ftcr < 0: continue

    # get host's moria idx
    alive = hdata['id'][i] >= 0
    host_moria_mask = np.isin(mdata['id'], hdata['id'][i, alive], assume_unique = True)
    host_moria_idx = np.unique(np.where(host_moria_mask)[1])[0]
    # get tracer radii, set invalid values to nan
    tcr_slc = slice(ftcr, ltcr)
    tcr_r = tjy['r'][tcr_slc, :].copy()
    tcr_r[tcr_r <= 0] = np.nan 
    tcr_vr = tjy['vr'][tcr_slc, :].copy()

    tcr_nperi = oct['n_pericenter'][tcr_slc].copy()
    tcr_lowlim = oct['n_is_lower_limit'][tcr_slc].copy()
    tcr_perisnap = oct['last_pericenter_snap'][tcr_slc].copy()

    # get host radius, set invalid values to nan
    host_rad = mdata[RADIUS_DEF][:, host_moria_idx]
    host_rad[host_rad <= 0] = np.nan

    # test if any tracers reach beyond r/R200m > 3 at any snapshot
    mask1 = np.any(tcr_r / host_rad >= 3, axis = 1)
    
    mask2 = (tcr_nperi > 0) | (tcr_lowlim > 0)
    if not np.any(mask1 & mask2): continue
    # test if any tracers have
    # get the indices for the tracers that do meet this condition
    tcr_idxs = np.where(mask1 & mask2)[0]
    tcr_idxs = np.unique(tcr_idxs)
    # iterate over the tracers that meet this condition
    print('starting tracer loop:')
    for tidx in tcr_idxs:
        
        # find the tracer's sparta idx
        tracer_id = tjy['tracer_id'][tcr_slc][tidx]
        tcr_moria_mask = np.isin(mdata['id'], tracer_id, assume_unique=True)
        tcr_moria_idx = np.where(tcr_moria_mask)[1][0]

        # check if host is in tracer's parent IDs
        tcr_pids = mdata['parent_id_cat'][mdata['mask_alive'][:, tcr_moria_idx], tcr_moria_idx]
        host_ids_alive = hdata['id'][i, :] > 0
        in_mask = np.isin(hdata['id'][i, host_ids_alive], tcr_pids)

        # create matplotlib figure, axes
        fig, axes = plt.subplots(2, 2, figsize = (8, 6))
        
        # plot r and host R200m vs snapshot, label is tracer_id
        r = tcr_r[tidx, :]
        r_mask = ~np.isnan(r)

        host_mask = ~np.isnan(host_rad)
        
        print('plotting radius info...')
        # calculate r from moria
        peri_snap = tcr_perisnap[tidx]
        sub_pos = mdata['x'][:, tcr_moria_idx, :]
        host_pos = mdata['x'][:, host_moria_idx, :]
        sub_alv = mdata['mask_alive'][:, tcr_moria_idx]
        host_alv = mdata['mask_alive'][:, host_moria_idx]
        sub_pos, host_pos = unwrapPositions([sub_pos, host_pos], [sub_alv, host_alv], boxsize)
        moria_r = np.sqrt(np.sum((sub_pos - host_pos)**2, axis = 1))
        moria_r *= 1/(snap_z + 1) * 1e3
        
        plt.suptitle(title%(tracer_id, tcr_moria_idx, i, host_moria_idx))
        # prepare axis
        plt.sca(axes[0, 0])
        axes[0,0].axvline(peri_snap, color = 'pink', linestyle = 'dashed', label = 'pericenter', linewidth = tcr_line_thick)
        plt.ylabel('r (kpc/h)')
        plt.plot(snaps, r, label = 'from tracer data', linewidth = tcr_line_thick)
        plt.plot(snaps[host_mask], host_rad[host_mask], label = 'host R200m')
        if np.any(in_mask):
            plt.plot(snaps[sub_alv & host_alv], moria_r[sub_alv & host_alv], label = 'from moria')
            max_r = np.max(moria_r[r_mask])
            max_R = np.max(host_rad[r_mask])
        plt.legend(loc = 'upper right', fontsize = 8)
        
        plt.xlim(snaps[r_mask][0] - 4, snaps[r_mask][-1] + 4)
        
        # plot subhalo's mass vs snapshot
        plt.sca(axes[0, 1])
        
        print('plotting subhalo mass info...')
        sub_mass = mdata[MASS_DEF][:, tcr_moria_idx]
        plt.plot(snaps[sub_alv], sub_mass[sub_alv], label = 'Subhalo Mass')
        plt.ylabel('M200m (SM/h)')
        plt.yscale('log')
        axes[0, 1].axhline(res_lim, label = '200 * mptl', color = 'orange', linestyle = 'dashed')
        # find snapshots where subhalo/tracer was a phantom
        sub_phant = mdata['phantom'][:, tcr_moria_idx]
        phant_snaps = snaps[sub_phant > 0]
        has_label = False
        for phs in phant_snaps:
            if not has_label:
                has_label = True
                axes[0, 1].axvline(phs, label = 'is phantom', color = 'gray', linestyle = 'dashed')
                axes[0, 0].axvline(phs, label = 'sub is phantom', color = 'blue', linestyle = 'dashed')
            else:
                axes[0, 1].axvline(phs, color = 'gray', linestyle = 'dashed')
                axes[0, 0].axvline(phs, color = 'blue', linestyle = 'dashed')

        # find snapshots where major merger occurs in subhalo
        sub_mm = mdata['scale_of_last_MM'][:, tcr_moria_idx]
        mm_idxs = np.where(sub_mm == snap_a)
        has_label = False
        if len(mm_idxs)> 0:
            for mm in mm_idxs[0]:
                if not has_label:
                    axes[0, 1].axvline(mm, label = 'has major merger', color = 'green', linestyle = 'dashed')
                    has_label = True
                else:
                    axes[0, 1].axvline(mm, color = 'green', linestyle = 'dashed')
        
        # plot sub and host position
        print('plotting position info...')
        plt.sca(axes[1, 0])
        plt.xlabel('x (kpc/h)')
        plt.ylabel('y (kpc/h)')
        ax1 = 0
        ax2 = 1
        plt.plot(sub_pos[sub_alv & host_alv, ax1], sub_pos[sub_alv & host_alv, ax2], color = 'blue', label = 'subhalo position')
        plt.plot(host_pos[sub_alv & host_alv, ax1], host_pos[sub_alv & host_alv, ax2], color = 'orange', label = 'host position')
        plt.scatter(sub_pos[peri_snap, ax1], sub_pos[peri_snap, ax2], color = 'pink', s = 40 )
        sub_phant = mdata['phantom'][:, tcr_moria_idx]
        phant_snaps = snaps[(sub_phant > 0) & sub_alv & host_alv]
        plt.scatter(sub_pos[phant_snaps, ax1], sub_pos[phant_snaps, ax2], color = 'blue', marker='x')
        host_phant = mdata['phantom'][:, host_moria_idx]
        phant_snaps = snaps[(host_phant > 0) & sub_alv & host_alv]
        plt.scatter(host_pos[phant_snaps, ax1], host_pos[phant_snaps, ax2], color = 'orange', marker='x')
        plt.legend(fontsize = 8)
        plt.sca(axes[1, 1])
        print('plotting host mass info...')
        host_mass = mdata[MASS_DEF][:, host_moria_idx]
        axes[1, 1].axhline(res_lim, label = '200 * mptl', color = 'orange', linestyle = 'dashed')
        plt.plot(snaps[host_alv], host_mass[host_alv], label = 'Host Mass')
        plt.ylabel('M200m (SM/h)')

        # find snapshots where host was a phantom
        host_phant = mdata['phantom'][:, host_moria_idx]
        phant_snaps = snaps[host_phant > 0]
        has_label = False
        for phs in phant_snaps:
            if not has_label:
                has_label = True
                axes[1, 1].axvline(phs, label = 'is phantom', color = 'gray', linestyle = 'dashed')
                axes[0, 0].axvline(phs, label = 'host is phantom', color = 'orange', linestyle = 'dashed')
            else:
                axes[1, 1].axvline(phs, color = 'gray', linestyle = 'dashed')
                axes[0, 0].axvline(phs, color = 'orange', linestyle = 'dashed')
        
        # find snapshots where major merger occurs in host
        host_mm = mdata['scale_of_last_MM'][:, host_moria_idx]
        mm_idxs = np.where(host_mm == snap_a)
        has_label = False
        if len(mm_idxs)> 0:
            for mm in mm_idxs[0]:
                if not has_label:
                    axes[1, 1].axvline(mm, label = 'has major merger', color = 'green', linestyle = 'dashed')
                    has_label = True
                else:
                    axes[1, 1].axvline(mm, color = 'green', linestyle = 'dashed')
        
        
        # snapshots where halo becomes subhalo
        # plot these events in bottom panel
        plt.legend(loc = 'lower left', fontsize = 8)
        plt.yscale('log')
        print('saving plot...')
        plt.savefig('event_plots/%d_%d.png'%(tracer_id, i), bbox_inches = 'tight')
        plt.clf()
        plt.close()
    #     count += 1
    # if count > 25:
    #     break


