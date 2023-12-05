"""
Investigating how often phantom halos interfere with finding
pericenters.

Tasks:
    1. Plot fraction of all halos that exist as a phantom at any
        point in their lives.
    2. Plot a histogram of the fraction of snapshots each halo
        spends as a phantom (if they live more than 7 snapshots).
    3. Plot the fraction of halos at each snapshot that are 
        phantoms.
    4. Plot the fraction of pericenters found for tracers that
        are phantoms within +- 2 snapshots of the pericenter over
        the total number of host-tracer pairs.
    5. Find how many particles of the subhalo tracer had a pericenter
        around that snapshot.

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


phmask = mdata['phantom'] > 0
ntot = np.sum(mdata['mask_alive'], axis = 1)
nphan = np.sum(phmask, axis = 1)
plt.plot(snaps, nphan/ntot)
plt.xlabel("snapshot")
plt.ylabel("Phantoms / Total")
any_phant_frac = np.any(phmask, axis = 0)
any_phant_frac = np.sum(any_phant_frac) / mdata['id'].shape[1]
plt.title("%.3f = fraction of halos that are phantoms at least once"%(any_phant_frac))
plt.savefig("phantom_fraction_per_snap.png")
plt.clf()

ph_snap_count = np.sum(phmask, axis = 1)
alive_count = np.sum(mdata['mask_alive'], axis = 1)

alive_bins = np.arange(0, 20, 5)
for i in range(len(alive_bins)):

    alive_mask = alive_count > alive_bins[i]

    alive_frac = np.sum(alive_mask) / alive_count.shape[0]

    ph_over_alive = ph_snap_count[alive_mask] / alive_count[alive_mask]

    label = "%d min alive; %.3f frac cut"%(alive_bins[i], alive_frac)

    hist, edges = np.histogram(ph_over_alive)

    plt.plot((edges[:-1] + edges[1:])/ 2, hist, label = label)
plt.legend()
plt.xlabel('Fraction of snaps spent as phantom')
plt.ylabel('Count')
plt.savefig("phantoms_timespan.png")
    
