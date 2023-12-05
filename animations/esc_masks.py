import pickle as pkl
import sys
import numpy as np
from tqdm import tqdm

if len(sys.argv) > 1:
    rth = float(sys.argv[1])
else:
    rth = 3
print("making escaping subhalos mask for subhalos outside %.2f R200m..."%rth)
BASE = '/Users/cosinga/code/'
SIMPATH = '/Users/cosinga/sims/sparta_output/'
sys.path.append(BASE + 'sparta_dev/analysis/build/lib/')
from sparta_tools import moria, sparta
MORIA_FILE = 'moria_tree_cbol_l0063_n0256_no_cuts.hdf5'
SIMNAME = 'L0063_N0256_CBol'

mdata = moria.load(
    SIMPATH + '%s/%s'%(SIMNAME, MORIA_FILE), 
    ['R200m_all_spa', 'mask_alive', 'x', 'id', 'parent_id_cat']
)

nsnaps = mdata['x'].shape[0]
nhalos = mdata['x'].shape[1]
esc_sub_mask = np.zeros(nhalos, dtype = bool)
esc_sub_host_mask = np.zeros( nhalos, dtype = bool)
ids = mdata['id']; alive = mdata['mask_alive']; pos = mdata['x'].copy(); parent_ids = mdata['parent_id_cat']
pos[~alive, :] = np.nan

for i in tqdm(range(nhalos), desc = "Looping over halos..."):
    host_alive = alive[:, i]
    host_pos = pos[:, i, :]
    host_rad = mdata['R200m_all_spa'][:, i]
    sub_mask = np.isin(parent_ids, ids[host_alive, i])
    if not sub_mask.any():
        continue
    sub_idxs = np.where(np.any(sub_mask, axis = 0))[0]
    
    both_alive = alive[:,sub_idxs] & host_alive[:, np.newaxis]

    r = np.sqrt(np.sum((host_pos[:, np.newaxis, :] - pos[:, sub_idxs, :])**2, axis = 2))

    host_rad = np.tile(host_rad, (len(sub_idxs), 1)).T
    mask = np.any(r / host_rad > rth, axis = 0)
    esc_sub_host_mask[i] = np.any(mask)
    esc_sub_mask[sub_idxs[mask]] = True

pkl.dump([esc_sub_mask, esc_sub_host_mask], open('esc_masks%.2f.pkl'%rth, 'wb'), pkl.HIGHEST_PROTOCOL)