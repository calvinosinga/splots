import numpy as np
from tqdm import tqdm
import sys
import pickle as pkl

sys.path.append('/Users/cosinga/code/sparta_dev/analysis/build/lib/')
from sparta_tools import moria, sparta

SIMPATH = '/Users/cosinga/sims/sparta_output/'
SPARTA_FILE = 'sparta_no_cuts.hdf5'
MORIA_FILE = 'moria_tree_cbol_l0063_n0256_no_cuts.hdf5'
SIMNAME = 'L0063_N0256_CBol'

print('loading data...')
mconfig = moria.loadConfig(SIMPATH + '%s/%s'%(SIMNAME, MORIA_FILE))
fields = mconfig['field_list']
load_fields = ['Time_to_future_merger', 'id', 'mask_alive', 'mask_cut', 'nu200m_internal',
               'phantom','rs','scale_of_last_MM', 'x', 'v', 'vmax', 'vrms']
for f in fields:
    if 'Acc' in f:
        load_fields.append(f)
    elif 'parent' in f:
        load_fields.append(f)
    elif 'M' == f[0]:
        load_fields.append(f)
    elif 'R' == f[0]:
        load_fields.append(f)
    elif 'status' in f:
        load_fields.append(f)

mdata = moria.load(SIMPATH + '%s/%s'%(SIMNAME, MORIA_FILE), load_fields)
sdata = sparta.load(SIMPATH + '%s/%s'%(SIMNAME, SPARTA_FILE), load_halo_data = True)
data = {}

for f in tqdm(range(len(load_fields)), desc = 'loop replacing parent IDs with idxs...'):
    field = load_fields[f]
    if 'parent_id_cat' == field or 'parent_id_orbinf' == field:
        # replace IDs with idxs
        pids = mdata[field]
        nhalos = pids.shape[1]
        pidxs = np.zeros_like(pids) - 1
        for i in tqdm(range(nhalos), desc = "loop over halos for %s"%field):
            halo_pids = pids[:, i]
            is_sub = halo_pids > 0
            if np.any(is_sub):
                pid_matches = np.where(np.isin(mdata['id'], halo_pids[is_sub]))
                pidxs[is_sub, i] = pid_matches[1]
        new_field = field[:9] + 'x' + field[9:]
        data[new_field] = pidxs
    data[field] = mdata[field]

for res_type in list(sdata['tcr_sho'].keys()):
    tids = sdata['tcr_sho'][res_type]['tracer_id']
    _, counts = np.unique(tids, return_counts=True)
    field_name = 'tracer_id_%s'%res_type
    tidxs = np.empty(mdata['id'].shape[1], dtype=tuple)
    for i in tqdm(range(tidxs.shape[0]), 'loop over tracers for %s'%res_type):
        halo_ids = mdata['id'][:, i]
        match_idxs = np.where(np.isin(halo_ids, tids))
        if len(match_idxs[0]) >= 1:
            tidxs[i] = match_idxs

            if len(match_idxs[0]) >= 2:
                print(tidxs[i], tids[tidxs[i]])
                break
        
    data[field_name] = tidxs


pkl.dump(data, open('%s_dict.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)

        
