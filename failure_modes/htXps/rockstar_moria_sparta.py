import sys
import os
sys.path.append('/Users/cosinga/code/sparta_dev/analysis/build/lib/')
from sparta_tools import moria, sparta
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import re

# paths
BASE = '/Users/cosinga/code/'
SIMPATH = '/Users/cosinga/sims/sparta_output/'
MORIA_FILE = 'moria_tree_cbol_l0063_n0256_no_cuts.hdf5'
SIMNAME = 'L0063_N0256_CBol'
SPARTA_FILE = 'sparta_no_cuts.hdf5'
ROCKSTAR_FILE = '/Users/cosinga/sims/nbody/cbol_l0063_n0256/rockstar_trees_200m_bnd/tree_0_0_0.dat'
# verbosity - 1 includes print statemtents, 
# 2 breaks after one halo (with at least one tracer)
V = 2
V_COUNT = 50
# desired quantities
mdata = moria.load(
    SIMPATH + '%s/%s'%(SIMNAME, MORIA_FILE), 
    ['mask_alive', 'id',
     'phantom', 'parent_id_cat', 'parent_id_orbinf']
)

sdata = sparta.load(
    SIMPATH + '%s/%s'%(SIMNAME, SPARTA_FILE), load_halo_data = True,
    anl_match = ['hps', 'rsp'], anl_pad_unmatched = True,
    res_match = ['ifl', 'oct', 'tjy'], res_pad_unmatched = True
)

rdata = np.loadtxt(ROCKSTAR_FILE)
f = open(ROCKSTAR_FILE, 'r')
line = f.readline()
line = re.sub(r'\(\d+\)', '', line)
rhead = line[1:].split(' ')


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

temp = {}
for i, key in enumerate(rhead):
    temp[key] = rdata[:, i]
rdata = temp


# simple check to see if the number of halos in moria and the tree match
print('number of halos')
print(rdata['M200b'].shape[0], mdata['mask_alive'].sum())
print('number of subhalos')
rsub_count = (rdata['pid'] > 0).sum()
msub_count = (mdata['parent_id_cat'] > 0).sum()
print(rsub_count, msub_count)
print('number of tracers')
dif = tjy['last_snap'] - tjy['first_snap']
print(dif.sum())

# creating output data product

def getArr(dim):
    if dim  == 1:
        arr = np.zeros(ntcrs, dtype = int) - 1
    elif dim == 2:
        arr = np.zeros((ntcrs, nsnaps), dtype = int) - 1
    return arr

outdata = dict(
    tcr_sidx = getArr(1),
    host_sidx = getArr(1),
    tcr_midx = getArr(1),
    host_midx = getArr(1),
    tcr_ridx = getArr(2),
    host_ridx = getArr(2)
)

from tqdm import tqdm

# for each halo in sparta
for i in tqdm(range(nhalos), desc = "looping over halos"):
    # find the tracers for this host
    ftcr = hdata['sho_tjy_first'][i]
    ntcr = hdata['sho_tjy_n'][i]

    # if this halo does not have tracers, skip
    if ntcr == 0: continue

    # find hosts moria index
    host_alive = hdata['id'][i] >= 0
    host_moria_mask = np.isin(mdata['id'], hdata['id'][i, host_alive])
    host_moria_idx = np.unique(np.where(host_moria_mask)[1])
    

    if not len(host_moria_idx) == 1:
        host_moria_idx = -1
    else:
        host_moria_idx = host_moria_idx[0]
    
    # find hosts rockstar indices
    host_ridxs = np.zeros(nsnaps, dtype = int) - 1
    host_rmask = np.isin(rdata['id'], hdata['id'][i, host_alive])
    host_ridxs[host_alive] = np.where(host_rmask)[0]

    # for each tracer
    for itcr in range(ftcr, ftcr + ntcr):
        # find moria idx of the tracer
        tracer_id = tjy['tracer_id'][itcr]
        tcr_moria_mask = np.isin(mdata['id'], tracer_id)
        tcr_moria_idx = np.where(tcr_moria_mask)[1]
        # find rockstar idx of tracer

        tcr_ridxs = np.zeros(nsnaps, dtype = int) - 1
        if not len(tcr_moria_idx) == 1:
            tcr_moria_idx = -1
            
        else:
            tcr_moria_idx = tcr_moria_idx[0]
            tcr_mids = mdata['id'][:, tcr_moria_idx]
            tcr_malive = mdata['mask_alive'][:, tcr_moria_idx]
            tcr_rmask = np.isin(rdata['id'], tcr_mids[tcr_malive])
            tcr_ridxs[tcr_malive] = np.where(tcr_rmask)[0]

        
        
        # save output
        outdata['host_midx'][itcr] = host_moria_idx
        outdata['host_ridx'][itcr] = host_ridxs
        outdata['host_sidx'][itcr] = i
        outdata['tcr_midx'][itcr] = tcr_moria_idx
        outdata['tcr_ridx'][itcr] = tcr_ridxs
        outdata['tcr_sidx'][itcr] = itcr

    
pkl.dump(outdata, open("sporstar.pkl", 'wb'), pkl.HIGHEST_PROTOCOL)