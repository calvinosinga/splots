#!usr/bin/python3

import numpy as np

def reformat_sparta(sdata, nsnaps):

    sho = sdata['tcr_sho']

    def reformat_peri(lps):
        empty = np.zeros((lps.shape[0], nsnaps), dtype = int)
        for i in range(lps.shape[0]):
            snap = lps[i]
            empty[i, snap:] = 1
        return empty
    
    def reshape_arr(arr):
        # just ntracers
        if len(arr.shape) == 1:
            new_arr = np.reshape(arr, (arr.shape[0], 1))
            new_arr = np.repeat(new_arr, nsnaps, axis = 1)
            return new_arr
        elif len(arr.shape) >= 2:
            if nsnaps == arr.shape[1]:
                return arr
            else:
                new_arr = np.reshape(arr, (arr.shape[0], 1, arr.shape[1]))
                new_arr = np.repeat(new_arr, nsnaps, axis = 1)
                return new_arr   
            
    # turning structured array into a dictionary
    new_dict = {}
    for res in list(sho.keys()):
        shand = res.split('_', 1)[1]
        names = sho[res].dtype.names
        for name in names:
            arr = sho[res][name]
            if name == 'last_pericenter_snap':
                arr = reformat_peri(arr)
                new_dict[shand + '_had_pericenter'] = arr
            else:
                new_dict[shand + '_' + name] = arr

    # reshaping arrays
    for k,v in new_dict.items():
        new_dict[k] = reshape_arr(v)
    return new_dict

def norm_rad(tjy, hdata):
    # make array of the R200m values for each tracer
    # if all positions are -1, then subhalo tracer does not exist at that snapshot
    alive = ~(np.all(tjy['x'] == -1, axis = 2))
    hrad = np.ones_like(alive, dtype = float) * -1 # null val is -1

    nhalos = hdata['R200m'].shape[0]
    for i in range(nhalos):
        fidx = hdata['sho_tjy_first'][i]
        lidx = fidx + hdata['sho_tjy_n'][i]
        hrad[fidx:lidx, :] = hdata['R200m'][i, :]
    
    rnorm = np.ones_like(alive, dtype = float) * -1
    for i in range(rnorm.shape[0]):
        rnorm[i, alive[i]] = tjy['r'][i, alive[i]] / hrad[i, alive[i]]
    return rnorm

def norm_pos(tjy, hdata):
    # make array of the R200m values for each tracer
    # if all positions are -1, then subhalo tracer does not exist at that snapshot
    alive = ~(np.all(tjy['x'] == -1, axis = 2))
    hrad = np.ones_like(alive, dtype = float) * -1 # null val is -1

    nhalos = hdata['R200m'].shape[0]
    for i in range(nhalos):
        fidx = hdata['sho_tjy_first'][i]
        lidx = fidx + hdata['sho_tjy_n'][i]
        hrad[fidx:lidx, :] = hdata['R200m'][i, :]
    
    rnorm = np.ones((alive.shape[0], alive.shape[1], 3)) * -1
    for i in range(rnorm.shape[0]):
        rnorm[i, alive[i], :] = tjy['x'][i, alive[i], :] / hrad[i, alive[i], np.newaxis]
    return rnorm

    
# def reshape_host_to_tracer(tdata, hdata, host_fields):
#     nhalos = hdata.shape[0]
#     host_data = {}
#     for hf in host_fields:
#         arr = np.zeros(tdata.shape)
#         for i in range(nhalos):

def link_tracer_halo_data(tdata, hdata, sim):
    """
    Creates new array of sparta data, where the 
    data from the halos and tracers are combined.

    Args:
        tdata (_type_): _description_
        hdata (_type_): _description_
    """
    from colossus.halo import mass_so
    from colossus.cosmology import cosmology
    cosmology.setCosmology('WMAP7')

    nhalos = hdata.shape[0]
    zs = sim['snap_z']
    nsnaps = zs.shape[0]
    ntcrs = tdata['res_tjy'].shape[0]
    hrad = np.zeros((ntcrs, nsnaps)) - 1
    hmass = np.zeros_like(hrad) - 1
    host_ids = np.zeros((ntcrs, nsnaps), dtype = int) - 1
    host_idxs = np.zeros((ntcrs), dtype = int) - 1
    for i in range(nhalos):
        fidx = hdata['sho_tjy_first'][i]
        lidx = fidx + hdata['sho_tjy_n'][i]
        hrad[fidx:lidx, :] = hdata['R200m'][i, :]
        host_ids[fidx:lidx, :] = hdata['id'][i, :]
        host_idxs[fidx:lidx] = i
        for j in range(nsnaps):
            if hdata['R200m'][i, j] > 0:
                host_mass = mass_so.R_to_M(hdata['R200m'][i, j], zs[j], '200m')
                hmass[fidx:lidx, :] = host_mass

    trad = np.zeros((ntcrs, nsnaps)) - 1
    tmass = np.zeros_like(trad) - 1
    tidx = np.zeros(ntcrs, dtype = int) - 1
    for itcr in range(ntcrs):
        tcr_sparta_idx = np.where(tdata['res_tjy']['tracer_id'][itcr] == hdata['id'])
        if len(tcr_sparta_idx[0]) > 0:
            tidx[itcr] = tcr_sparta_idx[0][0]
            trad[itcr, :] = hdata['R200m'][tcr_sparta_idx[0][0], :]
            for j in range(trad.shape[1]):
                tmass[itcr, j] = mass_so.R_to_M(trad[itcr, j],  zs[j], '200m')

    tdata['hosts'] = {
        'R200m' : hrad,
        'M200m' : hmass,
        'id' : host_ids,
        'sparta_idx' : host_idxs
        }
    
    tdata['tcr_halos'] = {
        'R200m' : trad,
        'M200m' : tmass,
        'sparta_idx' : tidx
    }

    tdata['sim'] = {
        'snap_z' : zs
    }
    return tdata





