import sys
import os
import post_props as pp
import pickle as pkl

BASE = '/Users/cosinga/code/'
SIMPATH = '/Users/cosinga/sims/sparta_output/'


sys.path.append(BASE + 'sparta_dev/analysis/build/lib')
sys.path.append(BASE + 'tree_tracks')
from sparta_tools import moria, sparta
from tree_tracks.storage import Tree, Vines, Simulation

SIMNAME = 'L0063_N0256_CBol'
SPARTA_FILE = 'sparta_no_cuts.hdf5'
MORIA_FILE = 'moria_tree_cbol_l0063_n0256_no_cuts.hdf5'
SAVEPATH = '/Users/cosinga/code/splots/'
tree_filename = '%s_tree.pkl'%SIMNAME
link_filename = '%s_linked_tcr_data.pkl'%SIMNAME
vines_filename = '%s_vines.pkl'%SIMNAME
link_data_exists = os.path.exists(SAVEPATH + link_filename)
vines_exists = os.path.exists(SAVEPATH + vines_filename)
tree_exists = os.path.exists(SAVEPATH + tree_filename)


if not link_data_exists or not vines_exists or not tree_exists:
    if not link_data_exists or not vines_exists:
        sdata = sparta.load(SIMPATH + '%s/%s'%(SIMNAME, SPARTA_FILE), load_halo_data = True, 
                    anl_match = ['hps', 'rsp'], anl_pad_unmatched = True,
                    res_match = ['ifl', 'oct', 'tjy'], res_pad_unmatched = True)
   
    if not link_data_exists:
        data = pp.link_tracer_halo_data(sdata['tcr_sho'], sdata['halos'], sdata['simulation'])
        pkl.dump(data, open(link_filename, 'wb'), pkl.HIGHEST_PROTOCOL)
    # if not vines_exists:
    #     simdata = sdata['simulation']
    #     sim = Simulation(simdata['box_size'], simdata['snap_t'], ['snap_t', 'snap_z'], simdata)
    #     sdict = pp.reformat_sparta(sdata, sim.getSnaps())
    #     vines = Vines(sdict, sdata['halos'], sim)
    #     pkl.dump(vines, open(vines_filename, 'wb'), pkl.HIGHEST_PROTOCOL)
    if not tree_exists:
        massdefs = ['M200m_all_spa', 'Morb-all_all_spa', 'Morb-all_orb_spa', 'Mvir_all_spa', 'Mpeak']
        pids = ['parent_id_cat', 'parent_id_orbinf']
        others = ['mask_alive', 'mask_cut', 'id', 'x', 'R200m_all_spa', 'phantom']

        mdata = moria.load(SIMPATH + '%s/%s'%(SIMNAME, MORIA_FILE), massdefs + pids + others)
        mconfig = moria.loadConfig(SIMPATH + '%s/%s'%(SIMNAME, MORIA_FILE))
        simdata = mconfig['simulation']
        sim = Simulation(simdata['box_size'], simdata['snap_t'], ['snap_t', 'snap_z'], simdata)
        tree = Tree(mdata, sim)
        pkl.dump(tree, open(tree_filename, 'wb'), pkl.HIGHEST_PROTOCOL)


else:
    data = pkl.load(open(link_filename, 'rb'))
    vines = pkl.load(open(vines_filename, 'rb'))
    tree = pkl.load(open(tree_filename, 'rb'))