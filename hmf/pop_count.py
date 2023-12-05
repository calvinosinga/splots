from sparta_tools import moria
import numpy as np


class HPop(object):

    def __init__(self, mdata, snap):
        self.snap = snap
        self.md = mdata
        self.masks = {}
        return

    def __str__(self) -> str:
        out = ''
        for hp in self.list:
            out += str(hp)
        return out

    def addMask(self, name, )
def main():
    from sparta_tools import moria
    mpath = "/Users/cosinga/sims/sparta_output/L0063_N0256_CBol/moria_trees/moria_tree_orb_defs.hdf5"
    massdefs = ['M200m_all_spa', 'Morb-all_all_spa', 'Morb-all_orb_spa', 'Mvir_all_spa', 'Mpeak']
    pids = ['parent_id_cat', 'parent_id_orbinf']
    others = ['mask_alive', 'mask_cut', 'id']
    redshifts = [0, 0.3, 0.5, 1, 1.5, 2, 4]
    mconfig = moria.loadConfig(mpath)
    mdata = moria.load(mpath, massdefs + pids + others)
    snap_idx = np.argmin(np.abs(mconfig['simulation']['snap_z'] - 0))

    return

if __name__ == '__main__':
    main()