#!usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from figrid.data_sort import DataSort
from figrid.data_container import DataContainer
import copy
import abc
import itertools

class HFSort():

    def __init__(self, moria_data, moria_config, mdefs, sdefs, zs):
        self.md = moria_data
        self.mdefs = mdefs     
        self.sdefs = sdefs
        self.pdefs = self.getParentDefs(sdefs)
        self.zs = zs
        self.mconfig = moria_config
        self.ds = DataSort()
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['font.family'] = 'serif'
        return

    def getMass(self, mdef):
        if mdef == 'Mpeak-orb':
            mass = self.getMass('Morb-all_all_spa')
            peak = np.maximum.accumulate(mass, axis = 0)
            return peak
        else:
            return self.md[mdef]
    
    def getDS(self) -> DataSort:
        return self.ds
    
    @abc.abstractmethod
    def hasRadDef(m_label):
        is_morball = 'Morb-all' in m_label
        mpeak_orb = 'Mpeak' in m_label
        
        return not (is_morball or mpeak_orb)
    
    def getRad(self, m_label):
        if not HFSort.hasRadDef(m_label):
            raise ValueError("%s does not have radius def"%m_label)
        rstr = HFSort.mToRstr(m_label)
        return self.md[rstr]
    
    @abc.abstractmethod
    def mToRstr(m_label):
        r_label = copy.copy(m_label)
        r_label.replace('M', 'R', 1)
        return r_label
    
    def getParentDefs(self, subdefs):
        pdefs = []
        for sd in subdefs:
            if 'parent_id' in sd:
                pdefs.append(sd)
        return pdefs
    
    def getIter(self, order = ['sdefs', 'mdefs', 'zs']):
        args = []
        for o in order:
            if o == 'mdefs':
                args.append(self.mdefs)
            elif o == 'sdefs':
                args.append(self.sdefs)
            elif o == 'zs':
                args.append(self.zs)
            elif o == 'pdefs':
                args.append(self.pdefs)
            elif o == 'm+rdefs':
                mass_w_r = []
                for md in self.mdefs:
                    if HFSort.hasRadDef(md):
                        mass_w_r.append(md)
                args.append(mass_w_r)

        return itertools.product(*args)
    
    def getSnap(self, z):
        snapz = self.mconfig['simulation']['snap_z']
        close_idx = np.argmin(np.abs(snapz - z))
        return close_idx

    @abc.abstractmethod
    def cutToStr(cut):
        mn = cut[0]; mx = cut[1]
        out = ''
        if not mn == -np.inf:
            out = '%.2e_'%mn
        else:
            out = '_'
        
        if not mx == np.inf:
            out += '%.2e'%mx
        return out
            
    def findMIdx(self, halo_id):
        idx = np.where(self.md['id'] == halo_id)
        if len(idx[0]) == 0:
            raise ValueError('id %d not found'%halo_id)
        
        return idx[1][0]
    
    def getMask(self, pid_str, with_alive = True):
        alive = self.md['mask_alive'][:]
        if 'cat' in pid_str:
            pid = self.md['parent_id_cat']
        elif 'orb' in pid_str:
            pid = self.md['parent_id_orbinf']
        else:
            if with_alive:
                return alive
            else:
                return np.ones(self.md.shape, dtype = bool)
        if 'sub' in pid_str:
            mask = pid >= 0
        elif 'parent' in pid_str:
            mask = pid < 0
        if with_alive:
            return mask & alive
        else:
            return mask

    def defaultProps(self, mtype, htype, redshift) -> dict:
        snap_idx = self.getSnap(redshift)
        props = dict(
            host_type = htype,
            redshift = redshift,
            boxsize = self.mconfig['simulation']['box_size'],
            mptl = self.mconfig['simulation']['particle_mass'],
            snapshot = snap_idx,
            full_mdef_str = mtype
        )
        msplit = mtype.split('_')
        props['mass_def'] = msplit[0]
        if len(msplit) > 1:
            props['ptl_sel'] = msplit[1]
        else:
            props['ptl_sel'] = ''
        if len(msplit) == 4:
            props['is_internal'] = 'true'
        else:
            props['is_internal'] = 'false'
        if 'parent' in htype:
            props['is_host'] = 'true'
        else:
            props['is_host'] = 'false'
        if 'orbinf' in htype:
            props['orb_htype'] = 'true'
        else:
            props['orb_htype'] = 'false'
        return props
    
    def makeHMF(self, bins):
        print("making halo mass functions...")
        def _HMF(mass, bins = None):
            if bins is None:
                bins = np.geomspace(np.min(mass), np.max(mass), bins)
            elif isinstance(bins, int) or isinstance(bins, float):
                bins = np.geomspace(np.min(mass), np.max(mass), bins)
            
            hist, edges = np.histogram(mass, bins)
            mid = (edges[:-1] + edges[1:]) / 2
            return mid, hist
        


        for s, m, z in self.getIter():
            mask = self.getMask(s)
            mass = self.getMass(m)
            snap_idx = self.getSnap(z)
            
            props = dict(
                function = 'host_mass',
                nhalos = np.sum(mask[snap_idx])
            )

            props.update(self.defaultProps(m, s, z))


            data = _HMF(mass[snap_idx, mask[snap_idx, :]], bins)
            dc = DataContainer(data)
            dc.update(props)
            self.ds.append(dc)

        return

    def makeSHMFF(self, mu_bins, host_cut = (-np.inf, np.inf), sub_cut = (-np.inf, np.inf)):
        print("making subhalo mass fraction function...")
        def _SHMFF(mu, bins = None):

            if bins is None:
                bins = np.geomspace(np.min(mu), np.max(mu), bins)
            elif isinstance(bins, int) or isinstance(bins, float):
                bins = np.geomspace(np.min(mu), np.max(mu), bins)
            
            hist, edges = np.histogram(mu, bins)
            mid = (edges[:-1] + edges[1:]) / 2
            return mid, np.cumsum(hist[::-1])[::-1]
        

        alive = self.md['mask_alive']
        props = {
                'function':'sub_mass',
                'host_cut' : HFSort.cutToStr(host_cut),
                'sub_cut' : HFSort.cutToStr(sub_cut)
            }
        for m, p, z in self.getIter(['mdefs', 'pdefs', 'zs']):
            snap_idx = self.getSnap(z)
            mass = self.getMass(m)[snap_idx, :]
            host_mask = self.getMask(p, False)[snap_idx, :]
            sub_mask = ~host_mask
            sub_mask &= alive[snap_idx, :]; host_mask &= alive[snap_idx, :]

            host_cut_mask = (mass >= host_cut[0]) & (mass < host_cut[1])
            sub_cut_mask = (mass >= sub_cut[0]) & (mass < sub_cut[1])

            host_mask &= host_cut_mask
            sub_mask &= sub_cut_mask

            nsubs = np.sum(sub_mask)
            host_ids = self.md['id'][snap_idx, host_mask]
            par_ids = self.md[p][snap_idx, sub_mask]
            sub_mass = mass[sub_mask]
            host_mass = mass[host_mask]
            nhosts = len(np.intersect1d(host_ids, par_ids))
            # initialize mu to -1
            mu = np.zeros_like(sub_mass) - 1
            
            for i in range(nsubs):
                pid = par_ids[i]
                host_moria_idx = np.where(host_ids == pid)[0]

                # if host not found, skip (outside mass cut)
                if len(host_moria_idx) == 0:
                    continue

                host_moria_idx = host_moria_idx[0]
                mu[i] = sub_mass[i] / host_mass[host_moria_idx]
            
            
            # get histogram
            mid, hist = _SHMFF(mu, mu_bins)
            props.update(self.defaultProps(m, p, z))
            dc = DataContainer([mid, hist / nhosts])
            dc.update(props)
            self.ds.append(dc)
        return

    def makeSF(self, bins):
        print("making subhalo fractions...")
        def _SF(mass, bins = None):

            if bins is None:
                bins = np.geomspace(np.min(mass), np.max(mass), bins)
            elif isinstance(bins, int) or isinstance(bins, float):
                bins = np.geomspace(np.min(mass), np.max(mass), bins)
            
            hist, edges = np.histogram(mass, bins)
            mid = (edges[:-1] + edges[1:]) / 2
            return mid, hist
        
        alive = self.md['mask_alive']
        props = {'function' : 'sub_frac'}
        for p, m, z in self.getIter(['pdefs', 'mdefs', 'zs']):
            snap_idx = self.getSnap(z)
            host_mask = self.getMask(p, False)
            sub_mask = ~host_mask
            host_mask &= alive; sub_mask &= alive
            mass = self.getMass(m)

            sub_mass = mass[snap_idx, sub_mask[snap_idx]]
            host_mass = mass[snap_idx, host_mask[snap_idx]]

            mid, sub_hist = _SF(sub_mass, bins)
            mid, host_hist = _SF(host_mass, bins)

            ratio = sub_hist / (host_hist + sub_hist)
            zmask = (host_hist + sub_hist) <= 0
            ratio[zmask] = np.nan

            props.update(self.defaultProps(m, p, z))
            dc = DataContainer([mid, ratio])
            dc.update(props)
            self.ds.append(dc)


        return
    
    def makeProf(self, bins, host_cut = (-np.inf, np.inf), 
                sub_cut = (-np.inf, np.inf), frac_cut = (0, np.inf)):
        print("making profiles...")
        alive = self.md['mask_alive']
        props = {
            'function' : 'prof',
            'host_cut' : HFSort.cutToStr(host_cut),
            'sub_cut' : HFSort.cutToStr(sub_cut),
            'frac_cut' : HFSort.cutToStr(frac_cut)
        }

    
        
        for p, m, z in self.getIter(['pdefs', 'm+rdefs', 'zs']):
            # print(f"making profiles for {p},{m}, {z}")
            # get data
            snap_idx = self.getSnap(z)
            mass = self.getMass(m)[snap_idx, :]
            host_mask = self.getMask(p, False)[snap_idx, :]
            sub_mask = ~host_mask
            host_mask &= alive[snap_idx, :]; sub_mask &= alive[snap_idx, :]

            # make mass cuts
            host_cut_mask = (mass >= host_cut[0]) & (mass < host_cut[1])
            sub_cut_mask = (mass >= sub_cut[0]) & (mass < sub_cut[1])

            host_mask &= host_cut_mask
            sub_mask &= sub_cut_mask


            nsubs = np.sum(sub_mask)
            sub_rad = np.zeros(nsubs) - 1
            
            # print("\t number of parents:" + str(np.sum(host_mask)))
            # print("\t number of potential subhalos:" + str(nsubs))


            # temp_sub_idx = np.where(sub_mask)[0]
            host_ids = self.md['id'][snap_idx, host_mask]
            par_ids = self.md[p][snap_idx, sub_mask]
            host_rads = self.getRad(m)[snap_idx, host_mask]
            sub_pos = self.md['x'][snap_idx, sub_mask, :]
            host_pos = self.md['x'][snap_idx, host_mask, :]

            for i in range(nsubs):
                # first test if mass fraction is within cut

                # find the moria idx of the host halo
                pid = par_ids[i]
                host_moria_idx = np.where(host_ids == pid)[0]
                # if host not found, skip
                if len(host_moria_idx) == 0:
                    continue
                
                host_moria_idx = host_moria_idx[0]
                host_mass = mass[host_mask][host_moria_idx]
                frac = mass[sub_mask][i] / host_mass

                if not ((frac >= frac_cut[0]) and (frac < frac_cut[1])):
                    continue

                hrad = host_rads[host_moria_idx]
                hpos = host_pos[host_moria_idx] #cMpc/h
                spos = sub_pos[i] #cMpc/h
                sub_rad[i] = np.sqrt(np.sum((hpos - spos)**2))
                # adjust to kpc, divide by host radius and mult by scale factor
                sub_rad[i] = sub_rad[i] / hrad * 1e3 * (1 / 1 + z)
            
            # print("\tnumber of included radii: " + str(np.sum(sub_rad > 0)))
            hist, edges = np.histogram(sub_rad, bins)
            props.update(self.defaultProps(m, p, z))
            mid = (edges[1:] + edges[:-1]) / 2
            dc = DataContainer([mid, hist])
            dc.update(props)
            self.ds.append(dc)
        
        return





def main(mpath = '', outpath = '') -> DataSort:
    from sparta_tools import moria
    import pickle as pkl
    massdefs = ['M200m_all_spa', 'Morb-all_all_spa', 'Morb-all_orb_spa', 'Mvir_all_spa', 'Mpeak']
    raddefs = []
    for md in massdefs:
        if HFSort.hasRadDef(md):
            raddefs.append(HFSort.mToRstr(md))
    pids = ['parent_id_cat', 'parent_id_orbinf']
    others = ['mask_alive', 'id', 'x']
    redshifts = [0, 0.3, 0.5, 1, 1.5, 2, 4]

    mdata = moria.load(mpath, massdefs + pids + others + raddefs)
        
    mconfig = moria.loadConfig(mpath)
    subdefs = ['none', 'sub_cat', 'sub_orbinf'] + pids
    massdefs += ['Mpeak-orb']
    mptl = mconfig['simulation']['particle_mass']
    hfs = HFSort(mdata, mconfig, massdefs, subdefs, redshifts)
    mbins = np.geomspace(1e10, 2e14, 50)
    mu_bins = np.geomspace(5e-3, 1, 50)
    rbins = np.linspace(0.1, 2, 50)
    hfs.makeProf(rbins)
    hfs.makeHMF(mbins)
    hfs.makeSHMFF(mu_bins)
    hfs.makeSHMFF(mu_bins, host_cut = (1e11, 1e12), sub_cut=(25*mptl, np.inf))
    hfs.makeSF(mbins)
    
    ds = hfs.getDS()

    # default tick/axis/legend
    ds.tickArgs(direction = 'in')
    ds.axisArgs(xscale = 'log', yscale = 'log')
    ds.legendArgs(slc = (-1, -1), frameon = False)
    # set how to display various quantities
    mass_labels = []
    for md in massdefs:
        mass_labels.append(moria.getLatexLabel(md))
    ds.displayAs("full_mdef_str", massdefs, mass_labels)

    z_labels = []
    for z in redshifts:
        z_labels.append("z = %.1f"%z)
    ds.displayAs('redshift', redshifts, z_labels)

    p_labels = ['Overdensity', 'Orbiting']
    ds.displayAs('host_type', pids, p_labels)

    # setting plot properties
    ds.setColors('redshift', 'inferno')
    
    for md in massdefs:
        pargs = {}
        if 'M200m' in md:
            pargs['color'] = 'red'
        elif 'Mvir' in md:
            pargs['color'] = 'black'
        elif 'Morb' in md:
            pargs['color'] = 'blue'
        ds.plotArgs('full_mdef_str', md, pargs)
    
    colors = ['gray', 'red']
    ds.setColors('host_type', colors, pids)
    ds.axisLabelArgs('both', {'fontsize':16})
    ds.rowLabelArgs(pos = [0.05, 0.05], fontsize = 14, ha = 'left', va = 'bottom')
    ds.colLabelArgs(pos = [0.5, 0.95], fontsize = 14, ha = 'center', va = 'top')
    pkl.dump(ds, open(outpath, 'wb'), pkl.HIGHEST_PROTOCOL)


    return ds

if __name__ == '__main__':
    mpath = "/Users/cosinga/sims/sparta_output/L0063_N0256_CBol/moria_trees/moria_tree_orb_defs.hdf5"
    outpath = '/Users/cosinga/code/splots/datasort/massf_sort.pkl'
    main(mpath, outpath)