import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py

class SensoryBase(Dataset):
    """Parent class meant to hold standard variables and functions used by general
    sensory datasets"""

    def __init__(self,
        filename_list,
        datadir, 
        # Stim setuup
        num_lags=10,
        time_embed = 2,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
        maxT = None,
        # other
        include_MUs = False,
        preload = True,
        drift_interval = None,
        device=torch.device('cpu'),
        **kwargs):
        """Constructor options"""

        self.datadir = datadir
        self.sess_list = filename_list
        self.device = device
        
        self.num_lags = num_lags
        self.time_embed = time_embed
        self.preload = preload
        self.drift_interval = drift_interval

        # Assign standard variables
        self.num_units, self.num_sus, self.num_mus = [], [], []
        self.sus = []
        self.NC = 0    
        self.block_inds = []
        self.block_filemapping = []
        self.include_MUs = include_MUs
        self.SUinds = []
        self.MUinds = []
        self.cells_out = []  # can be list to output specific cells in get_item
        self.avRs = None

        # Set up to store default train_, val_, test_inds
        self.test_inds = None
        self.val_inds = None
        self.train_inds = None
        self.used_inds = []

        # Basic default memory things
        self.stim = []
        self.dfs = []
        self.robs = []
        self.NT = 0
    
        # General file i/o
        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.mat'), 'r') for sess in self.sess_list]

    # END SensoryBase.__init__

    def construct_drift_design_matrix( self, anchors=None):
        """Note this uses self.block_inds currently"""

        assert anchors is None, "Haven't implemented external anchor handling yet"

        if self.drift_interval is None:
            self.Xdrift = None
        else:
            assert self.block_inds is not None, "Need to specify block_inds (currently)"
            NBL = len(self.block_inds)
            Nanchors = np.ceil(NBL/self.drift_interval).astype(int)
            anchors = np.zeros(Nanchors, dtype=np.int64)
            for bb in range(Nanchors):
                anchors[bb] = self.block_inds[self.drift_interval*bb][0]
            self.Xdrift = utils.design_matrix_drift( self.NT, anchors, zero_left=False, const_right=True)
    # END SenspryBase.construct_drift_design_matrix()

    def avrates( self, inds=None ):
        """
        Calculates average firing probability across specified inds (or whole dataset)
        -- Note will respect datafilters
        -- will return precalc value to save time if already stored
        """
        if inds is None:
            inds = range(self.NT)
        if len(inds) == self.NT:
            # then calculate across whole dataset
            if self.avRs is not None:
                # then precalculated and do not need to do
                return self.avRs

        # Otherwise calculate across all data
        if self.preload:
            Reff = (self.dfs * self.robs).sum(dim=0).cpu()
            Teff = self.dfs.sum(dim=0).clamp(min=1e-6).cpu()
            return (Reff/Teff).detach().numpy()
        else:
            print('Still need to implement avRs without preloading')
            return None
    # END .avrates()

    def crossval_setup(self, folds=5, random_gen=False, test_set=True, verbose=False):
        """This sets the cross-validation indices up We can add featuers here. Many ways to do this
        but will stick to some standard for now. It sets the internal indices, which can be read out
        directly or with helper functions. Perhaps helper_functions is the best way....
        
        Inputs: 
            random_gen: whether to pick random fixations for validation or uniformly distributed
            test_set: whether to set aside first an n-fold test set, and then within the rest n-fold train/val sets
        Outputs:
            None: sets internal variables test_inds, train_inds, val_inds
        """
        assert self.valid_inds is not None, "Must first specify valid_indices before setting up cross-validation."

        # Partition data by saccades, and then associate indices with each
        te_fixes, tr_fixes, val_fixes = [], [], []
        for ee in range(len(self.fixation_grouping)):  # Loops across experiments
            fixations = np.array(self.fixation_grouping[ee])  # fixations associated with each experiment
            val_fix1, tr_fix1 = self.fold_sample(len(fixations), folds, random_gen=random_gen)
            if test_set:
                te_fixes += list(fixations[val_fix1])
                val_fix2, tr_fix2 = self.fold_sample(len(tr_fix1), folds, random_gen=random_gen)
                val_fixes += list(fixations[tr_fix1[val_fix2]])
                tr_fixes += list(fixations[tr_fix1[tr_fix2]])
            else:
                val_fixes += list(fixations[val_fix1])
                tr_fixes += list(fixations[tr_fix1])

        if verbose:
            print("Partitioned %d fixations total: tr %d, val %d, te %d"
                %(len(te_fixes)+len(tr_fixes)+len(val_fixes),len(tr_fixes), len(val_fixes), len(te_fixes)))  

        # Now pull  indices from each saccade 
        tr_inds, te_inds, val_inds = [], [], []
        for nn in tr_fixes:
            tr_inds += range(self.sacc_inds[nn][0], self.sacc_inds[nn][1])
        for nn in val_fixes:
            val_inds += range(self.sacc_inds[nn][0], self.sacc_inds[nn][1])
        for nn in te_fixes:
            te_inds += range(self.sacc_inds[nn][0], self.sacc_inds[nn][1])

        if verbose:
            print( "Pre-valid data indices: tr %d, val %d, te %d" %(len(tr_inds), len(val_inds), len(te_inds)) )

        # Finally intersect with valid indices
        self.train_inds = np.array(list(set(tr_inds) & set(self.valid_inds)))
        self.val_inds = np.array(list(set(val_inds) & set(self.valid_inds)))
        self.test_inds = np.array(list(set(te_inds) & set(self.valid_inds)))

        if verbose:
            print( "Valid data indices: tr %d, val %d, te %d" %(len(self.train_inds), len(self.val_inds), len(self.test_inds)) )

    # END MultiDatasetFix.crossval_setup

    def fold_sample( self, num_items, folds, random_gen=False):
        """This really should be a general method not associated with self"""
        if random_gen:
            num_val = int(num_items/folds)
            tmp_seq = np.random.permutation(num_items)
            val_items = np.sort(tmp_seq[:num_val])
            rem_items = np.sort(tmp_seq[num_val:])
        else:
            offset = int(folds//2)
            val_items = np.arange(offset, num_items, folds, dtype='int32')
            rem_items = np.delete(np.arange(num_items, dtype='int32'), val_items)
        return val_items, rem_items

    def get_max_samples(self, gpu_n=0, history_size=1, nquad=0, num_cells=None, buffer=1.2):
        """
        get the maximum number of samples that fit in memory -- for GLM/GQM x LBFGS

        Inputs:
            dataset: the dataset to get the samples from
            device: the device to put the samples on
        """
        if gpu_n == 0:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:1')

        if num_cells is None:
            num_cells = self.NC
        
        t = torch.cuda.get_device_properties(device).total_memory
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        free = t - (a+r)

        data = self[0]
        mempersample = data[self.stimname].element_size() * data[self.stimname].nelement() + 2*data['robs'].element_size() * data['robs'].nelement()
    
        mempercell = mempersample * (nquad+1) * (history_size + 1)
        buffer_bytes = buffer*1024**3

        maxsamples = int(free - mempercell*num_cells - buffer_bytes) // mempersample
        print("# samples that can fit on device: {}".format(maxsamples))
        return maxsamples
    # END .get_max_samples

    def __getitem__(self, idx):
        return {}
    