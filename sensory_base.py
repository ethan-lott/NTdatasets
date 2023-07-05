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
    """Parent class meant to hold standard variables and functions used by general sensory datasets
    
    General consistent formatting:
    -- self.robs, dfs, and any design matrices are generated as torch vectors on device
    -- stimuli are imported separately as dataset-specific numpy arrays, and but then prepared into 
        self.stim (tensor) by a function self.prepare_stim, which must be overloaded
    -- self.stim_dims gives the dimension of self.stim in 4-dimensional format
    -- all tensors are stored on default device (cpu)

    General book-keeping variables
    -- self.block_inds is empty but must be filled in by specific datasets
    """

    def __init__(self,
        filenames, # this could be single filename or list of filenames, to be processed in specific way
        datadir, 
        # Stim setup
        trial_sample=False,
        num_lags=10,
        time_embed=0,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
        #maxT = None,
        # other
        include_MUs = False,
        preload = True,
        drift_interval = None,
        device=torch.device('cpu')
        ):
        """Constructor options"""

        self.datadir = datadir
        self.filenames = filenames
        self.device = device
        
        self.trial_sample = trial_sample
        self.num_lags = num_lags
        self.stim_dims = None
        self.time_embed = time_embed
        self.preload =preload
        self.drift_interval = drift_interval

        # Assign standard variables
        self.num_units, self.num_SUs, self.num_MUs = [], [], []
        self.SUs = []
        self.NC = 0    
        self.block_inds = []
        self.block_filemapping = []
        self.include_MUs = include_MUs
        self.SUinds = []
        self.MUinds = []
        self.cells_out = []  # can be list to output specific cells in get_item
        self.robs_out = None
        self.dfs_out = None

        self.avRs = None

        # Set up to store default train_, val_, test_inds
        self.test_inds = None
        self.val_inds = None
        self.train_inds = None
        self.used_inds = []
        self.speckled = False
        self.Mtrn, self.Mval = None, None  # Data-filter masks for speckled XV
        self.Mtrn_out, self.Mval_out = None, None  # Data-filter masks for speckled XV
        self.Xdrift = None
        
        # Basic default memory things
        self.stim = []
        self.dfs = []
        self.robs = []
        self.NT = 0
    
        # Additional covariate list
        self.covariates = {}
        self.cov_dims = {}
        # General file i/o -- this is not general, so taking out
        #self.fhandles = [h5py.File(os.path.join(datadir, sess + '.mat'), 'r') for sess in self.sess_list]
            
    # END SensoryBase.__init__

    def add_covariate( self, cov_name=None, cov=None ):
        assert cov_name is not None, "Need cov_name"
        assert cov is not None, "Missing cov"
        if len(cov.shape) == 1:
            cov = cov[:, None]
        if len(cov.shape) > 2:
            dims = cov.shape[1:]
            if len(dims) < 4:
                dims = np.concatenate( (dims, np.ones(4-len(dims))), axis=0 )
            cov = cov.reshape([-1, np.prod(dims)])
        else:
            dims = [1, cov.shape[1], 1, 1]
        NT = cov.shape[0]
        assert self.NT == NT, "Wrong number of time points"

        self.cov_dims[cov_name] = dims
        if isinstance(cov, torch.Tensor):
            self.covariates[cov_name] = deepcopy(cov)
        else:
            self.covariates[cov_name] = torch.tensor(cov, dtype=torch.float32)
    # END SensoryBase.add_covariate()

    def append_covariates( self, out, idx ):
        """Complements __get_item__ to add covariates to existing dictionary"""
        for cov_name, cov in self.covariates.items():
            out[cov_name] = cov[idx, :]
        # Return out, or not?

    def prepare_stim( self ):
        print('Default prepare stimulus method.')

    def set_cells( self, cell_list=None, verbose=True):
        """Set outputs to potentially limit robs/dfs to certain cells 
        This sets cells_out but also constructs efficient data structures"""
        if cell_list is None:
            # Then reset to full list
            self.cells_out = []
            self.robs_out = None
            self.dfs_out = None
            self.Mtrn_out = None
            self.Mval_out = None
            print("  Reset cells_out to full dataset (%d cells)."%self.NC )
        else:
            if not isinstance(cell_list, list):
                if utils.is_int(cell_list):
                    cell_list = [cell_list]
                else:
                    cell_list = list(cell_list)
            assert np.max(np.array(cell_list)) < self.NC, "ERROR: cell_list too high."
            if verbose:
                print("Output set to %d cells"%len(cell_list))
            self.cells_out = cell_list
            self.robs_out = deepcopy(self.robs[:, cell_list])
            self.dfs_out = deepcopy(self.dfs[:, cell_list])
            if self.Mtrn is not None:
                self.Mtrn_out = deepcopy(self.Mtrn[:, cell_list])
                self.Mval_out = deepcopy(self.Mval[:, cell_list])
    # END SensoryBase.set_cells()

    def time_embedding( self, stim=None, nlags=None, verbose=True ):
        """Assume all stim dimensions are flattened into single dimension. 
        Will only act on self.stim if 'stim' argument is left None"""

        if nlags is None:
            nlags = self.num_lags
        if stim is None:
            assert self.stim_dims is not None, "Need to assemble stim before time-embedding."
            tmp_stim = deepcopy(self.stim)
            if self.stim_dims[3] == 1:  # should only time-embed stim by default, but not all the time
                self.stim_dims[3] = nlags
        else:
            if isinstance(stim, np.ndarray):
                tmp_stim = torch.tensor( stim, dtype=torch.float32)
            else:
                tmp_stim = deepcopy(stim)
 
        if verbose:
            print( "  Time embedding..." )
        NT = stim.shape[0]
        original_dims = None
        if len(tmp_stim.shape) != 2:
            original_dims = tmp_stim.shape
            if verbose:
                print( "Time embed: flattening stimulus from", original_dims)
        tmp_stim = tmp_stim.reshape([NT, -1])  # automatically generates 2-dimensional stim

        #assert self.NT == NT, "TIME EMBEDDING: stim length mismatch"

        # Actual time-embedding itself
        tmp_stim = tmp_stim[np.arange(NT)[:,None]-np.arange(nlags), :]
        tmp_stim = torch.permute( tmp_stim, (0,2,1) ).reshape([NT, -1])
        if verbose:
            print( "  Done.")
        return tmp_stim
    # END SensoryBase.time_embedding()

    def construct_drift_design_matrix( self, block_anchors=None):
        """Note this requires self.block_inds, and either uses self.drift_interval or block_anchors"""

        assert self.block_inds is not None, "Need block_inds defined as an internal variable"

        if block_anchors is None:
            NBL = len(self.block_inds)
            if self.drift_interval is None:
                self.Xdrift = None
                return
            block_anchors = np.arange(0, NBL, self.drift_interval)

        Nanchors = len(block_anchors)
        anchors = np.zeros(Nanchors, dtype=np.int64)
        for bb in range(Nanchors):
            anchors[bb] = self.block_inds[block_anchors[bb]][0]
        
        self.anchors = anchors
        self.Xdrift = torch.tensor( 
            self.design_matrix_drift( self.NT, anchors, zero_left=False, const_right=True),
            dtype=torch.float32)
    # END SenspryBase.construct_drift_design_matrix()

    def trial_psths( self, trials=None, R=None ):
        """Computes average firing rate of cells_out at bin-resolution, averaged across trials
        given in block_inds"""

        Ntr = len(self.block_inds)
        assert Ntr > 0, "Cannot compute PSTHs without block_inds established in dataset."

        if len(self.cells_out) > 0:
            ccs = self.cells_out
        else:
            ccs = np.arange(self.NC)
        dfs = self.dfs[:, ccs].detach().numpy()

        if R is None:  #then use [internal] Robs
            R = deepcopy( self.robs[:, ccs].detach().numpy() )  
        if len(R.shape) == 1:
            R = R[:, None]         
        num_psths = R.shape[1]  # otherwise use existing input

        # Compute minimum trial size
        T = len(self.block_inds[0])
        for bb in range(1, Ntr):
            if len(self.block_inds[bb]) > T:
                T = len(self.block_inds[bb])
        psths = np.zeros([T, num_psths])
        df_count = np.zeros([T, num_psths])

        if trials is None:
            trials = np.arange(Ntr)

        if len(trials) > 0:
            for ii in trials:
                psths += R[self.block_inds[ii][:T], :] * dfs[self.block_inds[ii][:T], :]
                df_count += dfs[self.block_inds[ii][:T], :]
            
            psths = np.divide( psths, np.maximum(df_count, 1.0) )

        return psths
    # END SensoryBase.calculate_psths()

    @staticmethod
    def design_matrix_drift( NT, anchors, zero_left=True, zero_right=False, const_left=False, const_right=False, to_plot=False):
        """Produce a design matrix based on continuous data (s) and anchor points for a tent_basis.
        Here s is a continuous variable (e.g., a stimulus) that is function of time -- single dimension --
        and this will generate apply a tent basis set to s with a basis variable for each anchor point. 
        The end anchor points will be one-sided, but these can be dropped by changing "zero_left" and/or
        "zero_right" into "True".

        Inputs: 
            NT: length of design matrix
            anchors: list or array of anchor points for tent-basis set
            zero_left, zero_right: boolean whether to drop the edge bases (default for both is False)
        Outputs:
            X: design matrix that will be NT x the number of anchors left after zeroing out left and right
        """
        anchors = list(anchors)
        if anchors[0] > 0:
            if not const_left:
                anchors = [0] + anchors
        #if anchors[-1] < NT:
        #    anchors = anchors + [NT]
        NA = len(anchors)

        X = np.zeros([NT, NA])
        for aa in range(NA):
            if aa > 0:
                dx = anchors[aa]-anchors[aa-1]
                X[range(anchors[aa-1], anchors[aa]), aa] = np.arange(dx)/dx
            if aa < NA-1:
                dx = anchors[aa+1]-anchors[aa]
                X[range(anchors[aa], anchors[aa+1]), aa] = 1-np.arange(dx)/dx

        if zero_left:
            X = X[:, 1:]
        elif const_left:  # makes constant from first anchor back to origin -- wont work without zero-left
            X[range(anchors[0]), 0] = 1.0

        if const_right:
            X[range(anchors[-1], NT), -1] = 1.0

        if zero_right:
            X = X[:, :-1]

        if to_plot:
            import matplotlib.pyplot as plt
            plt.imshow(X.T, aspect='auto', interpolation='none')
            plt.show()

        return X

    @staticmethod
    def construct_onehot_design_matrix( stim=None, return_categories=False ):
        """the stimulus should be numpy -- not meant to be used with torch currently"""
        assert stim is not None, "Must pass in stimulus"
        assert len(stim.shape) < 3, "Stimulus must be one-dimensional"
        assert isinstance( stim, np.ndarray ), "stim must be a numpy array"

        category_list = np.unique(stim)
        NSTIM = len(category_list)
        assert NSTIM < 50, "Must have less than 50 classifications in one-hot: something wrong?"
        OHmatrix = np.zeros([stim.shape[0], NSTIM], dtype=np.float32)
        for ss in range(NSTIM):
            OHmatrix[stim == category_list[ss], ss] = 1.0
        
        if return_categories:
            return OHmatrix, category_list
        else:
            return OHmatrix
    # END staticmethod.construct_onehot_design_matrix()

    def avrates( self, inds=None ):
        """
        Calculates average firing probability across specified inds (or whole dataset)
        -- Note will respect datafilters
        -- will return precalc value to save time if already stored
        """
        if inds is None:
            inds = range(self.NT)
        if len(self.cells_out) == 0:
            cells = np.arange(self.NC)
        else:
            cells = self.cells_out

        if len(inds) == self.NT:
            # then calculate across whole dataset
            if self.avRs is not None:
                # then precalculated and do not need to do
                return self.avRs[cells]

        # Otherwise calculate across all data
        if self.preload:
            Reff = (self.dfs * self.robs).sum(dim=0).cpu()
            Teff = self.dfs.sum(dim=0).clamp(min=1e-6).cpu()
            return (Reff/Teff)[cells].detach().numpy()
        else:
            print('Still need to implement avRs without preloading')
            return None
    # END .avrates()


    def crossval_setup(self, folds=5, random_gen=False, test_set=False, verbose=False):
        """This sets the cross-validation indices up We can add featuers here. Many ways to do this
        but will stick to some standard for now. It sets the internal indices, which can be read out
        directly or with helper functions. Perhaps helper_functions is the best way....
        
        Inputs: 
            random_gen: whether to pick random fixations for validation or uniformly distributed
            test_set: whether to set aside first an n-fold test set, and then within the rest n-fold train/val sets
        Outputs:
            None: sets internal variables test_inds, train_inds, val_inds
        """
        assert self.used_inds is not None, "Must first specify valid_indices before setting up cross-validation."

        # Reflect block structure
        Nblks = len(self.block_inds)
        val_blk1, tr_blk1 = self.fold_sample(Nblks, folds, random_gen=random_gen)

        if test_set:
            self.test_blks = val_blk1
            val_blk2, tr_blk2 = self.fold_sample(len(tr_blk1), folds, random_gen=random_gen)
            self.val_blks = tr_blk1[val_blk2]
            self.train_blks = tr_blk1[tr_blk2]
        else:
            self.val_blks = val_blk1
            self.train_blks = tr_blk1
            self.test_blks = []

        if verbose:
            print("Partitioned %d fixations total: tr %d, val %d, te %d"
                %(len(self.test_blks)+len(self.train_blks)+len(self.val_blks),len(self.train_blks), len(self.val_blks), len(self.test_blks)))  

        # Now pull indices from each saccade 
        tr_inds, te_inds, val_inds = [], [], []
        for nn in self.train_blks:
            tr_inds += list(deepcopy(self.block_inds[nn]))
        for nn in self.val_blks:
            val_inds += list(deepcopy(self.block_inds[nn]))
        for nn in self.test_blks:
            te_inds += list(deepcopy(self.block_inds[nn]))

        if verbose:
            print( "Pre-valid data indices: tr %d, val %d, te %d" %(len(tr_inds), len(val_inds), len(te_inds)) )

        # Finally intersect with used_inds
        if len(self.used_inds) > 0:
            self.train_inds = np.array(list(set(tr_inds) & set(self.used_inds)))
            self.val_inds = np.array(list(set(val_inds) & set(self.used_inds)))
            self.test_inds = np.array(list(set(te_inds) & set(self.used_inds)))

            if verbose:
                print( "Valid data indices: tr %d, val %d, te %d" %(len(self.train_inds), len(self.val_inds), len(self.test_inds)) )
        else:
            self.train_inds = tr_inds
            self.val_inds = val_inds
            self.test_inds = te_inds
    # END SensoryBase.crossval_setup

    def fold_sample( self, num_items, folds, random_gen=False, which_fold=None):
        """This really should be a general method not associated with self"""
        if random_gen:
            num_val = int(num_items/folds)
            tmp_seq = np.random.permutation(num_items)
            val_items = np.sort(tmp_seq[:num_val])
            rem_items = np.sort(tmp_seq[num_val:])
        else:
            if which_fold is None:
                offset = int(folds//2)
            else:
                offset = which_fold%folds
            val_items = np.arange(offset, num_items, folds, dtype='int32')
            rem_items = np.delete(np.arange(num_items, dtype='int32'), val_items)
        return val_items, rem_items

    def speckledXV_setup( self, folds=5, random_gen=False ):
        """
        Produce data-filter masks for training and XV speckles
        Will be produced for whole dataset, and must be reduced if cells_out used
        """
        Ntr = len(self.block_inds)

        # Choose trials to leave out for each unit
        self.Mval = torch.zeros(self.dfs.shape, dtype=torch.float32)
        self.Mtrn = torch.ones(self.dfs.shape, dtype=torch.float32)
        for cc in range(self.NC):
            ival,_ = self.fold_sample( 
                Ntr, folds=folds, random_gen=random_gen, which_fold=cc%folds)
            for tr in ival:
                self.Mval[self.block_inds[tr], cc] = 1.0
                self.Mtrn[self.block_inds[tr], cc] = 0.0
        if self.cells_out is not None:
            self.Mtrn_out = deepcopy(self.Mtrn[:, self.cells_out])
            self.Mval_out = deepcopy(self.Mval[:, self.cells_out])
    # END SensoryBase.speckledXV_setup
    
    def set_speckledXV(self, val=True, folds=5, random_gen=False):
        self.speckled = val
        if val:
            if self.Mval is None:
                self.speckledXV_setup(folds=folds, random_gen=random_gen) 
            if len(self.cells_out) > 0:
                self.Mval_out = self.Mval[:, self.cells_out]
                self.Mtrn_out = self.Mtrn[:, self.cells_out]
            else:
                self.Mval_out = None
                self.Mtrn_out = None
    # END SensoryBase.set_speckledXV

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
        mempersample = data['stim'].element_size() * data['stim'].nelement() + 2*data['robs'].element_size() * data['robs'].nelement()
    
        mempercell = mempersample * (nquad+1) * (history_size + 1)
        buffer_bytes = buffer*1024**3

        maxsamples = int(free - mempercell*num_cells - buffer_bytes) // mempersample
        print("# samples that can fit on device: {}".format(maxsamples))
        return maxsamples
    # END .get_max_samples

    def assemble_stimulus( self, **kwargs ):
        print("SensoryBase: assemble_stimulus not implemented in class child.")
        return

    def __getitem__(self, idx):
        return {}

    def __len__(self):
        return self.robs.shape[0]

    @staticmethod
    def is_int( val ):
        """returns Boolean as to whether val is one of many types of integers"""
        if isinstance(val, int) or \
            isinstance(val, np.int) or isinstance(val, np.int32) or isinstance(val, np.int64) or \
            (isinstance(val, np.ndarray) and (len(val.shape) == 0)):
            return True
        else:
            return False

    @staticmethod
    def index_to_array( index, max_val ):
        """This converts any for index to dataset, including slices, and plain ints, into numpy array"""

        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step
            if start is None:
                start = 0
            if stop is None:
                stop = max_val
            if step is None:
                step = 1
            return np.arange(start,stop, step)
        elif SensoryBase.is_int(index):
            return [index]
        elif isinstance(index, list):
            return np.array(index, dtype=np.int64)
        return index
