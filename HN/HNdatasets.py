import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py
from NTdatasets.sensory_base import SensoryBase

class HNdataset(SensoryBase):

    def __init__(self, filename=None, which_stim='left', skip_lags=2, **kwargs):
        """
        Inputs: 
            filename: currently the pre-processed matlab file from Dan's old-style format
            which_stim: which stim is relevant for the neurons in this dataset (default 'left')
            skip_lags: shift stim to throw out early lags
            **kwargs: non-dataset specific arguments that get passed into SensoryBase
            """

        # call parent constructor
        super().__init__(filename, **kwargs)
        print(self.datadir, filename)
        matdat = sio.loadmat(self.datadir+filename)
        print('Loaded ' + filename)
        #matdat = sio.loadmat('Data/'+exptname+'py.mat')
        self.disp_list = matdat['disp_list'][:,0]
        self.stimlist = np.unique(matdat['stimL'])
        self.Nstim = len(self.stimlist)

        self.TRcued = matdat['cued'][:,0] # Ntr 
        self.TRchoice = matdat['choice'][:,0] # Ntr 
        self.TRsignal = matdat['signal']  # Ntr x 2 (sorted by RF)
        self.TRstrength = matdat['strength']  # Ntr x 2 (sorted by RF)
        #self.TRstim = matdat['cued_stim']  # Ntr x 4 (sorted by cued, then uncued)
        # Detect disparities used for decision (indexed by stimulus number)
        #decision_stims = np.where(matdat['disp_list'] == np.unique(matdat['cued_stim'][:,0]))[0]

        self.TRstim = np.multiply(self.TRsignal, self.TRstrength)  # stim strength and direction combined
        
        ### Process neural data
        self.robs = torch.tensor( matdat['Robs'], dtype=torch.float32 )
        self.NT, self.NC = self.robs.shape
        # Make datafilters
        self.dfs = torch.zeros( [self.NT, self.NC], dtype=torch.float32 )
        self.used_inds = matdat['used_inds'][:,0].astype(np.int64) - 1
        self.dfs[self.used_inds, :] = 1.0
        #modvars = matdat['moduvar']

        # High resolution stimuli: note these are already one-hot matrices
        self.stimL = matdat['stimL']
        self.stimR = matdat['stimR']

        # Saccade info
        self.Xsacc = torch.tensor( matdat['Xsacc'], dtype=torch.float32 )
        self.Xadapt = None
        self.ACinput = None  # autoencoder input
        #saccdirs = matdat['sacc_dirs']

        # Make block_inds
        blks = matdat['blks']
        self.Ntr = blks.shape[0]
        self.Nframes = np.min(np.diff(blks))
        for bb in range(self.Ntr):
            self.block_inds.append( np.arange(blks[bb,0]-1, blks[bb,1], dtype=np.int64) )
            # Take out the first num_lags part of each data-filter
            self.dfs[np.arange(blks[bb,0]-1, blks[bb,0]+np.maximum(20,self.num_lags+1)), :] = 0.0

        self.CHnames = [None]*self.NC
        for cc in range(self.NC):
            self.CHnames[cc] = matdat['CHnames'][0][cc][0]
        #expt_info = {'exptname':filename, 'CHnames': CHname, 'blks':blks, 'dec_stims': decision_stims, 
        #            'DispList': dislist, 'StimList': stimlist, #'Xsacc': Xsacc, 'sacc_dirs': saccdirs, 
        #            'stimL': stimL, 'stimR':stimR, 'Robs':Robs, 'used_inds': used_inds}
    
        twin = np.arange(25,self.Nframes, dtype=np.int64)
        self.Rtr = np.zeros([self.Ntr, self.NC], dtype='float32')
        for ii in range(self.Ntr):
            self.Rtr[ii,:] = torch.sum(self.robs[twin+blks[ii,0], :], axis=0)

        print("%d frames, %d units, %d trials with %d frames each"%(self.NT, self.NC, self.Ntr, self.Nframes))
    
        # Generate cross-validation
        use_random = False
        # Cued and uncued trials
        trC = np.where(self.TRcued > 0)[0]
        trU = np.where(self.TRcued < 0)[0]
        # zero-strength trials
        tr0 = np.where(self.TRstrength[:,0] == 0)[0]
        # sort by cued/uncued
        tr0C = np.where((self.TRstrength[:,0] == 0) & (self.TRcued > 0))[0]
        tr0U = np.where((self.TRstrength[:,0] == 0) & (self.TRcued < 0))[0]
        # for purposes of cross-validation, do the same for non-zero-strength trials
        trXC = np.where((self.TRstrength[:,0] != 0) & (self.TRcued > 0))[0]
        trXU = np.where((self.TRstrength[:,0] != 0) & (self.TRcued < 0))[0]

        # Assign train and test indices sampled evenly from each subgroup (note using default 4-fold)
        Ut0C, Xt0C = self.train_test_assign( tr0C, use_random=use_random )
        Ut0U, Xt0U = self.train_test_assign( tr0U, use_random=use_random )
        UtXC, XtXC = self.train_test_assign( trXC, use_random=use_random )
        UtXU, XtXU = self.train_test_assign( trXU, use_random=use_random )

        # Putting together for larger groups
        Ut0 = np.sort( np.concatenate( (Ut0C, Ut0U), axis=0 ) )
        Xt0 = np.sort( np.concatenate( (Xt0C, Xt0U), axis=0 ) )
        UtC = np.sort( np.concatenate( (Ut0C, UtXC), axis=0 ) )
        XtC = np.sort( np.concatenate( (Xt0C, XtXC), axis=0 ) )
        UtU = np.sort( np.concatenate( (Ut0U, UtXU), axis=0 ) )
        XtU = np.sort( np.concatenate( (Xt0U, XtXU), axis=0 ) )

        Ut = np.sort( np.concatenate( (Ut0, UtXC, UtXU), axis=0 ) )
        Xt = np.sort( np.concatenate( (Xt0, XtXC, XtXU), axis=0 ) )
        
        self.trs = {'c':trC, 'u':trU, '0':tr0, '0c': tr0C, '0u': tr0U}
        self.Utr = {'all':Ut, '0':Ut0, 'c':UtC, 'u':UtU, '0c':Ut0C, '0u':Ut0U}
        self.Xtr = {'all':Xt, '0':Xt0, 'c':XtC, 'u':XtU, '0c':Xt0C, '0u':Xt0U}

        self.train_inds, self.val_inds = [], []
        for tr in Ut:
            self.train_inds = np.concatenate( 
                (self.train_inds, self.block_inds[tr]), axis=0 ).astype(np.int64)
        for tr in Xt:
            self.val_inds = np.concatenate( 
                (self.val_inds, self.block_inds[tr]), axis=0 ).astype(np.int64)

        # Figure out minimum and max trial size, and then make function that fits anywhere
        self.min_trial_size = len(self.block_inds[0])
        self.max_trial_size = len(self.block_inds[0])
        for ii in range(self.Ntr):
            if len(self.block_inds[ii]) < self.min_trial_size:
                self.min_trial_size = len(self.block_inds[ii])
            elif len(self.block_inds[ii]) > self.max_trial_size:
                self.max_trial_size = len(self.block_inds[ii])

        # Additional processing check
        # Cued and uncued stim
        #Cstim = np.multiply(TRstim[:,1], np.sign(TRstim[:,0])) # Cued stim
        #Ustim = np.multiply(TRstim[:,3], np.sign(TRstim[:,2]))  # Uncued stim
        #f_far = np.zeros([Nstim,2])
        #for nn in range(Nstim):
        #    tr1 = np.where(Cstim == stimlist[nn])[0]
        #    tr2 = np.where(Ustim == stimlist[nn])[0]
        #    f_far[nn,0] = np.sum(TRchoice[tr1] > 0)/len(tr1)
        #    f_far[nn,1] = np.sum(TRchoice[tr2] > 0)/len(tr2)

        # Prepare stimulus using input argument 'which_stim'
        self.prepare_stim( which_stim=which_stim, skip_lags=skip_lags )

        # Make drift-design matrix using anchor points at each cycle
        cued_transitions = np.where(abs(np.diff(self.TRcued)) > 0)[0]
        anchors = [0] + list(cued_transitions[range(1,len(cued_transitions), 2)])
        self.construct_drift_design_matrix(block_anchors = anchors) 
    # END HNdata.__init__()

    def prepare_stim( self, which_stim='left', skip_lags=None, num_lags=None ):

        if skip_lags is not None:  
            self.skip_lags = skip_lags
        # otherwise will use already set value
            
        if which_stim in ['left', 'L', 'Left']:
            stim = torch.tensor( self.stimL, dtype=torch.float32 )
        else:
            stim = torch.tensor( self.stimR, dtype=torch.float32 )

        # Zero out invalid time points (disp=0) before time embedding
        df_generic = torch.zeros( stim.shape, dtype=torch.float32 )
        df_generic[self.used_inds, :] = 1.0
        stim = stim * df_generic

        self.stim_dims = [1, stim.shape[1], 1, 1]  # Put one-hot on first spatial dimension

        # Shift stimulus by skip_lags (note this was prev multiplied by DF so will be valid)
        if self.skip_lags > 0:
            stim[self.skip_lags:, :] = deepcopy( stim[:-self.skip_lags, :] )
            stim[:self.skip_lags, :] = 0.0
        elif self.skip_lags < 0:
            print("Currently cannot use negative skip_lags, and doesnt make sense anyway")
            self.skip_lags = 0

        if num_lags is None:
            # then read from dataset (already set):
            num_lags = self.num_lags

        self.stim = self.time_embedding( stim=stim, nlags=num_lags )
        # This will return a torch-tensor
    # END HNdata.prepare_stim()

    def construct_Xadapt( self, tent_spacing=12, cueduncued=False ):
        """Constructs adaptation-within-trial tent function
        Inputs: 
            num_tents: default 11
            cueduncued: whether to fit separate kernels to cued/uncued
            """
        # automatically wont have any anchors past min_trial_size
        anchors = np.arange(0, self.min_trial_size, tent_spacing) 
        # Generate master tent_basis
        trial_tents = self.design_matrix_drift(
            self.max_trial_size, anchors, zero_left=False, zero_right=True, const_right=False)
        num_tents = trial_tents.shape[1]

        if cueduncued:
            self.Xadapt = torch.zeros((self.NT, 2*num_tents), dtype=torch.float32)
        else:
            self.Xadapt = torch.zeros((self.NT, num_tents), dtype=torch.float32)

        for tr in range(self.Ntr):
            L = len(self.block_inds[tr])
            if cueduncued:
                tmp = torch.zeros([L, 2*num_tents], dtype=torch.float32) 
                if self.TRcued[tr] < 0:
                    tmp[:, range(num_tents, 2*num_tents)] = torch.tensor(trial_tents[:L, :], dtype=torch.float32)
                else:
                    tmp[:, range(num_tents)] = torch.tensor(trial_tents[:L, :], dtype=torch.float32)
                self.Xadapt[self.block_inds[tr], :] = deepcopy(tmp)
            else:
                self.Xadapt[self.block_inds[tr], :] = torch.tensor(trial_tents[:L, :], dtype=torch.float32)
    # END HNdataset.construct_Xadapt()

    def autoencoder_design_matrix( self, pre_win=0, post_win=0, blank=0, cells=None ):
        """Makes auto-encoder input using windows described above, and including the
        chosen cells. Will put as additional covariate "ACinput" in __get_item__
        Inputs:
            pre_win: how many time steps to include before origin
            post_win: how many time steps to include after origin
            blank: how many time steps to blank in each direction, including origin
            """

        if cells is None:
            cells = np.arange(self.NC)
        Rraw = deepcopy(self.robs[:, cells])
        self.ACinput = torch.zeros(Rraw.shape, dtype=torch.float32)
        nsteps = 0
        if blank == 0:
            self.ACinput += Rraw
            nsteps = 1
        for ii in range(1, (pre_win+1)):
            self.ACinput[ii:, :] += Rraw[:(-ii), :]
            nsteps += 1
        for ii in range(1, (post_win+1)):
            self.ACinput[:(-ii), :] += Rraw[ii:, :]
            nsteps += 1
        assert nsteps > 0, "autoencoder design: invalid parameters"
        self.ACinput *= 1.0/nsteps
        self.ACinput *= self.dfs[:, cells]
    # END autoencoder_design_matrix

    # Moved to SensoryBase
    #def trial_psths( self, trials=None, R=None ):
    #    """Computes average firing rate of cells_out at bin-resolution"""
    #
    #    if R is None:  #then use [internal] Robs
    #        if len(self.cells_out) > 0:
    #            ccs = self.cells_out
    #        else:
    #            ccs = np.arange(self.NC)
    #        R = deepcopy( self.robs[:, ccs].detach().numpy() )  
    #    if len(R.shape) == 1:
    #        R = R[:, None]         
    #    num_psths = R.shape[1]  # otherwise use existing input
    #
    #    T = self.min_trial_size
    #    psths = np.zeros([T, num_psths])
    #
    #    if trials is None:
    #        trials = np.arange(self.Ntr)
    #
    #    if len(trials) > 0:
    #        for ii in trials:
    #            psths += R[self.block_inds[ii][:T]]
    #        psths *= 1.0/len(trials)
    #
    #    return psths
    # END HNdataset.calculate_psths()

    @staticmethod
    def train_test_assign( trial_ns, fold=4, use_random=True ):  # this should be a static function
        num_tr = len(trial_ns)
        if use_random:
            permu = np.random.permutation(num_tr)
            xtr = np.sort( trial_ns[ permu[range(np.floor(num_tr/fold).astype(int))] ]) 
            utr = np.sort( trial_ns[ permu[range(np.floor(num_tr/fold).astype(int), num_tr)] ])
        else:
            xlist = np.arange(fold//2, num_tr, fold, dtype='int32')
            ulist = np.setdiff1d(np.arange(num_tr), xlist)
            xtr = trial_ns[xlist]
            utr = trial_ns[ulist]
        return utr, xtr
    # END HNdata.train_test_assign()

    @staticmethod
    def channel_list_scrub( fnames, subset=None, display_names=True ):  # This should also be a static function
        chnames = []
        if subset is None:
            subset = list(range(len(fnames)))
        for nn in subset:
            fn = fnames[nn]
            a = fn.find('c')   # finds the 'ch'
            b = fn.find('s')-1  # finds the 'sort'
            chn = deepcopy(fn[a:b])
            chnames.append(chn)
        if display_names:
            print(chnames)
        return chnames

    def __getitem__(self, idx):

        if len(self.cells_out) == 0:
            out = {'stim': self.stim[idx, :],
                'robs': self.robs[idx, :],
                'dfs': self.dfs[idx, :]}
            if self.speckled:
                out['Mval'] = self.Mval[idx, :]
                out['Mtrn'] = self.Mtrn[idx, :]

        else:
            assert isinstance(self.cells_out, list), 'cells_out must be a list'
            robs_tmp =  self.robs[:, self.cells_out]
            dfs_tmp =  self.dfs[:, self.cells_out]
            out = {'stim': self.stim[idx, :],
                'robs': robs_tmp[idx, :],
                'dfs': dfs_tmp[idx, :]}
            if self.speckled:
                M1tmp = self.Mval[:, self.cells_out]
                M2tmp = self.Mtrn[:, self.cells_out]
                out['Mval'] = M1tmp[idx, :]
                out['Mtrn'] = M2tmp[idx, :]
            
        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]

        if self.Xadapt is not None:
            out['Xadapt'] = self.Xadapt[idx, :]
            
        if self.ACinput is not None:
            out['ACinput'] = self.ACinput[idx, :]

        if len(self.covariates) > 0:
            self.append_covariates( out, idx)

        return out
    # END HNdata.__getitem()
    

class MotionNeural(SensoryBase):

    def __init__(self, filename=None, num_lags=30, tr_gap=10, **kwargs):
        """
        Inputs: 
            filename: currently the pre-processed matlab file from Dan's old-style format
            datadir: directory for data (goes directly into SensoryBase)
            **kwargs: non-dataset specific arguments that get passed into SensoryBase
            """

        # Call parent constructor
        super().__init__(filename, **kwargs)
        matdat = sio.loadmat(self.datadir+filename)
        print('Loaded ' + filename)

        # Essential variables
        self.robs = torch.tensor( matdat['robs'], dtype=torch.float32)
        self.NT = len(self.robs)
        self.dfs = torch.tensor( matdat['rf_criteria'], dtype=torch.float32)
        self.NC = self.robs.shape[1]

        # Use stim onsets for stim timing (starts with black)
        self.stimB = matdat['stimB'][:, 0].astype(np.float32)
        self.stimW = matdat['stimW'][:, 0].astype(np.float32)
        self.trial_stim = matdat['trial_stim'].astype(np.float32)
        self.stimts = np.where(np.diff(self.stimB) > 0)[0] + 1
        self.flash_stim = torch.zeros([self.NT, 1], dtype=torch.float32)
        self.flash_stim[self.stimts] = 1.0
        self.dims = [1, 1, 1, 1]
        self.Xadapt = None

        # Make trial block and classify trials
        blocks = matdat['blocks']-1  # Convert to python indexing
        trial_type_t = matdat['trial_type'][:, 0]
        Ntr = len(blocks)
        self.trial_type = np.zeros(Ntr, dtype=np.int64) # 1=free viewing, 2=fixation
        for bb in range(Ntr):
            if bb < Ntr-1:
                self.block_inds.append( np.arange(blocks[bb], blocks[bb+1]) )
            else:
                self.block_inds.append( np.arange(blocks[bb], self.NT) )
            self.trial_type[bb] = np.median(trial_type_t[self.block_inds[bb]])

        # Mask out blinks using data_filters +/ 4 frames around each blink
        self.blinks = matdat['blinks'][:, 0].astype(np.float32)
        valid_data = 1-self.blinks
        blink_onsets = np.where(np.diff(self.blinks) > 0)[0] + 1
        blink_offsets = np.where(np.diff(self.blinks) > 0)[0] + 1
        for ii in range(len(blink_onsets)):
            valid_data[range( np.maximum(0, blink_onsets[ii]-4), np.minimum(self.NT, blink_offsets[ii]+4) )] = 0.0
        self.dfs *= valid_data[:, None]
        if tr_gap > 0:
            for bb in range(Ntr):
                self.dfs[self.block_inds[bb][:tr_gap], :] = 0.0

        # Saccades
        self.saccades = matdat['saccades'][:, 0].astype(np.float32)
        saccade_onsets = np.where(np.diff(self.saccades) > 0)[0] + 1
        saccade_offsets = np.where(np.diff(self.saccades) < 0)[0] + 1
        self.sacc_on = torch.zeros( [self.NT, 1], dtype=torch.float32)
        self.sacc_on[saccade_onsets] = 1.0
        self.sacc_off = torch.zeros( [self.NT, 1], dtype=torch.float32)
        self.sacc_off[saccade_offsets] = 1.0

        # Non-essential
        self.eye_pos = matdat['eye_pos']
        self.eye_speed = matdat['eye_speed'][:, 0]
        self.framerate = matdat['framerate'][0, 0]
        # Bharath's used time points
        self.BTtimepts = matdat['timepts2use'][0, :]-1 # Convert to python indexing

        # Make drift term -- need anchors at every other cycle
        transitions = np.where(abs(np.diff(self.trial_type)) > 0)[0] + 1
        drift_anchors = [0] + list(transitions[np.arange(1,len(transitions),2)]) # every other transition
        self.construct_drift_design_matrix( block_anchors=drift_anchors)
    
        self.crossval_setup(test_set=False)

        if num_lags is not None:
            self.assemble_stimulus(num_lags=num_lags)
    # END MotionNeural.__init__

    def assemble_stimulus(self, which_stim='all', use_trial_type=False, num_lags=180 ):
        """
        'which_stim' could be 'trial' (default) using just the first stim onset" or could
        mark each flash (calling 'which_stim' anything else), although this would not implicitly 
        take things like adapation into account
        'use_trial_type' could fit different stim responses in the two conditions to gauge the effects"""

        self.num_lags = num_lags
        if which_stim == 'trial':
            stim = self.trial_stim
        else:
            stim = self.flash_stim.detach().numpy()

        if use_trial_type:
            self.stim_dims = [2, 1, 1, 1]
            stim2 = np.concatenate( (deepcopy(stim), deepcopy(stim)), axis=1)
            for bb in range(len(self.block_inds)):
                stim2[self.block_inds[bb], self.trial_type[bb]-1] = 0.0
                # this will have stim present in 0 for fixation and 1 for free viewing
            self.stim = self.time_embedding( stim=stim2, nlags=num_lags)
        else:
            self.stim_dims = [1, 1, 1, 1]
            self.stim = self.time_embedding( stim=stim, nlags=num_lags)
    # END MotionNeural.assemble_stimulus()

    def construct_Xadapt( self, tent_start=18+8, tent_spacing=20 ):
        """Constructs adaptation-within-trial tent function
        Inputs: 
            """
        # Put first anchor 
        anchors = tent_start + tent_spacing*np.arange(0, 6)  # 5 pulses (ignoring first)

        # Generate master tent_basis
        trial_tents = self.design_matrix_drift(
            len(self.block_inds[0]), anchors, zero_left=True, const_left=True, zero_right=False, const_right=True)
        num_tents = trial_tents.shape[1]

        self.Xadapt = torch.zeros((self.NT, num_tents), dtype=torch.float32)

        for tr in range(len(self.block_inds)):
            L = len(self.block_inds[tr])
            self.Xadapt[self.block_inds[tr], :] = torch.tensor(trial_tents[:L, :], dtype=torch.float32)
    # END MotionNeural.construct_Xadapt()

    def __getitem__(self, idx):

        if len(self.cells_out) == 0:
            out = {
                'stim': self.stim[idx, :],
                'robs': self.robs[idx, :],
                'dfs': self.dfs[idx, :]}
            if self.speckled:
                out['Mval'] = self.Mval[idx, :]
                out['Mtrn'] = self.Mtrn[idx, :]

        else:
            assert isinstance(self.cells_out, list), 'cells_out must be a list'
            robs_tmp =  self.robs[:, self.cells_out]
            dfs_tmp =  self.dfs[:, self.cells_out]
            out = {'stim': self.stim[idx, :],
                'robs': robs_tmp[idx, :],
                'dfs': dfs_tmp[idx, :]}
            if self.speckled:
                M1tmp = self.Mval[:, self.cells_out]
                M2tmp = self.Mtrn[:, self.cells_out]
                out['Mval'] = M1tmp[idx, :]
                out['Mtrn'] = M2tmp[idx, :]
        
        out['sacc_on'] = self.sacc_on[idx, :]
        out['sacc_off'] = self.sacc_off[idx, :]
        out['Xdrift'] = self.Xdrift[idx, :]
        if self.Xadapt is not None:
            out['Xadapt'] = self.Xadapt[idx, :]

        if len(self.covariates) > 0:
            self.append_covariates( out, idx)

        return(out)
    # END MotionNeural.__get_item__()
