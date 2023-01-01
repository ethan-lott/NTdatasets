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

class binocular_single(SensoryBase):

    def __init__(self, expt_num=None, time_embed=0, num_lags=12, skip_lags=0, **kwargs):
        """
        Inputs: 
            filename: currently the pre-processed matlab file from Dan's old-style format
            
            Inherited (but needed from SensoryBase init):
                datadir, 
                time_embed=2,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
                include_MUs = False,
                drift_interval = None,

            skip_lags: shift stim to throw out early lags
            **kwargs: non-dataset specific arguments that get passed into SensoryBase
            """

        assert expt_num is not None, "Binocular experiment number needed (expt_n)."
        filename = 'B2Sexpt'+ str(expt_num) + '.mat'

        # call parent constructor
        super().__init__(
            filename, 
            num_lags=num_lags, time_embed=time_embed,
            **kwargs)
        print( "Loading", self.datadir + filename)

        # Store stimulus trimmed to 36 - 36 binocular configuration
        stim_trim = np.concatenate( (np.arange(3,39), np.arange(45,81)))
        Bmatdat = sio.loadmat(self.datadir+filename)
        self.Bstim = Bmatdat['stim'][:, stim_trim]
        self.dims = [1, 72, 1, 1]
        # Note Bstim is stored as numpy

        # Responses
        RobsSU = Bmatdat['RobsSU']
        dfsSU = Bmatdat['SUdata_filter']
        self.NT, self.numSUs = RobsSU.shape

        RobsMU = Bmatdat['RobsMU']
        self.numMUs = RobsMU.shape[1]
    
        if self.include_MUs:
            self.NC = self.numSUs + self.numMUs

            dfsMU = np.ones( RobsMU.shape, dtype=np.float32) 

            self.robs = torch.tensor(
                np.concatenate( (RobsSU, RobsMU), axis=1 ),
                dtype=torch.float32 )
            self.dfs = torch.tensor( 
                np.concatenate( (dfsSU, dfsMU), axis=1 ),
                dtype=torch.float32 )
        else:
            self.NC = self.numSUs
            self.robs = torch.tensor(RobsSU, dtype=torch.float32 )
            self.dfs = torch.tensor(dfsSU, dtype=torch.float32 )

        # used_inds and XV
        used_inds = np.add(np.transpose(Bmatdat['used_inds'])[0,:], -1) # note adjustment for python v matlab indexing
        self.Ui_analog = Bmatdat['Ui_analog'][:,0]  # these are automaticall in register
        self.XiA_analog = Bmatdat['XiA_analog'][:,0]
        self.XiB_analog = Bmatdat['XiB_analog'][:,0]
        # two cross-validation datasets -- for now combine
        self.Xi_analog = self.XiA_analog+self.XiB_analog  # since they are non-overlapping, will make 1 in both places

        # Derive full-dataset Ui and Xi from analog values
        self.used_inds = used_inds
        self.train_inds = np.intersect1d(used_inds, np.where(self.Ui_analog > 0)[0])
        self.val_inds = np.intersect1d(used_inds, np.where(self.Xi_analog > 0)[0])

        dispt_raw = Bmatdat['all_disps'][:,0]
        # this has the actual disparity values, which are at the resolution of single bars, and centered around the neurons
        # disparity (sometime shifted to drive neurons well)
        # Sometimes a slightly disparity is used, so it helps to round the values at some resolution
        self.dispt = np.round(dispt_raw*100)/100
        self.frs = Bmatdat['all_frs'][:,0]
        self.corrt = Bmatdat['all_corrs'][:,0]
        # Make dispt consistent with corrt (early experiments had dispt labeled incorrectly)
        corr_funny = np.where((self.corrt == 0) & (self.dispt != -1005))[0]
        if len(corr_funny) > 0:
            print( "Warning: %d indices have corr=0 but labeled disparity."%len(corr_funny) )
            self.dispt[corr_funny] = -1005

        self.disp_list = np.unique(self.dispt)
        # where it is -1009 this corresponds to a blank frame
        # where it is -1005 this corresponds to uncorrelated images between the eyes

        if Bmatdat['rep_inds'] is None:
            #rep_inds = [None]*numSUs
            rep_inds = None
        elif len(Bmatdat['rep_inds'][0][0]) < 10:
            rep_inds = None
        else:
            rep_inds = []
            for cc in range(self.numSUs):
                rep_inds.append( np.add(Bmatdat['rep_inds'][0][cc], -1) ) 
        self.rep_inds = rep_inds
        print( "Expt %d: %d SUs, %d total units, %d out of %d time points used."%(expt_num, self.numSUs, self.NC, len(used_inds), self.NT))

        # If number of lags entered, then prepare stimulus
        if time_embed > 0:
            self.prepare_stim( time_embed=time_embed, skip_lags=skip_lags, num_lags=num_lags)
    # END binocular_single.__init__

    def prepare_stim( self, time_embed=0, skip_lags=None, num_lags=None ):

        if skip_lags is not None:  
            self.skip_lags = skip_lags
            
        # Shift stimulus by skip_lags (note this was prev multiplied by DF so will be valid)
        stim = deepcopy(self.Bstim)
        assert self.skip_lags >= 0, "Negative skip_lags does not make sense"
        if self.skip_lags > 0:
            stim[self.skip_lags:, :] = deepcopy( stim[:-self.skip_lags, :] )
            stim[:self.skip_lags, :] = 0.0

        self.stim_dims = deepcopy(self.dims)
        if time_embed == 0:
            self.stim = torch.tensor( self.Bstim, dtype=torch.float32 )
        else:
            if num_lags is None:
                # then read from dataset (already set):
                num_lags = self.num_lags
        self.stim = self.time_embedding( stim=stim, nlags=num_lags )
        # This will return a torch-tensor
        self.stim_dims[3] = num_lags
    # END binocular_single.prepare_stim()

    def __getitem__(self, idx):

        if len(self.cells_out) == 0:
            out = {
                'stim': self.stim[idx, :], 
                'robs': self.robs[idx, :],
                'dfs': self.dfs[idx, :]}
            #if self.speckled:
            #    out['Mval'] = self.Mval[idx, :]
            #    out['Mtrn'] = self.Mtrn[idx, :]
        else:
            robs_tmp =  self.robs[:, self.cells_out]
            dfs_tmp =  self.dfs[:, self.cells_out]
            out = {
                    'stim': self.stim[idx, :], 
                    'robs': robs_tmp[idx, :],
                    'dfs': dfs_tmp[idx, :]}

        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]

        #if len(self.covariates) > 0:
        #   self.append_covariates( out, idx)

        return out
    # END binocular_single.__getitem()
    