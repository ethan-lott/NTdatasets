
from inspect import BlockFinder
import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py


class ColorClouds(Dataset):
    """
    -- can load batches from multiple datasets
    -- hdf5 files must have the following information:
        Robs
        RobsMU
        stim: 4-d stimulus: time x nx x ny x color
        block_inds: start and stop of 'trials' (perhaps fixations for now)
        other things: saccades? or should that be in trials? 

    Constructor will take eye position, which for now is an input from data
    generated in the session (not on disk). It should have the length size 
    of the total number of fixations x1.

    Input arguments (details):
        stim_crop = None, should be of form [x1, x2, y1, y2] where each number is the 
            extreme point to be include as an index, e.g. range(x1, x2+1), ... 
    """

    def __init__(self,
        sess_list,
        datadir, 
        # Stim setuup
        num_lags=10, 
        which_stim = None,
        stim_crop = None,  # should be list/array of 4 numbers representing inds of edges
        time_embed = 2,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
        folded_lags=True, 
        luminance_only=True,
        maxT = None,
        eye_config = 2,  # 0 = all, 1, -1, and 2 are options (2 = binocular)
        # other
        include_MUs = False,
        preload = True,
        eyepos = None,
        drift_interval = None,
        device=torch.device('cpu')):
        """Constructor options"""

        self.datadir = datadir
        self.sess_list = sess_list
        self.device = device
        
        self.num_lags = num_lags
        if time_embed == 2:
            assert preload, "Cannot pre-time-embed without preloading."
        self.time_embed = time_embed
        self.preload = preload
        self.stim_crop = stim_crop
        self.folded_lags = folded_lags
        self.eye_config = eye_config
        self.luminance_only = luminance_only
        self.drift_interval = drift_interval

        # get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.mat'), 'r') for sess in self.sess_list]

        # build index map
        self.data_threshold = 6  # how many valid time points required to include saccade?
        self.file_index = [] # which file the block corresponds to
        self.sacc_inds = []
        #self.unit_ids = []
        self.num_units, self.num_sus, self.num_mus = [], [], []
        self.sus = []
        self.NC = 0    
        #self.stim_dims = None
        self.eyepos = eyepos
        self.generate_Xfix = False
        self.num_blks = np.zeros(len(sess_list), dtype=int)
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

        # Data to construct and store in memory
        self.fix_n = []
        #self.sacc_on = []
        #self.sacc_off = []
        self.used_inds = []

        tcount = 0

        if which_stim is None:
            stimname = 'stim'
        else:
            stimname = which_stim
        self.stimname = stimname

        for fnum, fhandle in enumerate(self.fhandles):

            NT, NSUfile = fhandle['Robs'].shape
            NMUfile = fhandle['RobsMU'].shape[1]
            self.num_sus.append(NSUfile)
            self.num_mus.append(NMUfile)
            self.sus = self.sus + list(range(self.NC, self.NC+NSUfile))
            blk_inds = np.array(fhandle['block_inds'], dtype=np.int64)
            blk_inds[:, 0] += -1  # convert to python so range works

            self.channel_mapSU = np.array(fhandle['Robs_probe_ID'], dtype=np.int64)[0, :]
            self.channel_mapMU = np.array(fhandle['RobsMU_probe_ID'], dtype=np.int64)[0, :]
            self.channel_map = np.concatenate((self.channel_mapSU, self.channel_mapMU), axis=0)
            
            NCfile = NSUfile
            if self.include_MUs:
                NCfile += NMUfile
            
            # This will associate each block with each file
            self.block_filemapping += list(np.ones(blk_inds.shape[0], dtype=int)*fnum)
            self.num_blks[fnum]= blk_inds.shape[0]

            #if self.stim_dims is None:
            #if folded_lags:
            self.dims = list(fhandle[stimname].shape[1:4]) + [1]
            #else:
            #    self.dims = list(fhandle[stimname].shape[1:4]) + [1]

            """ EYE configuration """
            if self.eye_config > 0:
                Lpresent = np.array(fhandle['useLeye'], dtype=int)
                Rpresent = np.array(fhandle['useReye'], dtype=int)
                LRpresent = Lpresent + 2*Rpresent

            if luminance_only:
                if self.dims[0] > 1:
                    print("Reducing stimulus channels (%d) to first dimension"%self.dims[0])
                self.dims[0] = 1
            
            if self.time_embed > 0:
                self.dims[3] = self.num_lags

            print('Stim check:', stimname, folded_lags, self.dims)

            #self.unit_ids.append(self.NC + np.asarray(range(NCfile)))
            self.num_units.append(NCfile)
            self.NC += NCfile

            sacc_inds = np.array(fhandle['sacc_inds'], dtype=np.int64)
            sacc_inds[:, 0] += -1  # convert to python so range works

            valid_inds = np.array(fhandle['valid_data'], dtype=np.int64)[:,0]-1  #range(self.NT)  # default -- to be changed at end of init
            
            tcount += NT
            # make larger fix_n, valid_inds, sacc_inds, block_inds as self

        self.NT  = tcount

        # For now let's just debug with one file
        if len(sess_list) > 1:
            print('Warning: currently ignoring multiple files')
        self.used_inds = deepcopy(valid_inds)
        self.sacc_inds = deepcopy(sacc_inds)
        self.LRpresent = LRpresent

        # Go through saccades to establish val_indices and produce saccade timing vector 
        # Note that sacc_ts will be generated even without preload -- small enough that doesnt matter
    #    self.sacc_ts = np.zeros([self.NT, 1], dtype=np.float32)
    #    self.fix_n = np.zeros(self.NT, dtype=np.int64)  # label of which fixation is in each range
    #    for nn in range(self.num_fixations):
    #        print(nn, self.sacc_inds[nn][0], self.sacc_inds[nn][1] )
    #        self.sacc_ts[self.sacc_inds[nn][0]] = 1 
    #        self.fix_n[range(self.sacc_inds[nn][0], self.sacc_inds[nn][1])] = nn
        #self.fix_n = list(self.fix_n)  # better list than numpy

        if self.eyepos is not None:
            assert len(self.eyepos) == self.num_fixations, \
                "eyepos input should have %d fixations."%self.num_fixations

        if preload:
            print("Loading data into memory...")
            self.preload_numpy()

            print('Stim shape', self.stim.shape)
            # Note stim is being represented as full 3-d + 1 tensor (time, channels, NX, NY)
            if self.eyepos is not None:
                # Would want to shift by input eye positions if input here
                print('eye-position shifting not implemented yet')

            if self.stim_crop is not None:
                self.crop_stim()

            if time_embed == 2:
                self.stim = self.time_embedding( self.stim, nlags = num_lags )
            # now stimulus is represented as full 4-d + 1 tensor (time, channels, NX, NY, num_lags)

            # Flatten stim 
            self.stim = np.reshape(self.stim, [self.NT, -1])

            # Have data_filters represend used_inds (in case it gets through)
            unified_df = np.zeros([self.NT, 1], dtype=np.float32)
            unified_df[self.used_inds] = 1.0
            self.dfs *= unified_df

        ### Process experiment to include relevant eye config
        if self.eye_config > 0:  # then want to return experiment part consistent with eye config
            eye_val = np.where(LRpresent == self.eye_config)[0]
            t0, t1 = np.min(eye_val), np.max(eye_val)+1
            assert len(eye_val) == (t1-t0), "EYE CONFIG: non-contiguous blocks of correct eye position"
        else:
            t0, t1 = 0, self.NT

        if maxT is not None:
            t1 = np.minimum( t0+maxT, t1 )
        ts = range(t0, t1)
        print('T-range:', t0, t1)

        # Save for potential future adjustments of signal
        self.startT = t0
        
        if len(ts) < self.NT:
            print("  Trimming experiment %d->%d time points based on eye_config and Tmax"%(self.NT, len(ts)) )

            self.stim = self.stim[ts, :]
            self.robs = self.robs[ts, :]
            self.dfs = self.dfs[ts, :]
            self.LRpresent = self.LRpresent[ts]

            self.NT = len(ts)

            self.used_inds = self.used_inds[(self.used_inds >= t0) & (self.used_inds < t1)] - t0

            # Only keep valid blocks/saccades
            blk_inds = blk_inds - t0 
            blk_inds = blk_inds[ blk_inds[:, 0] >= 0, :]
            blk_inds = blk_inds[ blk_inds[:, 1] < self.NT, :]  
            self.sacc_inds = self.sacc_inds - t0
            self.sacc_inds = self.sacc_inds[ self.sacc_inds[:, 0] >= 0, :]  
            self.sacc_inds = self.sacc_inds[ sacc_inds[:, 1] < self.NT, :]  

        ### Process blocks and fixations/saccades
        for ii in range(blk_inds.shape[0]):
            # note this will be the inds in each file -- file offset must be added for mult files
            self.block_inds.append( np.arange( blk_inds[ii,0], blk_inds[ii,1], dtype=np.int64) )

        self.process_fixations()

        ### Construct drift term if relevant
        if self.drift_interval is None:
            self.Xdrift = None
        else:
            NBL = len(self.block_inds)
            Nanchors = np.ceil(NBL/self.drift_interval).astype(int)
            anchors = np.zeros(Nanchors, dtype=np.int64)
            for bb in range(Nanchors):
                anchors[bb] = self.block_inds[self.drift_interval*bb][0]
            self.Xdrift = utils.design_matrix_drift( self.NT, anchors, zero_left=False, const_right=True)

        # Convert data to tensors
        self.to_tensor(self.device)
        print("Done.")

        # Cross-validation setup
        # Develop default train, validation, and test datasets 
        #self.crossval_setup()
        vblks, trblks = self.fold_sample(len(self.block_inds), 5, random_gen=False)
        self.train_inds = []
        for nn in trblks:
            self.train_inds += list(deepcopy(self.block_inds[nn]))
        self.val_inds = []
        for nn in vblks:
            self.val_inds += list(deepcopy(self.block_inds[nn]))
        self.train_inds = np.array(self.train_inds, dtype=np.int64)
        self.val_inds = np.array(self.val_inds, dtype=np.int64)
    # END ColorClouds.__init__

    def preload_numpy(self):
        """Note this loads stimulus but does not time-embed"""

        NT = self.NT
        ''' 
        Pre-allocate memory for data
        '''
        self.stim = np.zeros( [NT] + self.dims[:3], dtype=np.float32)
        self.robs = np.zeros( [NT, self.NC], dtype=np.float32)
        self.dfs = np.ones( [NT, self.NC], dtype=np.float32)
        #self.eyepos = np.zeros([NT, 2], dtype=np.float32)
        #self.frame_times = np.zeros([NT,1], dtype=np.float32)

        t_counter = 0
        unit_counter = 0
        for ee in range(len(self.fhandles)):
            
            fhandle = self.fhandles[ee]
            sz = fhandle[self.stimname].shape
            inds = range(t_counter, t_counter+sz[0])
            #inds = self.stim_indices[expt][stim]['inds']
            #self.stim[inds, ...] = np.transpose( np.array(fhandle[self.stimname], dtype=np.float32), axes=[0,3,1,2])
            if self.luminance_only:
                self.stim[inds, 0, ...] = np.array(fhandle[self.stimname][:, 0, ...], dtype=np.float32)
            else:
                self.stim[inds, ...] = np.array(fhandle[self.stimname], dtype=np.float32)

            """ Robs and DATAFILTERS"""
            robs_tmp = np.zeros( [len(inds), self.NC], dtype=np.float32 )
            dfs_tmp = np.zeros( [len(inds), self.NC], dtype=np.float32 )
            num_sus = fhandle['Robs'].shape[1]
            units = range(unit_counter, unit_counter+num_sus)
            robs_tmp[:, units] = np.array(fhandle['Robs'], dtype=np.float32)
            dfs_tmp[:, units] = np.array(fhandle['datafilts'], dtype=np.float32)
            if self.include_MUs:
                num_mus = fhandle['RobsMU'].shape[1]
                units = range(unit_counter+num_sus, unit_counter+num_sus+num_mus)
                robs_tmp[:, units] = np.array(fhandle['RobsMU'], dtype=np.float32)
                dfs_tmp[:, units] = np.array(fhandle['datafiltsMU'], dtype=np.float32)
            
            self.robs[inds,:] = deepcopy(robs_tmp)
            self.dfs[inds,:] = deepcopy(dfs_tmp)

            t_counter += sz[0]
            unit_counter += self.num_units[ee]

        # Adjust stimulus since stored as ints
        if np.std(self.stim) > 5: 
            if np.mean(self.stim) > 50:
                self.stim += -127
            self.stim *= 1/128
            print( "Adjusting stimulus read from disk: mean | std = %0.3f | %0.3f"%(np.mean(self.stim), np.std(self.stim)))

        # once stim loaded from disk, make sure stim is named regular
        #self.stimname = 'stim'
    # END .preload_numpy()

    def to_tensor(self, device):
        if isinstance(self.stim, torch.Tensor):
            # then already converted: just moving device
            self.stim = self.stim.to(device)
            self.robs = self.robs.to(device)
            self.dfs = self.dfs.to(device)
            self.fix_n = self.fix_n.to(device)
            if self.Xdrift is not None:
                self.Xdrift = self.Xdrift.to(device)
        else:
            self.stim = torch.tensor(self.stim, dtype=torch.float32, device=device)
            self.robs = torch.tensor(self.robs, dtype=torch.float32, device=device)
            self.dfs = torch.tensor(self.dfs, dtype=torch.float32, device=device)
            self.fix_n = torch.tensor(self.fix_n, dtype=torch.int64, device=device)
            if self.Xdrift is not None:
                self.Xdrift = torch.tensor(self.Xdrift, dtype=torch.float32, device=device)

    def time_embedding( self, stim=None, nlags=None ):
        if nlags is None:
            nlags = self.num_lags
        if stim is None:
            tmp_stim = deepcopy(self.stim)
        else:
            tmp_stim = deepcopy(stim)
        if not isinstance(tmp_stim, np.ndarray):
            tmp_stim = tmp_stim.cpu().numpy()
    
        print("Time embedding...")
        if len(tmp_stim.shape) == 2:
            print( "Time embed: reshaping stimulus ->", self.dims)
            tmp_stim = tmp_stim.reshape([NT] + self.dims)

        NT = stim.shape[0]
        assert self.NT == NT, "TIME EMBEDDING: stim length mismatch"

        tmp_stim = tmp_stim[np.arange(NT)[:,None]-np.arange(nlags), :, :, :]
        if self.folded_lags:
            tmp_stim = np.transpose( tmp_stim, axes=[0,2,1,3,4] ) 
            print("Folded lags: stim-dim = ", self.stim.shape)
        else:
            tmp_stim = np.transpose( tmp_stim, axes=[0,2,3,4,1] )
        return tmp_stim
    # END .time_embedding()

    def wrap_stim( self, vwrap=0, hwrap=0 ):
        """Take existing stimulus and move the whole thing around in horizontal and/or vertical dims,
        including if time_embedded"""

        tmp_stim = deepcopy(self.stim).reshape([self.NT] + self.dims)
        NY = self.dims[2]
        if vwrap > 0:
            self.stim = torch.zeros(tmp_stim.shape, device=tmp_stim.device)
            self.stim[:, :, :, :vwrap, :] = tmp_stim[:, :, :, (NY-vwrap):, :]
            self.stim[:, :, :, vwrap:, :] = tmp_stim[:, :, :, :(NY-vwrap), :]
        elif vwrap < 0:
            self.stim = torch.zeros(tmp_stim.shape, device=tmp_stim.device)
            self.stim[:, :, :, (NY-vwrap):, :] = tmp_stim[:, :, :, :vwrap, :]
            self.stim[:, :, :, :(NY-vwrap), :] = tmp_stim[:, :, :, vwrap:, :]
        
        if hwrap != 0:
            print("Horizontal wraps not implemented yet, since they are probably useless.")

        self.stim = self.stim.reshape( [self.NT, -1] )
    # END .stim_wrap()

    def crop_stim( self, stim_crop=None ):
        """Crop stimulus and change relevant variables"""
        if stim_crop is None:
            stim_crop = self.stim_crop
        else:
            self.stim_crop = stim_crop 
        assert len(stim_crop) == 4, "stim_crop must be of form: [x1, x2, y1, y2]"

        if len(self.stim.shape) != 5:
            self.stim = self.stim.reshape([self.NT] + self.dims)
            #print('  CROP: reshaping stim')
        #stim_crop = np.array(stim_crop, dtype=np.int64) # make sure array
        xs = np.arange(stim_crop[0], stim_crop[1]+1)
        ys = np.arange(stim_crop[2], stim_crop[3]+1)
        self.stim = self.stim[:, :, :, ys, :][:, :, xs, :, :]
        print("  CROP: New stim size: %d x %d"%(len(xs), len(ys)))
        self.dims[1] = len(xs)
        self.dims[2] = len(ys)
        self.stim = self.stim.reshape([self.NT, -1])
    # END .crop_stim()

    def process_fixations( self, sacc_in=None ):
        """Processes fixation informatiom from dataset, but also allows new saccade detection
        to be input and put in the right format within the dataset (main use)"""
        if sacc_in is None:
            sacc_in = self.sacc_inds[:, 0]

        fix_n = np.zeros(self.NT, dtype=np.int64) 
        fix_count = 0
        for ii in range(len(self.block_inds)):
            # note this will be the inds in each file -- file offset must be added for mult files
            #self.block_inds.append(np.arange( blk_inds[ii,0], blk_inds[ii,1], dtype=np.int64))

            # Parse fixation numbers within block
            rel_saccs = np.where((sacc_in > self.block_inds[ii][0]+6) & (sacc_in < self.block_inds[ii][-1]))[0]

            tfix = self.block_inds[ii][0]  # Beginning of fixation by definition
            for mm in range(len(rel_saccs)):
                fix_count += 1
                # Range goes to beginning of next fixation (note no gap)
                fix_n[ range(tfix, sacc_in[rel_saccs[mm]]) ] = fix_count
                tfix = sacc_in[rel_saccs[mm]] + 6
            # Put in last (or only) fixation number
            if tfix < self.block_inds[ii][-1]:
                fix_count += 1
                fix_n[ range(tfix, self.block_inds[ii][-1]) ] = fix_count

        # Determine whether to be numpy or tensor
        if isinstance(self.stim, torch.Tensor):
            self.fix_n = torch.tensor(fix_n, dtype=torch.int64, device=self.stim.device)
        else:
            self.fix_n = fix_n
    # END: ColorClouds.process_fixations()

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

    def shift_stim(self, pos_shifts, metrics=None, metric_threshold = 1, ts_thresh=8 ):
        """Shift stimulus given standard shifting input (TBD)"""
        NX = self.dims[1]
        nlags = self.dims[3]
        re_stim = deepcopy(self.stim).reshape([-1, NX, NX, nlags])[:,:,:,0]
        fix_n = np.array(self.fix_n, dtype=np.int64)
        NF = np.max(fix_n)
        NTtmp = re_stim.shape[0]
        
        #if xyflip:
        #    sh0 = -(pos_shifts[:,1]-self.dims[2]//2)
        #    sh1 = -(pos_shifts[:,0]-self.dims[1]//2)
        #else:
        sh0 = -(pos_shifts[:,0]-self.dims[1]//2)
        sh1 = -(pos_shifts[:,1]-self.dims[2]//2)

        sp_stim = deepcopy(re_stim)
        if metrics is not None:
            val_fix = metrics > metric_threshold
            print("  Applied metric threshold: %d/%d"%(np.sum(val_fix), NF))
        else:
            val_fix = np.array([True]*NF)

        for ff in range(NF):
            ts = np.where(fix_n == ff+1)[0]
            #print(ff, len(ts), ts[0], ts[-1])

            if (abs(sh0[ff])+abs(sh1[ff]) > 0) & (len(ts) > ts_thresh) & val_fix[ff]:

                # FIRST SP DIM shift
                sh = int(sh0[ff])
                stim_seg = re_stim[ts,:,:]
                if sh > 0:
                    stim_tmp = torch.zeros([len(ts),NX,NX], dtype=torch.float32)
                    stim_tmp[:,sh:,:] = deepcopy(stim_seg[:,:(-sh),:])
                elif sh < 0:
                    stim_tmp = torch.zeros([len(ts),NX,NX], dtype=torch.float32)
                    stim_tmp[:,:sh,:] = deepcopy(stim_seg[:,(-sh):,:])
                else:
                    stim_tmp = deepcopy(stim_seg)

                # SECOND SP DIM shift
                sh = int(sh1[ff])
                if sh > 0:
                    stim_tmp2 = torch.zeros([len(ts),NX,NX], dtype=torch.float32)
                    stim_tmp2[:,:,sh:] = deepcopy(stim_tmp[:,:, :(-sh)])
                elif sh < 0:
                    stim_tmp2 = torch.zeros([len(ts),NX,NX], dtype=torch.float32)
                    stim_tmp2[:,:,:sh] = deepcopy(stim_tmp[:,:,(-sh):])
                else:
                    stim_tmp2 = deepcopy(stim_tmp)
                    
                sp_stim[ts,:,:] = deepcopy(stim_tmp2)

        # Time-embed
        idx = np.arange(NTtmp)
        laggedstim = sp_stim[np.arange(NTtmp)[:,None]-np.arange(nlags), :, :]
        return np.transpose( laggedstim, axes=[0,2,3,1] )


    def shift_stim_fixation( self, stim, shift):
        """Simple shift by integer (rounded shift) and zero padded. Note that this is not in 
        is in units of number of bars, rather than -1 to +1. It assumes the stim
        has a batch dimension (over a fixation), and shifts the whole stim by the same amount."""
        print('Currently needs to be fixed to work with 2D')
        sh = round(shift)
        shstim = stim.new_zeros(*stim.shape)
        if sh < 0:
            shstim[:, -sh:] = stim[:, :sh]
        elif sh > 0:
            shstim[:, :-sh] = stim[:, sh:]
        else:
            shstim = deepcopy(stim)

        return shstim
    # END .shift_stim_fixation

    def create_valid_indices(self, post_sacc_gap=None):
        """
        This creates self.valid_inds vector that is used for __get_item__ 
        -- Will default to num_lags following each saccade beginning"""

        if post_sacc_gap is None:
            post_sacc_gap = self.num_lags

        # first, throw out all data where all data_filters are zero
        is_valid = np.zeros(self.NT, dtype=np.int64)
        is_valid[torch.sum(self.dfs, axis=1) > 0] = 1
        
        # Now invalid post_sacc_gap following saccades
        for nn in range(self.num_fixations):
            print(self.sacc_inds[nn, :])
            sts = self.sacc_inds[nn, :]
            is_valid[range(sts[0], np.minimum(sts[0]+post_sacc_gap, self.NT))] = 0
        
        #self.valid_inds = list(np.where(is_valid > 0)[0])
        self.valid_inds = np.where(is_valid > 0)[0]
    # END .create_valid_indices

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
        
        if self.preload:

            if self.time_embed == 1:
                print("get_item time embedding not implemented yet")
                # if self.folded_lags:
                #    stim = np.transpose( tmp_stim, axes=[0,2,1,3,4] ) 
                #else:
                #    stim = np.transpose( tmp_stim, axes=[0,2,3,4,1] )
    
            else:
                if len(self.cells_out) == 0:
                    out = {'stim': self.stim[idx, :],
                        'robs': self.robs[idx, :],
                        'dfs': self.dfs[idx, :],
                        'fix_n': self.fix_n[idx]}
                        # missing saccade timing vector -- not specified
                else:
                    assert isinstance(self.cells_out, list), 'cells_out must be a list'
                    robs_tmp =  self.robs[:, self.cells_out]
                    dfs_tmp =  self.dfs[:, self.cells_out]
                    out = {'stim': self.stim[idx, :],
                        'robs': robs_tmp[idx, :],
                        'dfs': dfs_tmp[idx, :],
                        'fix_n': self.fix_n[idx]}
            
        else:
            inds = self.valid_inds[idx]
            stim = []
            robs = []
            dfs = []
            num_dims = self.dims[0]*self.dims[1]*self.dims[2]

            """ Stim """
            # need file handle
            f = 0
            #f = self.file_index[inds]  # problem is this could span across several files

            stim = torch.tensor(self.fhandles[f][stimname][inds,:], dtype=torch.float32)
            # reshape and flatten stim: currently its NT x NX x NY x Nclrs
            stim = stim.permute([0,3,1,2]).reshape([-1, num_dims])
                
            """ Spikes: needs padding so all are B x NC """ 
            robs = torch.tensor(self.fhandles[f]['Robs'][inds,:], dtype=torch.float32)
            if self.include_MUs:
                robs = torch.cat(
                    (robs, torch.tensor(self.fhandles[f]['RobsMU'][inds,:], dtype=torch.float32)), 
                    dim=1)

                """ Datafilters: needs padding like robs """
            dfs = torch.tensor(self.fhandles[f]['DFs'][inds,:], dtype=torch.float32)
            if self.include_MUs:
                dfs = torch.cat(
                    (dfs, torch.tensor(self.fhandles[f]['DFsMU'][inds,:], dtype=torch.float32)),
                    dim=1)

            out = {'stim': stim, 'robs': robs, 'dfs': dfs, 'fix_n': self.fix_n[inds]}

        # Addition whether-or-not preloaded
        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]

        ### THIS IS NOT NEEDED WITH TIME-EMBEDDING: needs to be on fixation-process side...
        # cushion DFs for number of lags (reducing stim)
        #if (self.num_lags > 0) &  ~utils.is_int(idx):
        #    if out['dfs'].shape[0] > self.num_lags:
        #        out['dfs'][:self.num_lags, :] = 0.0
        #    else: 
        #        print( "Warning: requested batch smaller than num_lags %d < %d"%(out['dfs'].shape[0], self.num_lags) )
     
        return out
    # END: CloudDataset.__get_item__

    #@property
    #def NT(self):
    #    return len(self.used_inds)

    def __len__(self):
        return len(self.used_inds)
