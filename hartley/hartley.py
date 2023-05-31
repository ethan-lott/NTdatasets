import sys
import os
import h5py
from copy import deepcopy
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

sys.path.insert(0, '/home/elott1/code/')

import NDNT.utils as utils
from NTdatasets.sensory_base import SensoryBase

class HartleyDataset(SensoryBase):
        
    def __init__(self,
                filenames,
                datadir, 
                time_embed=2,               # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
                num_lags=10, 
                include_MUs=True,
                preload=True,
                drift_interval=None,
                device=torch.device('cpu'),
                # Dataset-specitic inputs
                # Stim setup -- if dont want to assemble stimulus: specify all things here for default options
                which_stim=1,            # 'et' or 0, or 1 for lam, but default assemble later
                stim_crop=None,             # should be list/array of 4 numbers representing inds of edges
                luminance_only=False,
                ignore_saccades=True,
                folded_lags=False, 
                binocular=False,          # whether to include separate filters for each eye
                eye_config=0,             # 0 = no subsampling, 1(left), 2(right), and 3(binocular) are options
                maxT=None,
                meta_dims = 4):
    
        super().__init__(
                        filenames=filenames,
                        datadir=datadir, 
                        time_embed=time_embed,
                        num_lags=num_lags, 
                        include_MUs=include_MUs,
                        preload=preload,
                        drift_interval=drift_interval,
                        device=device)
        
        # Stim-specific
        self.stim_crop = None 
        self.folded_lags = folded_lags
        self.eye_config = eye_config
        self.binocular = binocular
        self.luminance_only = luminance_only
        self.generate_Xfix = False
        self.output_separate_eye_stim = False
        self.meta_dims = meta_dims

        self.start_t = 0
        self.drift_interval = drift_interval

        # Get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.mat'), 'r') for sess in self.filenames]
        self.avRs = None

        self.metas = []
        self.fix_n = []
        self.used_inds = []
        self.NT = 0

        # build index map -- exclude variables already set in sensory-base
        self.num_blks = np.zeros(len(filenames), dtype=int)
        self.data_threshold = 6  # how many valid time points required to include saccade?
        self.file_index = [] # which file the block corresponds to
        self.sacc_inds = None
        self.stim_shifts = None

        # Get data from each file
        for fnum, fhandle in enumerate(self.fhandles):

            # Read and store this file's counts
            NT, NSUfile = fhandle['Robs'].shape                                                 # Read timeframe and single unit counts
            NMUfile = fhandle['RobsMU'].shape[1] if len(fhandle['RobsMU'].shape) > 1 else 0     # Read multi-unit count
            NCfile = NSUfile + NMUfile if self.include_MUs else NSUfile                         # Calculate total cell count
            self.num_SUs.append(NSUfile)                                                        # Store SU count
            self.num_MUs.append(NMUfile)                                                        # Store MU count
            self.num_units.append(NCfile)                                                       # Store cell count
            self.SUs = self.SUs + list(range(self.NC, self.NC+NSUfile))                         # Store SU indices
            #self.unit_ids.append(self.NC + np.asarray(range(NCfile)))                           # Store cell indices
            self.NC += NCfile                                                                   # Accumulate total cells
            self.NT += NT                                                                        # Accumulate total timeframes
            

            # Get this file's blocks
            blk_inds = np.array(fhandle['block_inds'], dtype=np.int64)          # Read this file's block indices
            blk_inds[:, 0] -= 1                                                 # Convert to python so range works
            if blk_inds.shape[0] == 2:                                          # Transpose array if old style
                print('WARNING: blk_inds is stored old-style: transposing')
                blk_inds = blk_inds.T
            # self.blockID = np.array(fhandle['blockID'], dtype=np.int64)[:, 0]   # Read and store block IDs

            # Construct channel map
            self.channel_mapSU = np.array(fhandle['Robs_probe_ID'], dtype=np.int64)[0, :]               # Read and store SU probe IDs
            self.channel_rating = np.array(fhandle['Robs_rating'])[0, :]                                # Read and store SU ratings
            if NMUfile > 0:
                self.channel_mapMU = np.array(fhandle['RobsMU_probe_ID'], dtype=np.int64)[0, :]         # Read and store MU probe IDs
                self.channelMU_rating = np.array(fhandle['RobsMU_rating'])[0, :]                        # Read and store MU ratings
                self.channel_map = np.concatenate((self.channel_mapSU, self.channel_mapMU), axis=0)     # Store full channel map
            else:
                self.channel_mapMU = []
                self.channel_map = self.channel_mapSU
            
            # Stimulus information
            self.fix_location = np.array(fhandle['fix_location'])               # Read location of visual fixation
            self.fix_size = np.array(fhandle['fix_size'])                       # Read size of visual fixation
            self.stim_location = np.array(fhandle['stim_location'])             # Read location of laminar stimulus
            self.stim_locationET = np.array(fhandle['ETstim_location'])         # Read location of ET stimulus
            self.stimscale = np.array(fhandle['stimscale'])                     # Read scale of stimulus
            self.stim_pos = None

            # ETtrace information
            self.ETtrace = np.array(fhandle['ETtrace'], dtype=np.float32)       # Read and store ET trace values
            self.ETtraceHR = np.array(fhandle['ETtrace_raw'], dtype=np.float32) # Read and store raw ET trace values

            
            # This will associate each block with each file
            self.block_filemapping += list(np.ones(blk_inds.shape[0], dtype=int)*fnum)
            self.num_blks[fnum] = blk_inds.shape[0]

            self.dims = list(fhandle['stim'].shape[1:4])      # this is basis of data (stimLP and stimET)
            self.dims.append(self.num_lags if self.time_embed > 0 else 1)
            self.stim_dims = None                             # when stim is constructed

            # Adjust dims as necessary
            if luminance_only:
                if self.dims[0] > 1:
                    print(f'Reducing stimulus channels ({self.dims[0]}) to first dimension.')
                self.dims[0] = 1

            if self.binocular:
                self.dims[0] *= 2

            """ EYE configuration """
            if self.eye_config > 0:
                Lpresent = np.array(fhandle['useLeye'], dtype=int)[:,0]
                Rpresent = np.array(fhandle['useReye'], dtype=int)[:,0]
                LRpresent = Lpresent + 2*Rpresent

            if not ignore_saccades:
                sacc_inds = np.array(fhandle['sacc_inds'], dtype=np.int64)
                if len(sacc_inds.shape) > 1:
                    sacc_inds[:, 0] += -1  # convert to python so range works
                else:
                    print('Ignoring sacc_inds. Assuming not valid')
                    sacc_inds = None
                self.sacc_inds = deepcopy(sacc_inds)

            valid_inds = np.array(fhandle['valid_data'], dtype=np.int64)-1  #range(self.NT)  # default -- to be changed at end of init
            if valid_inds.shape[0] == 1:   # Make version-read proof
                valid_inds = valid_inds[0, :]
                print('WARNING: Old-stype valid_inds saved. Transposing')
            else:
                valid_inds = valid_inds[:, 0]

            # make larger fix_n, valid_inds, sacc_inds, block_inds as self

        # For now let's just debug with one file
        if len(filenames) > 1:
            print('Warning: currently ignoring multiple files')
        self.used_inds = deepcopy(valid_inds)
        self.LRpresent = LRpresent

        # Make binocular gain term
        self.binocular_gain = torch.zeros( [len(LRpresent), 2], dtype=torch.float32 )
        self.binocular_gain[self.LRpresent == 1, 0] = 1.0
        self.binocular_gain[self.LRpresent == 2, 1] = 1.0

        # Preload data if specified.
        if preload:
            print("Loading data into memory...")
            self.preload_numpy()

            if which_stim is not None:
                #assert num_lags is not None, "Need to specify num_lags and other stim params"
                self.assemble_stimulus( which_stim=which_stim, 
                    time_embed=time_embed, num_lags=num_lags, stim_crop=stim_crop )

            # Have data_filters represend used_inds (in case it gets through)
            unified_df = np.zeros([self.NT, 1], dtype=np.float32)
            unified_df[self.used_inds] = 1.0
            self.dfs *= unified_df

        ### Process experiment to include relevant eye config
        if self.eye_config > 0:  # then want to return experiment part consistent with eye config
            eye_val = np.where(LRpresent == self.eye_config)[0]
            t0, t1 = np.min(eye_val), np.max(eye_val)+1
            if len(eye_val) < (t1-t0):
                print( "EYE CONFIG WARNING: non-contiguous blocks of correct eye position" )
                t1 = np.where(np.diff(eye_val) > 1)[0][0]+1
                print( "Taking first contiguous block up to %d"%t1)
        else:
            t0, t1 = 0, self.NT

        self.start_t = t0
        if maxT is not None:
            t1 = np.minimum( t0+maxT, t1 )
        ts = range(t0, t1)
        print('T-range:', t0, t1)

        # Save for potential future adjustments of signal
        self.startT = t0
        
        if len(ts) < self.NT:
            print("  Trimming experiment %d->%d time points based on eye_config and Tmax"%(self.NT, len(ts)) )

            self.stimLP = self.stimLP[ts, ...]
            if self.stimET is not None:
                self.stimET = self.stimET[ts, ...]
            self.robs = self.robs[ts, :]
            self.dfs = self.dfs[ts, :]
            self.LRpresent = self.LRpresent[ts]
            self.binocular_gain = self.binocular_gain[ts, :]

            self.NT = len(ts)

            self.used_inds = self.used_inds[(self.used_inds >= t0) & (self.used_inds < t1)] - t0

            # Only keep valid blocks/saccades
            blk_inds = blk_inds - t0 
            blk_inds = blk_inds[ blk_inds[:, 0] >= 0, :]
            blk_inds = blk_inds[ blk_inds[:, 1] <= self.NT, :]
            if self.sacc_inds is not None:
                self.sacc_inds = self.sacc_inds - t0
                self.sacc_inds = self.sacc_inds[ self.sacc_inds[:, 0] >= 0, :]  
                self.sacc_inds = self.sacc_inds[ sacc_inds[:, 1] < self.NT, :]  

        ### Process blocks and fixations/saccades
        for ii in range(blk_inds.shape[0]):
            # note this will be the inds in each file -- file offset must be added for mult files
            self.block_inds.append( np.arange( blk_inds[ii,0], blk_inds[ii,1], dtype=np.int64) )
        # go to end of time range if extends beyond block range
        if blk_inds[ii,1] < self.NT:
            print('Extending final block at ', blk_inds[ii,1], self.NT)
            self.block_inds.append( np.arange(blk_inds[-1,1], self.NT))
            # This is to fix zeroing of last block in fix_n.... (I think?)

        if self.sacc_inds is not None:
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
            self.Xdrift = self.design_matrix_drift( self.NT, anchors, zero_left=False, const_right=True)

        # Write all relevant data (other than stim) to pytorch tensors after organized
        self.to_tensor(self.device)

        # Cross-validation setup
        self.crossval_setup()
    # END ColorClouds.__init__

    # Develop default train, validation, and test datasets 
    def crossval_setup(self):
        vblks, trblks = self.fold_sample(len(self.block_inds), 5, random_gen=False)
        self.train_inds = []
        for nn in trblks:
            self.train_inds += list(deepcopy(self.block_inds[nn]))
        self.val_inds = []
        for nn in vblks:
            self.val_inds += list(deepcopy(self.block_inds[nn]))
        # Eliminate from time point any times when the datafilers are all zero
        # this will include all places where used-inds is zero as well
        self.train_inds = np.array(self.train_inds, dtype=np.int64)
        self.val_inds = np.array(self.val_inds, dtype=np.int64)
    # END .crossval_setup()

    def preload_numpy(self):
        """Note this loads stimulus but does not time-embed"""

        NT = self.NT
        ''' 
        Pre-allocate memory for data
        '''
        self.stimLP = np.zeros( [NT] + self.dims[:3], dtype=np.float32)
        if 'stimET' in self.fhandles[0]:
            # Check to see if 1-d or 2-d eye tracking
            tmp = np.array(self.fhandles[0]['stimET'], dtype=np.float32)[:100, ...]
            if len(tmp.shape) < 4:
                print("  ET stimulus is 1-d bars.")
                self.stimET = np.zeros( [NT, tmp.shape[1]], dtype=np.float32)
            else:
                self.stimET = np.zeros( [NT] + self.dims[:3], dtype=np.float32)
        else:
            self.stimET = None
            print('Missing ETstimulus')

        self.robs = np.zeros( [NT, self.NC], dtype=np.float32)
        self.dfs = np.ones( [NT, self.NC], dtype=np.float32)
        self.metas = np.zeros([NT, self.meta_dims], dtype=np.float32)

        #self.eyepos = np.zeros([NT, 2], dtype=np.float32)
        #self.frame_times = np.zeros([NT,1], dtype=np.float32)

        t_counter = 0
        unit_counter = 0
        for ee in range(len(self.fhandles)):
            
            fhandle = self.fhandles[ee]
            sz = fhandle['stim'].shape
            inds = np.arange(t_counter, t_counter+sz[0], dtype=np.int64)
            #inds = self.stim_indices[expt][stim]['inds']
            #self.stim[inds, ...] = np.transpose( np.array(fhandle[self.stimname], dtype=np.float32), axes=[0,3,1,2])
            if self.binocular:
                Leye = inds[self.LRpresent[inds] != 2]
                Reye = inds[self.LRpresent[inds] != 1]
                #Leye = inds[np.where(self.LRpresent[inds] != 2)[0]]
                #Reye = inds[np.where(self.LRpresent[inds] != 1)[0]]
                #print(len(Leye), len(Reye))
                if self.luminance_only:
                    self.stimLP[Leye, 0, ...] = np.array(fhandle['stim'], dtype=np.float32)[Leye, 0, ...]
                    self.stimLP[Reye, 1, ...] = np.array(fhandle['stim'], dtype=np.float32)[Reye, 0, ...]
                    if self.stimET is not None:
                        self.stimET[Leye, 0, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Leye, 0, ...]
                        self.stimET[Reye, 1, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Reye, 0, ...]
                else:
                    stim_tmp = deepcopy(self.stimLP[:, :3, ...])
                    stim_tmp[Leye, ...] = np.array(fhandle['stim'], dtype=np.float32)[Leye, ...]
                    self.stimLP[:, np.arange(3), ...] = deepcopy(stim_tmp) # this might not be necessary
                    stim_tmp = deepcopy(self.stimLP[:, 3:, ...])
                    stim_tmp[Reye, ...] = np.array(fhandle['stim'], dtype=np.float32)[Reye, ...]
                    self.stimLP[:, np.arange(3,6), ...] = deepcopy(stim_tmp) # this might not be necessary

                    if self.stimET is not None:
                        stim_tmp = deepcopy(self.stimET[:, :3, ...])
                        stim_tmp[Leye, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Leye, ...]
                        self.stimET[:, np.arange(3), ...] = deepcopy(stim_tmp) # this might not be necessary
                        stim_tmp = deepcopy(self.stimET[:, 3:, ...])
                        stim_tmp[Reye, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Reye, ...]
                        self.stimET[:, np.arange(3,6), ...] = deepcopy(stim_tmp) # this might not be necessary
                        #self.stimET[Leye, np.arange(3), ...] = np.array(fhandle['stimET'], dtype=np.float32)[Leye, ...]
                        #self.stimET[Reye, np.arange(3,6), ...] = np.array(fhandle['stimET'], dtype=np.float32)[Reye, ...]
                        
            else:  # Monocular
                if self.luminance_only:
                    self.stimLP[inds, 0, ...] = np.array(fhandle['stim'], dtype=np.float32)[:, 0, ...]
                    if self.stimET is not None:
                        #print(np.array(fhandle['stimET'], dtype=np.float32).shape, self.stimET[inds, 0, ...].shape)
                        self.stimET[inds, 0, ...] = np.array(fhandle['stimET'], dtype=np.float32)[:, 0, ...]
                else:
                    self.stimLP[inds, ...] = np.array(fhandle['stim'], dtype=np.float32)
                    if self.stimET is not None:
                        self.stimET[inds, ...] = np.array(fhandle['stimET'], dtype=np.float32)

            """ Hartley metadata """
            self.metas[inds,:] = np.array(fhandle['hartley_metas'], dtype=np.float32)

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
        if np.std(self.stimLP) > 5: 
            if np.mean(self.stimLP) > 50:
                self.stimLP += -127
                if self.stimET is not None:
                    self.stimET += -127
            self.stimLP *= 1/128
            if self.stimET is not None:
                self.stimET *= 1/128
            print( "Adjusting stimulus read from disk: mean | std = %0.3f | %0.3f"%(np.mean(self.stimLP), np.std(self.stimLP)))
    # END .preload_numpy()

    def to_tensor(self, device):
        if isinstance(self.robs, torch.Tensor):
            # then already converted: just moving device
            self.robs = self.robs.to(device)
            self.dfs = self.dfs.to(device)
            self.fix_n = self.fix_n.to(device)
            if self.Xdrift is not None:
                self.Xdrift = self.Xdrift.to(device)
        else:
            self.robs = torch.tensor(self.robs, dtype=torch.float32, device=device)
            self.dfs = torch.tensor(self.dfs, dtype=torch.float32, device=device)
            self.fix_n = torch.tensor(self.fix_n, dtype=torch.int64, device=device)
            if self.Xdrift is not None:
                self.Xdrift = torch.tensor(self.Xdrift, dtype=torch.float32, device=device)
    # END .to_tensor()

    def is_fixpoint_present( self, boxlim ):
        """Return if any of fixation point is within the box given by top-left to bottom-right corner"""
        if self.fix_location is None:
            return False
        fix_present = True
        if self.stim_location.shape[1] == 1:
            # if there is multiple windows, needs to be manual: so this is the automatic check:
            for dd in range(2):
                if (self.fix_location[dd]-boxlim[dd] <= -self.fix_size):
                    fix_present = False
                if (self.fix_location[dd]-boxlim[dd+2] > self.fix_size):
                    fix_present = False
        return fix_present
    # END .is_fixpoint_present()

    def assemble_stimulus(self,
        which_stim=None, stim_wrap=None, stim_crop=None, # conventional stim: ET=0, lam=1,
        top_corner=None, L=None,  # position of stim
        time_embed=0, num_lags=10,
        luminance_only=False,
        shifts=None, BUF=20, # shift buffer
        shift_times=None, # So that can put partial-shifts over time range in stimulus
        fixdot=0 ):
        """This assembles a stimulus from the raw numpy-stored stimuli into self.stim
        which_stim: determines what stimulus is assembled from 'ET'=0, 'lam'=1, None
            If none, will need top_corner present: can specify with four numbers (top-left, bot-right)
            or just top_corner and L
        which is torch.tensor on default device
        stim_wrap: only works if using 'which_stim', and will be [hwrap, vwrap]"""

        # Delete existing stim and clear cache to prevent memory issues on GPU
        if self.stim is not None:
            del self.stim
            self.stim = None
            torch.cuda.empty_cache()
        num_clr = self.dims[0]
        need2crop = False

        if which_stim is not None:
            assert L is None, "ASSEMBLE_STIMULUS: cannot specify L if using which_stim (i.e. prepackaged stim)"
            L = self.dims[1]
            if not isinstance(which_stim, int):
                if which_stim in ['ET', 'et', 'stimET']:
                    which_stim=0
                else:
                    which_stim=1
            if which_stim == 0:
                print("Stim: using ET stimulus")
                self.stim = torch.tensor( self.stimET, dtype=torch.float32, device=self.device )
                self.stim_pos = self.stim_locationET[:,0]
            else:
                print("Stim: using laminar probe stimulus")
                self.stim = torch.tensor( self.stimLP, dtype=torch.float32, device=self.device )
                self.stim_pos = self.stim_location[:, 0]

        else:
            assert top_corner is not None, "Need top corner if which_stim unspecified"
            # Assemble from combination of ET and laminer probe (NP) stimulus
            if len(top_corner) == 4:
                self.stim_pos = top_corner
                #L = self.stim_pos[2]-self.stim_pos[0]
                assert self.stim_pos[3]-self.stim_pos[1] == self.stim_pos[2]-self.stim_pos[0], "Stim must be square (for now)"
            else:
                if L is None:
                    L = self.dims[1]
                self.stim_pos = [top_corner[0], top_corner[1], top_corner[0]+L, top_corner[1]+L]
            if shifts is not None:
                need2crop = True
                # Modify stim window by 20-per-side
                self.stim_pos = [
                    self.stim_pos[0]-BUF,
                    self.stim_pos[1]-BUF,
                    self.stim_pos[2]+BUF,
                    self.stim_pos[3]+BUF]
                print( "  Stim expansion for shift:", self.stim_pos)

            L = self.stim_pos[2]-self.stim_pos[0]
            assert self.stim_pos[3]-self.stim_pos[1] == L, "Stimulus not square"

            newstim = np.zeros( [self.NT, num_clr, L, L] )
            for ii in range(self.stim_location.shape[1]):
                OVLP = self.rectangle_overlap_ranges(self.stim_pos, self.stim_location[:, ii])
                if OVLP is not None:
                    print("  Writing lam stim %d: overlap %d, %d"%(ii, len(OVLP['targetX']), len(OVLP['targetY'])))
                    strip = deepcopy(newstim[:, :, OVLP['targetX'], :]) #np.zeros([self.NT, num_clr, len(OVLP['targetX']), L])
                    strip[:, :, :, OVLP['targetY']] = deepcopy((self.stimLP[:, :, OVLP['readX'], :][:, :, :, OVLP['readY']]))
                    newstim[:, :, OVLP['targetX'], :] = deepcopy(strip)
                
            if self.stimET is not None:
                for ii in range(self.stim_locationET.shape[1]):
                    OVLP = self.rectangle_overlap_ranges(self.stim_pos, self.stim_locationET[:,ii])
                    if OVLP is not None:
                        print("  Writing ETstim %d: overlap %d, %d"%(ii, len(OVLP['targetX']), len(OVLP['targetY'])))
                        strip = deepcopy(newstim[:, :, OVLP['targetX'], :])
                        strip[:, :, :, OVLP['targetY']] = deepcopy((self.stimET[:, :, OVLP['readX'], :][:, :, :, OVLP['readY']]))
                        newstim[:, :, OVLP['targetX'], :] = deepcopy(strip) 

            self.stim = torch.tensor( newstim, dtype=torch.float32, device=self.device )

        #print('Stim shape', self.stim.shape)
        # Note stim stored in numpy is being represented as full 3-d + 1 tensor (time, channels, NX, NY)
        self.stim_dims = [self.dims[0], L, L, 1]
        if luminance_only:
            if self.dims[0] > 1:
                # Resample first dimension of stimulus
                print('  Shifting to luminance-only')
                stim_tmp = deepcopy(self.stim.reshape([-1]+self.stim_dims[:3]))
                del self.stim
                torch.cuda.empty_cache()
                self.stim = deepcopy(stim_tmp[:, [0], ...])
                self.stim_dims[0] = 1            

        # Insert fixation point
        if (fixdot is not None) and self.is_fixpoint_present( self.stim_pos ):
            fixranges = [None, None]
            for dd in range(2):
                fixranges[dd] = np.arange(
                    np.maximum(self.fix_location[dd]-self.fix_size-self.stim_pos[dd], 0),
                    np.minimum(self.fix_location[dd]+self.fix_size+1, self.stim_pos[dd+2])-self.stim_pos[dd] 
                    ).astype(int)
            # Write the correct value to stim
            #print(fixranges)
            assert fixdot == 0, "Haven't yet put in other fixdot settings than zero" 
            #strip = deepcopy(self.stim[:, :, fixranges[0], :])
            #strip[:, :, :, fixranges[1]] = 0
            #self.stim[:, :, fixranges[0], :] = deepcopy(strip) 
            print('  Adding fixation point')
            for xx in fixranges[0]:
                self.stim[:, :, xx, fixranges[1]] = 0

        self.stim_shifts = shifts
        if self.stim_shifts is not None:
            # Would want to shift by input eye positions if input here
            #print('eye-position shifting not implemented yet')
            print('  Shifting stim...')
            if shift_times is None:
                self.stim = self.shift_stim( shifts, shift_times=shift_times, already_lagged=False )
                self.metas = self.shift_metas(shifts, shift_times=shift_times, already_lagged=False)
            else:
                self.stim[shift_times, ...] = self.shift_stim( shifts, shift_times=shift_times, already_lagged=False )
                self.metas[shift_times, ...] = self.shift_metas(shifts, shift_times=shift_times, already_lagged=False)

        # Reduce size back to original If expanded to handle shifts
        if need2crop:
            #assert self.stim_crop is None, "Cannot crop stim at same time as shifting"
            self.crop_stim( [BUF, L-BUF-1, BUF, L-BUF-1] )  # move back to original size
            self.stim_crop = None
            L = L-2*BUF
            self.stim_dims[1] = L
            self.stim_dims[2] = L
        else:
            self.stim_crop = stim_crop 
            if self.stim_crop is not None:
                self.crop_stim()

        if time_embed is not None:
            self.time_embed = time_embed
        if time_embed > 0:
            #self.stim_dims[3] = num_lags  # this is set by time-embedding
            if time_embed == 2:
                self.stim = self.time_embedding( self.stim, nlags = num_lags )
        # now stimulus is represented as full 4-d + 1 tensor (time, channels, NX, NY, num_lags)

        self.num_lags = num_lags

        # Flatten stim 
        self.stim = self.stim.reshape([self.NT, -1])
        print( "  Done" )
    # END .assemble_stimulus()

    def time_embedding( self, stim=None, nlags=None ):
        """Note this overloads SensoryBase because reshapes in full dimensions to handle folded_lags"""
        assert self.stim_dims is not None, "Need to assemble stim before time-embedding."
        if nlags is None:
            nlags = self.num_lags
        if self.stim_dims[3] == 1:
            self.stim_dims[3] = nlags
        if stim is None:
            tmp_stim = deepcopy(self.stim)
        else:
            if isinstance(stim, np.ndarray):
                tmp_stim = torch.tensor( stim, dtype=torch.float32)
            else:
                tmp_stim = deepcopy(stim)
        #if not isinstance(tmp_stim, np.ndarray):
        #    tmp_stim = tmp_stim.cpu().numpy()
    
        NT = stim.shape[0]
        print("  Time embedding...")
        if len(tmp_stim.shape) == 2:
            print( "Time embed: reshaping stimulus ->", self.stim_dims)
            tmp_stim = tmp_stim.reshape([NT] + self.stim_dims)

        assert self.NT == NT, "TIME EMBEDDING: stim length mismatch"

        tmp_stim = tmp_stim[np.arange(NT)[:,None]-np.arange(nlags), :, :, :]
        if self.folded_lags:
            #tmp_stim = np.transpose( tmp_stim, axes=[0,2,1,3,4] ) 
            tmp_stim = torch.permute( tmp_stim, (0,2,1,3,4) ) 
            print("Folded lags: stim-dim = ", self.stim.shape)
        else:
            #tmp_stim = np.transpose( tmp_stim, axes=[0,2,3,4,1] )
            tmp_stim = torch.permute( tmp_stim, (0,2,3,4,1) )
        return tmp_stim
    # END .time_embedding()

    def crop_stim( self, stim_crop=None ):
        """Crop existing (torch) stimulus and change relevant variables [x1, x2, y1, y2]"""
        if stim_crop is None:
            stim_crop = self.stim_crop
        else:
            self.stim_crop = stim_crop 
        assert len(stim_crop) == 4, "stim_crop must be of form: [x1, x2, y1, y2]"
        if len(self.stim.shape) == 2:
            self.stim = self.stim.reshape([self.NT] + self.stim_dims)
            reshape=True
            #print('  CROP: reshaping stim')
        else:
            reshape=False
        #stim_crop = np.array(stim_crop, dtype=np.int64) # make sure array
        xs = np.arange(stim_crop[0], stim_crop[1]+1)
        ys = np.arange(stim_crop[2], stim_crop[3]+1)
        if len(self.stim.shape) == 4:
            self.stim = self.stim[:, :, :, ys][:, :, xs, :]
        else:  # then lagged -- need extra dim
            self.stim = self.stim[:, :, :, ys, :][:, :, xs, :, :]

        print("  CROP: New stim size: %d x %d"%(len(xs), len(ys)))
        self.stim_dims[1] = len(xs)
        self.stim_dims[2] = len(ys)
        if reshape:
            print(self.stim.shape, 'reshaping back')
            self.stim = self.stim.reshape([self.NT, -1])
    # END .crop_stim()

    def process_fixations( self, sacc_in=None ):
        """Processes fixation informatiom from dataset, but also allows new saccade detection
        to be input and put in the right format within the dataset (main use)"""
        if sacc_in is None:
            sacc_in = self.sacc_inds[:, 0]
        else:
            print( "  Redoing fix_n with saccade inputs: %d saccades"%len(sacc_in) )
            if self.start_t > 0:
                print( "  -> Adjusting timing for non-zero start time in this dataset.")
            sacc_in = sacc_in - self.start_t
            sacc_in = sacc_in[sacc_in > 0]

        fix_n = np.zeros(self.NT, dtype=np.int64) 
        fix_count = 0
        for ii in range(len(self.block_inds)):
            #if self.block_inds[ii][0] < self.NT:
            # note this will be the inds in each file -- file offset must be added for mult files
            #self.block_inds.append(np.arange( blk_inds[ii,0], blk_inds[ii,1], dtype=np.int64))

            # Parse fixation numbers within block
            rel_saccs = np.where((sacc_in > self.block_inds[ii][0]+6) & (sacc_in < self.block_inds[ii][-1]-5))[0]

            tfix = self.block_inds[ii][0]  # Beginning of fixation by definition
            for mm in range(len(rel_saccs)):
                fix_count += 1
                # Range goes to beginning of next fixation (note no gap)
                fix_n[ range(tfix, sacc_in[rel_saccs[mm]]) ] = fix_count
                tfix = sacc_in[rel_saccs[mm]]
            # Put in last (or only) fixation number
            if tfix < self.block_inds[ii][-1]:
                fix_count += 1
                fix_n[ range(tfix, self.block_inds[ii][-1]) ] = fix_count

        # Determine whether to be numpy or tensor
        if isinstance(self.robs, torch.Tensor):
            self.fix_n = torch.tensor(fix_n, dtype=torch.int64, device=self.robs.device)
        else:
            self.fix_n = fix_n
    # END .process_fixations()

    def augment_dfs( self, new_dfs, cells=None ):
        """Replaces data-filter for given cells. note that new_df should be np.ndarray"""
        
        NTdf, NCdf = new_dfs.shape 
        if cells is None:
            assert NCdf == self.dfs.shape[1], "new DF is wrong shape to replace DF for all cells."
            cells = range(self.dfs.shape[1])
        if self.NT < NTdf:
            self.dfs[:, cells] *= torch.tensor(new_dfs[:self.NT, :], dtype=torch.float32)
        else:
            if self.NT > NTdf:
                # Assume dfs are 0 after new valid region
                print("Truncating valid region to new datafilter length", NTdf)
                new_dfs = np.concatenate( 
                    (new_dfs, np.zeros([self.NT-NTdf, len(cells)], dtype=np.float32)), 
                    axis=0)
            self.dfs[:, cells] *= torch.tensor(new_dfs, dtype=torch.float32)
    # END .augment_dfs()

    def draw_stim_locations( self, top_corner=None, L=None, row_height=5.0 ):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        lamlocs = self.stim_location
        ETlocs = self.stim_locationET
        fixloc = self.fix_location
        fixsize = self.fix_size
        BUF = 10
        if L is None:
            L = lamlocs[2,0]-lamlocs[0,0]

        fig, ax = plt.subplots()
        fig.set_size_inches(row_height, row_height)
        nET = ETlocs.shape[1]
        nLAM = lamlocs.shape[1]
        x0 = np.minimum( np.min(lamlocs[0,:]), np.min(ETlocs[0,:]) )
        x1 = np.maximum( np.max(lamlocs[2,:]), np.max(ETlocs[2,:]) )
        y0 = np.minimum( np.min(lamlocs[1,:]), np.min(ETlocs[1,:]) )
        y1 = np.maximum( np.max(lamlocs[3,:]), np.max(ETlocs[3,:]) )
        #print(x0,x1,y0,y1)
        for ii in range(nLAM):
            ax.add_patch(
                Rectangle((lamlocs[0, ii], lamlocs[1, ii]), 60, 60, 
                edgecolor='red', facecolor='none', linewidth=1.5))
        clrs = ['blue', 'green', 'purple']
        for ii in range(nET):
            ax.add_patch(Rectangle((ETlocs[0,ii], ETlocs[1,ii]), 60, 60, 
                                edgecolor=clrs[ii], facecolor='none', linewidth=1))
        if fixloc is not None:
            ax.add_patch(Rectangle((fixloc[0]-fixsize-1, fixloc[1]-fixsize-1), fixsize*2+1, fixsize*2+1, 
                                facecolor='cyan', linewidth=0))
        if top_corner is not None:
            ax.add_patch(Rectangle(top_corner, L, L, 
                                edgecolor='orange', facecolor='none', linewidth=1, linestyle='dashed'))
            
        ax.set_aspect('equal', adjustable='box')
        plt.xlim([x0-BUF,x1+BUF])
        plt.ylim([y0-BUF,y1+BUF])
        plt.show()
    # END .draw_stim_locations()

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

    def shift_stim(
        self, pos_shifts, metrics=None, metric_threshold=1, ts_thresh=8,
        shift_times=None, already_lagged=True ):
        """Shift stimulus given standard shifting input (TBD)
        use 'shift-times' if given shifts correspond to range of times"""
        NX = self.stim_dims[1]
        nlags = self.stim_dims[3]

        # Check if has been time-lagged yet
  
        if already_lagged:
            re_stim = deepcopy(self.stim).reshape([-1] + self.stim_dims)[..., 0]
        else:
            assert len(self.stim) > 2, "Should be already lagged, but seems not"
            re_stim = deepcopy(self.stim)

        # Apply shift-times (subset)
        if shift_times is None:
            fix_n = np.array(self.fix_n, dtype=np.int64)
        else:
            fix_n = np.array(self.fix_n[shift_times], dtype=np.int64)
            # Find minimum fix_n and make = 1
            min_fix_n = np.min(fix_n[fix_n > 0])
            #print('min_fix_n', min_fix_n, 'adjust', 1-min_fix_n)
            fix_n[fix_n > 0] += 1-min_fix_n
            re_stim = re_stim[shift_times, ...]
            #print('max fix', np.max(fix_n), fix_n.shape)
        
        NF = np.max(fix_n)
        NTtmp = re_stim.shape[0]
        nclr = self.stim_dims[0]
        #sh0 = -(pos_shifts[:,0]-self.dims[1]//2)
        #sh1 = -(pos_shifts[:,1]-self.dims[2]//2)
        sh0 = pos_shifts[:, 0]  # this should be in units of pixels relative to 0
        sh1 = pos_shifts[:, 1]

        sp_stim = deepcopy(re_stim)
        if metrics is not None:
            val_fix = metrics > metric_threshold
            print("  Applied metric threshold: %d/%d"%(np.sum(val_fix), NF))
        else:
            val_fix = np.array([True]*NF)

        for ff in range(NF):
            ts = np.where(fix_n == ff+1)[0]
            #print(ff, len(ts), ts[0], ts[-1])

            if (abs(sh0[ff])+abs(sh1[ff]) > 0) & (len(ts) > ts_thresh) & val_fix[ff] & (ts[-1] < self.NT):
                # FIRST SP DIM shift
                sh = int(sh0[ff])
                stim_seg = re_stim[ts, ...]
                if sh > 0:
                    stim_tmp = torch.zeros([len(ts), nclr, NX, NX], dtype=torch.float32)
                    stim_tmp[:, :,sh:, :] = deepcopy(stim_seg[:, :, :(-sh), :])
                elif sh < 0:
                    stim_tmp = torch.zeros([len(ts), nclr, NX, NX], dtype=torch.float32)
                    stim_tmp[:, :, :sh, :] = deepcopy(stim_seg[:, :, (-sh):, :])
                else:
                    stim_tmp = deepcopy(stim_seg)

                # SECOND SP DIM shift
                sh = int(sh1[ff])
                if sh > 0:
                    stim_tmp2 = torch.zeros([len(ts), nclr, NX,NX], dtype=torch.float32)
                    stim_tmp2[... , sh:] = deepcopy(stim_tmp[..., :(-sh)])
                elif sh < 0:
                    stim_tmp2 = torch.zeros([len(ts), nclr, NX,NX], dtype=torch.float32)
                    stim_tmp2[..., :sh] = deepcopy(stim_tmp[..., (-sh):])
                else:
                    stim_tmp2 = deepcopy(stim_tmp)
                    
                sp_stim[ts, ... ] = deepcopy(stim_tmp2)

        if already_lagged:
            # Time-embed
            idx = np.arange(NTtmp)
            laggedstim = sp_stim[np.arange(NTtmp)[:,None]-np.arange(nlags), ...]
            return np.transpose( laggedstim, axes=[0,2,3,4,1] )
        else:
            return sp_stim
    # END .shift_stim() -- note outputs stim rather than overwrites

    def shift_metas(self, pos_shifts, metrics=None, metric_threshold=1, ts_thresh=8,
        shift_times=None, already_lagged=True ):
        """ Shift the relevant Hartley metadata. """
        sp_metas = deepcopy(self.metas)
        return sp_metas

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
    # END .shift_stim_fixation()

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
    # END .create_valid_indices()

    def __getitem__(self, idx):

        assert self.stim is not None, "Have to specify stimulus before pulling data."
        #if isinstance(idx, np.ndarray):
        #    idx = list(idx)
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
                        'dfs': self.dfs[idx, :]}
                    if len(self.fix_n) > 0:
                        out['fix_n'] = self.fix_n[idx]
                        # missing saccade timing vector -- not specified
                else:
                    if self.robs_out is not None:
                        robs_tmp = self.robs_out
                        dfs_tmp = self.dfs_out
                    else:
                        assert isinstance(self.cells_out, list), 'cells_out must be a list'
                        robs_tmp =  self.robs[:, self.cells_out]
                        dfs_tmp =  self.dfs[:, self.cells_out]
 
                    out = {'stim': self.stim[idx, :],
                        'robs': robs_tmp[idx, :],
                        'dfs': dfs_tmp[idx, :]}
                    if len(self.fix_n) > 0:
                        out['fix_n'] = self.fix_n[idx]

                if self.speckled:
                    if self.Mtrn_out is None:
                        M1tmp = self.Mval[:, self.cells_out]
                        M2tmp = self.Mtrn[:, self.cells_out]
                        out['Mval'] = M1tmp[idx, :]
                        out['Mtrn'] = M2tmp[idx, :]
                    else:
                        out['Mval'] = self.Mtrn_out[idx, :]
                        out['Mtrn'] = self.Mtrn_out[idx, :]

                if self.binocular and self.output_separate_eye_stim:
                    # Overwrite left stim with left eye only
                    tmp_dims = out['stim'].shape[-1]//2
                    stim_tmp = self.stim[idx, :].reshape([-1, 2, tmp_dims])
                    out['stim'] = stim_tmp[:, 0, :]
                    out['stimR'] = stim_tmp[:, 1, :]            
        else:
            inds = self.valid_inds[idx]
            stim = []
            robs = []
            dfs = []
            num_dims = self.stim_dims[0]*self.stim_dims[1]*self.stim_dims[2]

            """ Stim """
            # need file handle
            f = 0
            #f = self.file_index[inds]  # problem is this could span across several files

            stim = torch.tensor(self.fhandles[f]['stim'][inds,:], dtype=torch.float32)
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
        if self.binocular:
            out['binocular'] = self.binocular_gain[idx, :]
            
        ### THIS IS NOT NEEDED WITH TIME-EMBEDDING: needs to be on fixation-process side...
        # cushion DFs for number of lags (reducing stim)
        #if (self.num_lags > 0) &  ~utils.is_int(idx):
        #    if out['dfs'].shape[0] > self.num_lags:
        #        out['dfs'][:self.num_lags, :] = 0.0
        #    else: 
        #        print( "Warning: requested batch smaller than num_lags %d < %d"%(out['dfs'].shape[0], self.num_lags) )
     
        return out
    # END: CloudDataset.__getitem__

    def __len__(self):
        return self.robs.shape[0]

    @staticmethod
    def rectangle_overlap_ranges( A, B ):
        """Figures out ranges to write relevant overlap of B onto A
        All info is of form [x0, y0, x1, y1]"""
        C, D = np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)
        for ii in range(2):
            if A[ii] >= B[ii]: 
                C[ii] = 0
                D[ii] = A[ii]-B[ii]
                if A[2+ii] <= B[2+ii]:
                    C[2+ii] = A[2+ii]-A[ii]
                    D[2+ii] = A[2+ii]-B[ii] 
                else:
                    C[2+ii] = B[2+ii]-A[ii]
                    D[2+ii] = B[2+ii]-B[ii]
            else:
                C[ii] = B[ii]-A[ii]
                D[ii] = 0
                if A[2+ii] <= B[2+ii]:
                    C[2+ii] = A[2+ii]-A[ii]
                    D[2+ii] = A[2+ii]-B[ii]
                else:
                    C[2+ii] = B[2+ii]-A[ii]
                    D[2+ii] = B[2+ii]-B[ii]

        if (C[2]<=C[0]) | (C[3]<=C[1]):
            return None  
        ranges = {
            'targetX': np.arange(C[0], C[2]),
            'targetY': np.arange(C[1], C[3]),
            'readX': np.arange(D[0], D[2]),
            'readY': np.arange(D[1], D[3])}
        return ranges
    # END .rectangle_overlap_ranges()
