import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
from copy import deepcopy
import h5py
from NTdatasets.sensory_base import SensoryBase

class HartleyDataset(SensoryBase):
        
    def __init__(self,
                 filenames,
                 datadir, 
                 time_embed=None,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
                 num_lags=12, 
                 include_MUs=True,
                 drift_interval=None,
                 trial_sample=False,
                 device=torch.device('cpu'),
                 # Dataset-specitic inputs
                 # Stim setup -- if dont want to assemble stimulus: specify all things here for default options
                 ignore_saccades=True,
                 binocular=False,
                 eye_config=0,               # 0 = no subsampling, 1(left), 2(right), and 3(binocular) are options
                 maxT=None):
                 #meta_vals = 4):
    
        super().__init__(
            filenames=filenames, datadir=datadir, 
            time_embed=time_embed, num_lags=num_lags, 
            include_MUs=include_MUs, drift_interval=drift_interval,
            trial_sample=trial_sample, device=device)
        
        # Stim-specific
        self.eye_config = eye_config
        self.binocular = binocular
        self.generate_Xfix = False
        self.output_separate_eye_stim = False
        #self.meta_vals = meta_vals

        self.start_t = 0
        self.drift_interval = drift_interval

        # Get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.mat'), 'r') for sess in self.filenames]
        self.avRs = None

        self.one_hots = []
        self.meta_dims = []
        self.hartley_metadata = []
        self.meta = []
        self.OHcov = []

        self.fix_n = []
        self.used_inds = []
        self.NT = 0

        # build index map -- exclude variables already set in sensory-base
        self.num_blks = np.zeros(len(filenames), dtype=int)
        self.data_threshold = 6     # how many valid time points required to include saccade?
        self.file_index = []        # which file the block corresponds to
        self.sacc_inds = None
        self.stim_shifts = None
        self.meta_shift = None

        # Get data from each file
        if isinstance(self.fhandles, list):
            self.fhandles = self.fhandles[0]  # there can be only one
        fhandle = self.fhandles

        # Read and store this file's counts
        NT, NSUfile = fhandle['Robs'].shape                                                 # Read timeframe and single unit counts
        if maxT is not None:
            NT = np.minimum(maxT, NT)
        self.maxT = NT

        NMUfile = fhandle['RobsMU'].shape[1] if len(fhandle['RobsMU'].shape) > 1 else 0     # Read multi-unit count
        NCfile = NSUfile + NMUfile if self.include_MUs else NSUfile                         # Calculate total cell count
        self.num_SUs.append(NSUfile)                                                        # Store SU count
        self.num_MUs.append(NMUfile)                                                        # Store MU count
        self.num_units.append(NCfile)                                                       # Store cell count
        self.SUs = self.SUs + list(range(self.NC, self.NC+NSUfile))                         # Store SU indices
        self.NC += NCfile                                                                   # Accumulate total cells
        self.NT += NT                                                                       # Accumulate total timeframes
        
        # Get this file's blocks
        blk_inds = np.array(fhandle['block_inds'], dtype=np.int64)          # Read this file's block indices
        blk_inds[:, 0] -= 1                                                 # Convert to python so range works
        if blk_inds.shape[0] == 2:                                          # Transpose array if old style
            print('WARNING: blk_inds is stored old-style: transposing')
            blk_inds = blk_inds.T
        # self.blockID = np.array(fhandle['blockID'], dtype=np.int64)[:, 0]   # Read and store block IDs
        if self.maxT is not None:
            blk_inds=blk_inds[blk_inds[:,1]<self.maxT,:]

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
        self.block_filemapping += list(np.zeros(blk_inds.shape[0], dtype=int))
        self.num_blks[0] = blk_inds.shape[0]

        self.dims = list(fhandle['stim'].shape[1:4]) + [1] # this is basis of data (stimLP and stimET)
        self.stim_dims = None                               # when stim is constructed
        self.meta_dims = None                               # when meta is constructed

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

        # For now let's just debug with one file
        self.used_inds = deepcopy(valid_inds[valid_inds<self.maxT])
        self.LRpresent = LRpresent

        print("Loading data into memory...")
        self.preload_numpy()

        if time_embed is not None:
            self.assemble_metadata(time_embed=time_embed, num_lags=num_lags)

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
        if self.maxT is not None:
            t1 = np.minimum(t0+self.maxT, t1)
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
    # END HartleyDataset.__init__

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

        self.train_blks = trblks
        self.val_blks = vblks

        if self.maxT is not None:
            self.val_inds = self.val_inds[self.val_inds<self.maxT]
    # END .crossval_setup()

    def preload_numpy(self):
        """Note this loads stimulus but does not time-embed"""

        NT = self.NT
        ''' 
        Pre-allocate memory for data
        '''
        self.stimLP = np.zeros( [NT] + self.dims[:3], dtype=np.float32)
        self.hartley_metadata = np.zeros([NT, 4], dtype=np.float32)

        # No need to load ET stimulus in this dataset
        self.stimET = None
        
        self.robs = np.zeros( [NT, self.NC], dtype=np.float32)
        self.dfs = np.ones( [NT, self.NC], dtype=np.float32)
        #self.meta = np.zeros([NT, self.meta_vals], dtype=np.float32)

        fhandle = self.fhandles
        sz = fhandle['stim'].shape
        sz0 = np.minimum(self.maxT, sz[0]) if self.maxT is not None else sz[0]
        inds = np.arange(sz0, dtype=np.int64)

        self.stimLP[inds, ...] = np.array(fhandle['stim'], dtype=np.float32)[:sz0, ...]
        if self.stimET is not None:
            self.stimET[inds, ...] = np.array(fhandle['stimET'], dtype=np.float32)[:sz0, ...]

        """ Hartley metadata """
        self.hartley_metadata[inds, :] = np.array(fhandle['hartley_metas'], dtype=np.float32)[:sz0, ...]

        """ Robs and DATAFILTERS"""
        robs_tmp = np.zeros( [len(inds), self.NC], dtype=np.float32 )
        dfs_tmp = np.zeros( [len(inds), self.NC], dtype=np.float32 )
        num_sus = fhandle['Robs'].shape[1]
        units = range(num_sus)
        robs_tmp[:, units] = np.array(fhandle['Robs'], dtype=np.float32)[:sz0, ...]
        dfs_tmp[:, units] = np.array(fhandle['datafilts'], dtype=np.float32)[:sz0, ...]
        if self.include_MUs:
            num_mus = fhandle['RobsMU'].shape[1]
            units = range(num_sus, num_sus+num_mus)
            robs_tmp[:, units] = np.array(fhandle['RobsMU'], dtype=np.float32)[:sz0, ...]
            dfs_tmp[:, units] = np.array(fhandle['datafiltsMU'], dtype=np.float32)[:sz0, ...]
        
        self.robs[inds,:] = deepcopy(robs_tmp)
        self.dfs[inds,:] = deepcopy(dfs_tmp)

        # Adjust stimulus since stored as ints
        if np.std(self.stimLP) > 5: 
            if np.mean(self.stimLP) > 50:
                self.stimLP += -127
            self.stimLP *= 1/128
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

    def assemble_metadata(self, time_embed=0, num_lags=10):
        """ This assembles the Hartley metadata from the raw numpy-stored metadata into self.meta """
        
        # Delete existing metadata and clear cache to prevent memory issues on GPU
        if self.meta is not None:
            del self.meta
            self.meta = None
            torch.cuda.empty_cache()

        # Express meta_dims as spatial dimension by time
        if time_embed is not None:
            self.time_embed = time_embed
        else:
            time_embed = 0

        self.one_hots = []
        self.hartley_categories = []
        self.hartley_dims = np.zeros(4, dtype=np.int64)
        for meta_val in range(4):
            oh, categs = self.one_hot_encoder(self.hartley_metadata[:, meta_val])
            self.hartley_dims[meta_val] = len(categs)
            self.hartley_categories.append(deepcopy(categs))

            self.one_hots.append(deepcopy(oh.reshape([self.NT, -1])))
            if time_embed == 2:
                oh = self.time_embedding(oh, nlags=num_lags, verbose=False)
            else:
                oh = torch.tensor(oh, dtype=torch.float32, device=self.device)

            # Inelegant hard-coding
            if meta_val == 0:
                self.OHfreq = deepcopy(oh.reshape([self.NT, -1]))
            elif meta_val == 1:
                self.OHori = deepcopy(oh.reshape([self.NT, -1]))
            elif meta_val == 2:
                self.OHphase = deepcopy(oh.reshape([self.NT, -1]))
            elif meta_val == 3:
                self.OHcolor = deepcopy(oh.reshape([self.NT, -1]))

        self.meta_dims = [1, np.sum(self.hartley_dims), 1, 1]

        # Assemble ori/phase covariable
        ori_i, phase_i = 1, 2
        ori_dim, phase_dim = self.hartley_dims[ori_i], self.hartley_dims[phase_i]
        oh = np.zeros([self.NT, ori_dim*phase_dim], dtype=np.float32)
        for t in range(self.NT):
            i = phase_dim*self.one_hots[ori_i][t].argmax() + self.one_hots[phase_i][t].argmax()
            oh[t][i] = 1

        if time_embed == 2:
            self.OHcov = self.time_embedding(oh, nlags=num_lags, verbose=False)
        else:
            self.OHcov = torch.tensor(oh, dtype=torch.float32, device=self.device)

        # Assemble full one-hot 
        self.one_hots = np.concatenate(self.one_hots, axis=1)
        self.meta = torch.tensor(self.one_hots, dtype=torch.float32, device=self.device)

        '''self.meta_shifts = shifts
        if self.meta_shifts is not None:
            # Would want to shift by input eye positions if input here
            print('\tShifting stim...')
            if shift_times is None:
                self.meta = self.shift_meta(shifts, shift_times=shift_times, already_lagged=False)
            else:
                self.meta[shift_times, ...] = self.shift_meta(shifts, shift_times=shift_times, already_lagged=False)'''

        if time_embed is not None:
            self.time_embed = time_embed
            if time_embed == 2:
                self.meta_dims[3] = num_lags
                self.meta = self.time_embedding(self.meta, nlags=num_lags)
                self.meta_dims[3] = num_lags
        # now metadata is represented as full 4-d + 1 tensor (time, channels, NX, NY, num_lags)

        self.num_lags = num_lags

        # Flatten meta 
        self.meta = self.meta.reshape([self.NT, -1])
    # END .assemble_metadata()

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

    def shift_meta(self, pos_shifts, metrics=None, metric_threshold=1, ts_thresh=8,
        shift_times=None, already_lagged=True ):
        """ Shift the relevant Hartley metadata. """
        sp_meta = deepcopy(self.meta)
        return sp_meta

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
        bool_T = (is_valid > 0 and is_valid < self.maxT) if self.maxT is not None else is_valid > 0
        self.valid_inds = np.where(bool_T)[0]
    # END .create_valid_indices()

    def __getitem__(self, idx):

        if self.trial_sample:
            idx = self.index_to_array(idx, len(self.block_inds))
            ts = self.block_inds[idx[0]]
            for ii in idx[1:]:
                ts = np.concatenate( (ts, self.block_inds[ii]), axis=0 )
            idx = ts

        #assert self.stim is not None, "Have to specify stimulus before pulling data."
        #if isinstance(idx, np.ndarray):
        #    idx = list(idx)
        if self.time_embed == 1:
            print("get_item time embedding not implemented yet")

        out = {
            'meta': self.meta[idx,:],  # can probably phase this out
            'stim': self.meta[idx, :],
            'OHfreq': self.OHfreq[idx, :],
            'OHori': self.OHori[idx, :],
            'OHphase': self.OHphase[idx, :],
            'OHcolor': self.OHcolor[idx, :],
            'OHcov': self.OHcov[idx, :]}
        
        if len(self.fix_n) > 0:
            out['fix_n'] = self.fix_n[idx]

        if len(self.cells_out) == 0:
            out['robs'] = self.robs[idx, :]
            out['dfs'] = self.dfs[idx, :]
        else:
            if self.robs_out is not None:
                robs_tmp = self.robs_out
                dfs_tmp = self.dfs_out
            else:
                assert isinstance(self.cells_out, list), 'cells_out must be a list'
                robs_tmp =  self.robs[:, self.cells_out]
                dfs_tmp =  self.dfs[:, self.cells_out]

                out['robs'] = robs_tmp[idx, :]
                out['dfs'] = dfs_tmp[idx, :]

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

        # Addition whether-or-not preloaded
        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]
        if self.binocular:
            out['binocular'] = self.binocular_gain[idx, :]
        for cov in self.covariates.keys():
            out[cov] = self.covariates[cov][idx,...]
     
        return out
    # END: HartleyDataset.__getitem__

    def __len__(self):
        return self.robs.shape[0]

    @staticmethod
    def one_hot_encoder(arr):

        arr = np.array(arr, dtype=np.float32).squeeze()
        categories = np.unique(arr)
        arr_dict = {}

        for i, a in enumerate(categories):
            arr_dict[a] = i

        one_hot = np.zeros((arr.shape[0], len(arr_dict)), dtype=np.float32)
        for i, a in enumerate(arr):
            one_hot[i][arr_dict[a]] = 1
        
        return one_hot, categories