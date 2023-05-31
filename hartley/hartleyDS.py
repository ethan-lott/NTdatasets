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

class Hartley(SensoryBase):
    """
    HARTLEY Dataset, 3 color channels w/ 2D images.
    """

    def __init__(self,
                filenames,
                datadir, 
                time_embed=2,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
                num_lags=10, 
                include_MUs=False,
                preload=True,
                drift_interval=None,
                device=torch.device('cpu'),
                # Dataset-specitic inputs
                # Stim setup -- if dont want to assemble stimulus: specify all things here for default options
                which_stim=None,  # 'et' or 0, or 1 for lam, but default assemble later
                stim_crop=None,  # should be list/array of 4 numbers representing inds of edges
                luminance_only=True,
                ignore_saccades=True,
                folded_lags=False, 
                binocular = False, # whether to include separate filters for each eye
                eye_config = 2,  # 0 = all, 1, -1, and 2 are options (2 = binocular)
                maxT = None,
                **kwargs
                ):
        
        super().__init__(
                        filenames=filenames,
                        datadir=datadir,
                        **kwargs
                        )

        # Open .mat files as H5PYs
        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.mat'), 'r') for sess in self.filenames]
        
        # build index map
        self.file_index = []    # which file the block corresponds to
        self.NTfile = []        # Timeframe count in each file

        self.unit_ids = []      # Indeces of the units in each file
        self.dims_file = []     # Dimensions of stimulus in each file

        if (self.device is not None) and (not self.preload):
            self.preload = True
            print("Warning: switching preload to True so device argument is meaningful.")

        self.num_blocks = 0
        self.block_assign = []
        self.block_grouping = []
        nfiles = 0

        for f, fhandle in enumerate(self.fhandles):
            
            # Read this file's counts
            NTfile = fhandle['Robs'].shape[0]                           # Timeframe count in this file
            NSUfile = fhandle['Robs'].shape[1]                          # Single Unit count in this file
            NMUfile = fhandle['RobsMU'].shape[1]                        # Multi-Unit count in this file
            NCfile = NSUfile + NMUfile if include_MUs else NSUfile      # Total cell count in this file
            
            # Store file-specific counts
            self.NTfile.append(NTfile)                                  # Store this file's timeframe count
            self.num_SUs.append(NSUfile)                                # Store this file's SU count
            self.num_MUs.append(NMUfile)                                # Store this file's MU count
            self.num_units.append(NCfile)                               # Store this file's total cell count

            self.SUs += list(range(self.NC, self.NC+NSUfile))           # Store indeces of this file's SUs
            self.unit_ids.append(self.NC + np.asarray(range(NCfile)))   # Store indeces of this file's total cells
            
            self.dims_file.append(list(fhandle['stim'].shape[1:]))      # Store stim dims for this file

            # Get data filters
            SUdfs = fhandle['datafilts'][:,:]                                       # SU datafilters
            MUdfs = fhandle['datafiltsMU'][:,:]                                     # MU datafilters
            self.dfs = np.concatenate((SUdfs,MUdfs),1) if self.include_MUs else SUdfs    # Store either combined dfs or just SUs

            # Pull blocks from data_filters
            blocks = (np.sum(self.dfs, axis=1)==0).astype(np.float32)
            blocks[0] = 1       # set invalid first sample
            blocks[-1] = 1      # set invalid last sample

            blockstart = np.where(np.diff(blocks)==-1)[0]
            blockend = np.where(np.diff(blocks)==1)[0]
            nblocks = len(blockstart)

            for b in range(nblocks):
                self.file_index.append(f)                                                   # Store which blocks are from which file
                self.block_inds.append(self.NT + np.arange(blockstart[b], blockend[b]))     # Store indeces of all the blocks
            
            # Assign each block to a file
            self.block_assign = np.concatenate(                                             # Assign each dataum in each block to its file
                (self.block_assign, nfiles*np.ones(NTfile, dtype=int)), axis=0)
            self.block_grouping.append( self.num_blocks+np.arange(nblocks, dtype=int) )     # Store array of block indeces for this file 
            
            self.NT += NTfile               # Accumulate total timeframes
            self.NC += NCfile               # Accumulate total cells
            self.num_blocks += nblocks      # Accumulate total blocks
            nfiles += 1                     # Increment number of files

        # Set overall dataset variables
        NX = self.dims_file[0]                                  # Assumes they're all the same
        assert len(NX) == 3, f'problem: len(NX) == {len(NX)}'   
        lags = self.num_lags if self.time_embed else 1          # Default to 1 lag
        self.dims = NX + [lags]                                 # Set initial dimensions

        # For now do this without using assemble_stimlus
        self.stim_dims = deepcopy(self.dims)

        # Preload data
        if self.preload:
            
            self.stim = np.zeros([self.NT, np.prod(self.dims)], dtype=np.float32)   # Init stimulus array
            self.robs = np.zeros([self.NT, self.NC], dtype=np.float32)              # Init robs array
            self.dfs = np.zeros([self.NT, self.NC], dtype=np.float32)               # Init dfs array
            tcount, ccount = 0, 0                                                   # Init timeframe and cell counts
            
            for f, fhandle in enumerate(self.fhandles):

                print("Loading", self.filenames[f])

                NT = self.NTfile[f]                 # Get timeframe count for file
                NSU = self.num_SUs[f]               # Get SU count for file
                trange = range(tcount, tcount+NT)   # Get timeframe range fopr file
                crange = range(ccount, ccount+NSU)  # Get SU range for file
                
                # Stimulus
                stim_raw = np.array(self.fhandles[f]['stim'], dtype=np.float32)                           # Get non-embedded stimulus
                self.stim[trange, :] = self.time_embedding(stim_raw) if self.time_embed else stim_raw     # Embed if necessary

                # Robs and DFs
                robs_tmp = np.zeros([NT, self.NC], dtype=np.float32)
                dfs_tmp = np.zeros([NT, self.NC], dtype=np.float32)

                robs_tmp[:, crange] = np.array(self.fhandles[f]['Robs'], dtype=np.float32)
                dfs_tmp[:, crange] = np.array(self.fhandles[f]['datafilts'], dtype=np.float32)

                if self.include_MUs:
                    NC = self.num_units[f]
                    crange = range(ccount+NSU, ccount+NC)
                    robs_tmp[:, crange] = np.array(self.fhandles[f]['RobsMU'], dtype='float32')
                    dfs_tmp[:, crange] = np.array(self.fhandles[f]['datafiltsMU'], dtype='float32')
                else:
                    NC = NSU

                self.robs[trange, :] = deepcopy(robs_tmp)
                self.dfs[trange, :] = deepcopy(dfs_tmp)
                tcount += NT
                ccount += NC

            # Convert data to tensor
            self.to_tensor()


    def to_tensor(self, device=None):

        if device is None:
            device = torch.device('cpu') if self.device is None else self.device

        if type(self.robs) != torch.Tensor:
            self.stim = torch.tensor(self.stim, dtype=torch.float32, device=device)
            self.robs = torch.tensor(self.robs, dtype=torch.float32, device=device)
            self.dfs = torch.tensor(self.dfs, dtype=torch.float32, device=device)
        else: 
            # Simply move to device:
            self.stim = self.stim.to(device)
            self.robs = self.robs.to(device)
            self.dfs = self.dfs.to(device)

    # END MultiDataset.to_tensor


    def __getitem__(self, index):
        """
        Called by the DataLoader to build the batch up one item at a time.
        :param index: index to use for this batch
        :return: dictionary of tensors for this batch
        """
        if self.preload:
            stim = self.stim[index, :]
            robs = self.robs[index, :]
            dfs = self.dfs[index, :]
        else:
            stim = []
            robs = []
            dfs = []
            for ii in index:
                inds = self.block_inds[ii]
                NT = len(inds)
                f = self.file_index[ii]

                """ Stim """
                stim_tmp = torch.tensor(self.fhandles[f]['stim'][inds,:], dtype=torch.float32)

                """ Spikes: needs padding so all are B x NC """ 
                robs_tmp = torch.tensor(self.fhandles[f]['robs'][inds,:], dtype=torch.float32)
                NCbefore = int(np.asarray(self.num_units[:f]).sum())
                NCafter = int(np.asarray(self.num_units[f+1:]).sum())
                robs_tmp = torch.cat(
                    (torch.zeros( (NT, NCbefore), dtype=torch.float32),
                    robs_tmp,
                    torch.zeros( (NT, NCafter), dtype=torch.float32)),
                    dim=1)

                """ Datafilters: needs padding like robs """
                dfs_tmp = torch.tensor(self.fhandles[f]['dfs'][inds,:], dtype=torch.float32)
                dfs_tmp[:self.num_lags,:] = 0 # invalidate the filter length
                dfs_tmp = torch.cat(
                    (torch.zeros( (NT, NCbefore), dtype=torch.float32),
                    dfs_tmp,
                    torch.zeros( (NT, NCafter), dtype=torch.float32)),
                    dim=1)

                stim.append(stim_tmp)
                robs.append(robs_tmp)
                dfs.append(dfs_tmp)

            stim = torch.cat(stim, dim=0)
            robs = torch.cat(robs, dim=0)
            dfs = torch.cat(dfs, dim=0)

        if len(self.cells_out) > 0:
            cells_out = np.array(self.cells_out, dtype=np.int64)
            assert len(cells_out) > 0, "DATASET: cells_out must be a non-zero length"
            assert np.max(cells_out) < self.robs.shape[1],  "DATASET: cells_out must be a non-zero length"
            return {'stim': stim, 'robs': robs[:, cells_out], 'dfs': dfs[:, cells_out]}
        else:
            return {'stim': stim, 'robs': robs, 'dfs': dfs}
    # END MultiDataset.__get_item__