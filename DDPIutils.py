import os
import torch
import math
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from NDNT.utils import DanUtils as DU

class DDPIutils():
    '''
    Implemented by Ethan Lott 06/2023.
    '''
    def __init__(self,
                 filename,
                 data_dir,
                 trial_duration=4,
                 resolution=500,
                 blink_thresh=90,
                 device=torch.device('cpu')):
        
        self.fhandle = sio.loadmat(data_dir + filename + '.mat')

        self.blink_thresh = blink_thresh
        self.trial_duration = trial_duration
        self.dt = 1 / resolution

        # Load self
        self.trial_ts = np.array(self.fhandle['trial_ts'], dtype=np.float32).squeeze()
        self.dpi_ts = np.array(self.fhandle['dpi_ts'], dtype=np.float32).squeeze()
        self.et = np.array(self.fhandle['dpi_cal'], dtype=np.float32).squeeze()
        self.NT = self.dpi_ts.shape[0]
        self.num_trials = self.trial_ts.shape[0]
        self.processed = np.zeros([self.num_trials, 2], dtype=np.int32)

        # Saccade stats
        self.num_sacc = np.zeros(self.num_trials, dtype=np.int32)
        self.sacc_dt = [[] for tr in range(self.num_trials)]
        self.sacc_dt_raw = [[] for tr in range(self.num_trials)]
        self.sacc_xy = [[] for tr in range(self.num_trials)]
        self.sacc_sizes = [[] for tr in range(self.num_trials)]
        self.sacc_dirs = [[] for tr in range(self.num_trials)]

        # Fixation stats
        self.num_fixations = np.zeros(self.num_trials, dtype=np.int32)
        self.fix_dt = [[] for tr in range(self.num_trials)]
        self.fix_et = [[] for tr in range(self.num_trials)]
        self.fix_avg_et = [[] for tr in range(self.num_trials)]

        # Seperate ET self into trials
        print('Seperating trials...')
        self.trial_et = []
        for tr in range(self.num_trials):
            tr_indx = np.where((self.dpi_ts >= self.trial_ts[tr]) &
                               (self.dpi_ts <= self.trial_ts[tr]+self.trial_duration))[0]
            trial = np.zeros([tr_indx.shape[0], 3])
            trial[:,0] = self.dpi_ts[tr_indx]
            trial[:,1:] = self.et[tr_indx]
            self.trial_et.append(trial)
        print('Done.')
        self.trial_avg_et = [[] for tr in range(self.num_trials)]
    # END ETDataset.__init__

    def process_trial(self, tr, gap=2, sm=8, verbose=True):

        NT = self.trial_et[tr].shape[0]
        
        # Detect and eliminate blinks
        et, blink_ts = self.blink_process(ETin=self.trial_et[tr][:,1:], blink_thresh=self.blink_thresh, verbose=verbose)
        self.trial_et[tr][:,1:] = deepcopy(et)
        blink_ts60 = np.round(blink_ts/500*60).astype(np.int64) # translate blink times into 60 Hz
        val_blink = np.ones(NT, dtype=np.float32)    
        val_blink60 = np.ones(240, dtype=np.float32)

        # Get moving average
        avg_et = deepcopy(et)
        margin = 5
        for t in range(margin, NT-margin):
            avg_et[t] = np.mean(et[np.arange(t-margin,t+margin)], axis=0)

        self.trial_avg_et[tr] = np.zeros([NT, 3])
        self.trial_avg_et[tr][:,0] = deepcopy(self.trial_et[tr][:,0])
        self.trial_avg_et[tr][:,1:] = deepcopy(avg_et)

        # Approximate saccade detect
        dsacc = np.zeros(NT)
        far, near = sm + gap//2, gap//2
        for t in range(far, NT-far):
            medianL = np.median(et[range(t-far, t-near), :], axis=0)
            medianR = np.median(et[range(t+near, t+far), :], axis=0)
            dsacc[t] = np.sum((medianL - medianR)**2)
        Tsaccs = DU.find_peaks(dsacc, thresh=np.std(dsacc)/2, clearance=30, n_peaks=10)[0]
        Tsaccs = np.sort(Tsaccs)

        # Finish blinks and eliminate edge-saccades
        for bb in range(blink_ts.shape[0]):
           val_blink[range(blink_ts[bb,0], blink_ts[bb,1])] = 0.0
           val_blink60[range(blink_ts60[bb,0], blink_ts60[bb,1])] = 0.0
           a = np.where((Tsaccs >= blink_ts[bb,0]-sm-gap) & (Tsaccs <= blink_ts[bb,1]+sm+gap))[0]
           if len(a) > 0:
               Tsaccs = Tsaccs[(Tsaccs < blink_ts[bb,0]-sm-gap) | (Tsaccs > blink_ts[bb,1]+sm+gap)]

        # Calculate size/direction of saccades
        size = np.zeros(Tsaccs.shape[0])
        prev = np.zeros([Tsaccs.shape[0], 2])
        next = np.zeros([Tsaccs.shape[0], 2])
        dir = np.zeros(Tsaccs.shape[0])
        for i,st in enumerate(Tsaccs):
            prev[i] = np.median(avg_et[st-min(st, 18):max(1,st-10)], axis=0)
            next[i] = np.median(avg_et[min(NT-1,st+10):st+min(18, NT-st)], axis=0)
            if next[i] is not None:
                sacc = np.subtract(next[i], prev[i])
                size[i] = (sacc[0]**2 + sacc[1]**2)**0.5
                dir[i] = np.arctan(sacc[1] / sacc[0]) * (180 / np.pi)
                if sacc[0] < 0:    
                        dir[i] += -180 if sacc[1] < 0 else 180

        self.sacc_sizes[tr] = deepcopy(size)
        self.sacc_dirs[tr] = deepcopy(dir)
        self.num_sacc[tr] = size.shape[0]

        # Find standard deviation over fixations
        std_temp = [0,0]
        et_velo = [0]+[np.sum((avg_et[t]-avg_et[t-1])**2) for t in range(1, NT)]
        for i, st in enumerate(Tsaccs):
            start = 0 if i == 0 else Tsaccs[i-1] + 10
            end = st - 10 if st > 10 else st
            curr_std = [np.std(et_velo[start:end]), end-start]
            std_temp = [std_temp[0] + curr_std[0]*curr_std[1], std_temp[1] + curr_std[1]]
        std = std_temp[0]/std_temp[1]

        # Find start and stop times of saccades
        sacc_dt = np.ones([Tsaccs.shape[0], 2]) * -1
        for i, st in enumerate(Tsaccs):
            t = 1
            while -1 in sacc_dt[i]:
                if sacc_dt[i,0] == -1:
                    if st-t < 0:
                        sacc_dt[i,0] = 0
                    elif et_velo[st-t] < std and ((prev[i,0] - avg_et[st-t,0])**2 +
                                                  (prev[i,1] - avg_et[st-t,1])**2)**0.5 < size[i]/2:
                        sacc_dt[i,0] = st - t - 1
                else:
                    if st+t >= NT:
                        sacc_dt[i,1] = NT
                    elif et_velo[st+t] < std and ((next[i,0] - avg_et[st+t,0])**2 +
                                                  (next[i,1] - avg_et[st+t,1])**2)**0.5 < size[i]/2:
                        sacc_dt[i,1] = st + t
                t += 1

        self.sacc_dt[tr] = deepcopy(sacc_dt)
        self.processed[tr,0] = 1
        sacc_dt_raw = sacc_dt * self.dt + self.trial_ts[tr]
        self.sacc_dt_raw[tr] = deepcopy(sacc_dt_raw)
    # END process_trial

    def blink_process(self, ETin, blink_pad=50, verbose=True, blink_thresh=40):
        """Zero out eye traces where blink is detected, and refurn new eye trace and blink locations
        Works for EyeScan and Eyelink, although default set for EyeLink"""
        ETout = deepcopy(ETin)
        if ETin.shape[1] > 2:
            # then binocular
            a = np.where(
                (abs(ETin[:,0]) > blink_thresh) | (abs(ETin[:,1]) > blink_thresh) | \
                (abs(ETin[:,2]) > blink_thresh) | (abs(ETin[:,3]) > blink_thresh))[0]
        else:
            a = np.where((abs(ETin[:,0]) > blink_thresh) | (abs(ETin[:,1]) > blink_thresh))[0]

        b = np.where(np.diff(a) > 10)[0]
        b = np.append(b, len(a)-1)

        blinks = []
        if len(a) == 0:
            return ETout, np.array(blinks)

        tstart = 0
        for ii in range(len(b)):
            blink_start = np.maximum(a[tstart]-blink_pad, 0)
            blink_end = np.minimum(a[b[ii]]+blink_pad, ETin.shape[0])
            arng = np.arange( blink_start, blink_end )
            blinks.append([blink_start, blink_end])
            #avs = np.mean(ETraw[arng[0]-np.arange(5),:],axis=0)
            ETout[arng,:] = 0
            #ETout[arng,1] = 0
            tstart = b[ii]+1
        if verbose:
            print( "  %d blinks detected and zeroed"%len(blinks) )
        return ETout, np.array(blinks, dtype=np.int64)
    # END blink_process

    def plot_trial(self, tr):
        '''
        Won't work unless the trial has already been processed.
        '''
        assert(self.processed[tr,0] == 1), 'Process the trial first.'
        DU.ss(2,1,rh=3)
        plt.suptitle(f'Trial {tr} Eye Tracking', fontsize='x-large')
        plt.tight_layout(pad=1)
        col = ['blue', 'red']
        for p in range(2):
            plt.subplot(2,1,p+1)
            plt.xlim(0,self.trial_et[tr].shape[0])
            # plt.xticks([125*i for i in range(16)], [0.25*i for i in range(16)])
            plt.xticks([125*i for i in range(17)])
            plt.plot(self.trial_et[tr][:,p+1], color=col[p], alpha=0.4)
            plt.plot(self.trial_avg_et[tr][:,p+1], color=col[p], alpha=1)
            for sacc in self.sacc_dt[tr]:
                plt.axvline(x=sacc[0], color='green', linestyle='dashed', alpha=1)
                plt.axvline(x=sacc[1], color='green', linestyle='dashed', alpha=1)
    # END plot_trial

    def plot_saccades(self, tr, c=4):
        ''' Zoom in on each saccade of the trial. Requires trial to be processed.'''
        assert(self.processed[tr,0] == 1), 'Process the trial first.'
        r = math.ceil(self.num_sacc[tr]/c)
        DU.ss(r, c, row_height=4)
        plt.tight_layout(pad=3)
        plt.suptitle(f'Trial {tr} Saccades', fontsize='x-large')
        plt.subplots_adjust(top=0.9)
        for i, s in enumerate(self.sacc_dt[tr]):
            plt.subplot(r, c, i+1)
            plt.plot(self.trial_et[tr][:, 1], alpha=0.3, color='blue')
            plt.plot(self.trial_avg_et[tr][:, 1], color='blue')
            plt.plot(self.trial_et[tr][:, 2], alpha=0.3, color='red')
            plt.plot(self.trial_avg_et[tr][:, 2], color='red')
            plt.title(f'T{tr}; Sacc. {i}')
            plt.xlim(max(0,int(s[0])-40),min(self.trial_et[tr].shape[0], int(s[1])+40))
            plt.axvline(x=s[0], color='green', linestyle='dashed', alpha=1)
            plt.axvline(x=s[1], color='green', linestyle='dashed', alpha=1)
    # END plot_saccades

    def plot_fixations(self, tr, c=3):
        ''' Plot fixations of the trial. Trial and fixations must be processed.'''
        assert(self.processed[tr,1] == 1), 'Process the trial first.'
        r = math.ceil(len(self.fix_et[tr])/c)
        DU.ss(r, c, row_height=4)
        plt.tight_layout(pad=3)
        plt.suptitle(f'Trial {tr} Fixations', fontsize='x-large')
        plt.subplots_adjust(top=0.9)
        for i, fix in enumerate(self.fix_et[tr]):
            x = np.round((fix[:, 0]-self.trial_ts[tr])/self.dt)
            plt.subplot(r, c, i+1)
            plt.title(f'T{tr}; Fix. {i}')
            plt.plot(x, fix[:,1], color='blue', alpha=0.3)
            plt.plot(x, fix[:,2], color='red', alpha=0.3)
            plt.plot(x, self.fix_avg_et[tr][i][:,1], color='blue', alpha=1)
            plt.plot(x, self.fix_avg_et[tr][i][:,2], color='red', alpha=1)
    # END plot_fixations

    def process_fixations(self, tr):
        '''
        Won't work unless the trial has already been processed.
        '''
        assert(self.processed[tr,0] == 1), 'Process the trial first.'
        num_fix = self.num_sacc[tr]+1

        fix_indx = [[] for fix in range(num_fix)]
        fix_et = [[] for fix in range(num_fix)]
        fix_avg_et = [[] for fix in range(num_fix)]

        for fix in range(num_fix):
            start = 0 if fix == 0 else self.sacc_dt[tr][fix-1,1] + 1
            end = self.trial_et[tr].shape[0] if fix > num_fix-2 else self.sacc_dt[tr][fix,0]
            fix_indx[fix] = [start, end]
            fix_et[fix] = deepcopy(self.trial_et[tr][int(start):int(end)])
            fix_avg_et[fix] = deepcopy(self.trial_avg_et[tr][int(start):int(end)])

        self.processed[tr,1] = 1
        self.fix_dt[tr] = deepcopy(fix_indx)
        self.fix_et[tr] = deepcopy(fix_et)
        self.fix_avg_et[tr] = deepcopy(fix_avg_et)
    # END process_fixation

    def saccade_stats(self):
        ''' Calculate and plot ditributions of saccade stats. Requires all trials to be processed. '''
        for tr in range(self.num_trials):
            assert(self.processed[tr,0] == 1), f'Process trial {tr} first.'
        sizes_all = np.zeros(0)
        dirs_all = np.zeros(0)
        lens_all = np.zeros(0)
        freq_all = np.zeros(self.num_trials)
        fix_len_all = np.zeros(0)
        for trial in range(self.num_trials):
            sizes_all = np.concatenate([sizes_all, self.sacc_sizes[trial]])
            dirs_all = np.concatenate([dirs_all, self.sacc_dirs[trial]])
            if trial != 408:
                lens_all = np.concatenate([lens_all, self.sacc_dt_raw[trial][:,1] - self.sacc_dt_raw[trial][:,0]])
            freq_all[trial] = self.sacc_sizes[trial].shape[0] / self.trial_duration
            fixes = np.array([fix[1] - fix[0] for fix in self.fix_dt[trial]])
            fix_len_all = np.concatenate([fix_len_all, fixes])

        DU.ss(2, 2, rh=5)
        plt.tight_layout(pad=3)
        plt.subplot(2, 2, 1)
        plt.title('Size (arcmin)')
        plt.hist(sizes_all, bins=50)
        plt.subplot(2, 2, 2)
        plt.title('Direction (deg. from East)')
        plt.hist(dirs_all, bins=50)
        plt.subplot(2, 2, 3)
        plt.title('Duration (ms)')
        plt.hist(lens_all*1000, bins=[4*i for i in range(16)])
        plt.subplot(2, 2, 4)
        plt.title('Frequency (sacc/sec)')
        plt.hist(freq_all, bins=[0.25*i for i in range(12)])
        plt.show()
    # END saccade_stats

    def length(self):
        return (self.NT, self.dpi_ts[-1] - self.dpi_ts[0])
