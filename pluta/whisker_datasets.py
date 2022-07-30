import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py
from sensory_base import SensoryBase

class WhiskerData(Dataset):

    def __init__(self,
        **kwargs):

        # call parent constructor
        super().__init__(kwargs)
    # END WhiskerData.__init__()

    def __getitem__(self, idx):
        return {}
    # END WhiskerData.__getitem()
    