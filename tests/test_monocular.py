import sys
sys.path.append('../../') # for the NDNT folder
sys.path.append('../cumming') # for monocular.py

import torch
from monocular import MultiDataset
from torch.utils.data import DataLoader


# make sure the MultiDataset returns the correct shape for each batch
def test_MultiDataset_with_DataLoader_returns_correct_shape():
    # create a MultiDataset object
    datadir = '../../Mdata/'
    num_lags = 10
    dataset = MultiDataset(
        datadir=datadir,
        filenames=['expt04'],
        include_MUs=False,
        time_embed=True,
        num_lags=num_lags)
    
    # create a DataLoader object
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
    
    # iterate through the first 10 batches of the DataLoader
    for i, data in enumerate(dataloader):
        # validate the shape of the batches
        assert(data['stim'].shape == torch.Size([10, 360]))
        if i == 10:
            break
