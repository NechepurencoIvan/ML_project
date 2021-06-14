import os
from numpy.fft import fft2, fftshift, ifftshift
from torch.utils.data import Dataset
import re
import torch

class SeismogramDataset(Dataset):
    """
    Seismogram dataset.
    Implements the interface of torch.utils.dataset.Dataset
    Names of the files must contain numerals, due to the indexing issues
    File with the least numeral is considered to be a 'head' of the dataset
    
    """

    def __init__(self, root_dir, ellipsis, do_transform=True):
        """
        :param root_dir (string): 
            A root directory of the dataset. Following layout is assumed:
                /root
                    /seismograms
                    /masks
        :param scaling_factor (float):
            Constant multiplier for seismograms. Default=1e7
            
        """
        self.root_dir    = os.path.join(os.getcwd(), root_dir)
        self.data_dir    = os.path.join(self.root_dir, 'seismograms')
        self.mesh_dir    = os.path.join(self.root_dir, 'masks')

        self.seismograms = [(file, re.findall(r'(\d+)', file)[-1]) for file in os.listdir(self.data_dir)]
        self.seismograms = [x[0] for x in list(sorted(self.seismograms, key= lambda x: x[1]))][ellipsis[0]:ellipsis[1]]
        
        self.masks = [(file, re.findall(r'(\d+)', file)[-1]) for file in os.listdir(self.mesh_dir)]
        self.masks = [x[0] for x in list(sorted(self.masks, key= lambda x: x[1]))][ellipsis[0]:ellipsis[1]]
        self.do_transform = do_transform

    def __len__(self):
        return len(self.seismograms)

    def __getitem__(self, idx):

        sname = os.path.join(self.data_dir, self.seismograms[idx])
        mname = os.path.join(self.mesh_dir, self.masks[idx])

        s = None
        if not self.do_transform:
          s = torch.from_numpy(np.load(sname)).float()
        else:
          s = torch.from_numpy(transform_seismogram(np.load(sname))).float()

        m = torch.from_numpy(np.load(mname)).long()

        return s, m
      
class AutoencodeDataset(Dataset):
    """
    Seismogram dataset.
    Implements the interface of torch.utils.dataset.Dataset
    Names of the files must contain numerals, due to the indexing issues
    File with the least numeral is considered to be a 'head' of the dataset
    
    """

    def __init__(self, root_dir, ellipsis):
        """
        :param root_dir (string): 
            A root directory of the dataset. Following layout is assumed:
                /root
                    /seismograms
                    /masks
        :param scaling_factor (float):
            Constant multiplier for seismograms. Default=1e7
            
        """
        self.root_dir    = os.path.join(os.getcwd(), root_dir)
        self.data_dir    = os.path.join(self.root_dir, 'seismograms')
        self.mesh_dir    = os.path.join(self.root_dir, 'masks')

        self.seismograms = [(file, re.findall(r'(\d+)', file)[-1]) for file in os.listdir(self.data_dir)]
        self.seismograms = [x[0] for x in list(sorted(self.seismograms, key= lambda x: x[1]))][ellipsis[0]:ellipsis[1]]
        
        self.masks = [(file, re.findall(r'(\d+)', file)[-1]) for file in os.listdir(self.mesh_dir)]
        self.masks = [x[0] for x in list(sorted(self.masks, key= lambda x: x[1]))][ellipsis[0]:ellipsis[1]]

    def __len__(self):
        return len(self.seismograms)

    def __getitem__(self, idx):

        sname = os.path.join(self.data_dir, self.seismograms[idx])
        s = torch.from_numpy(np.load(sname)).float()

        return s, s.ravel()
