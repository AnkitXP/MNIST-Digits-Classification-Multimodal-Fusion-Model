import numpy as np
from typing import Any, Tuple
from torch.utils.data import Dataset
from PIL import Image

class MNIST(Dataset):
    def __init__(self, data_wr, data_sp, target, wr_transform=None, sp_transform = None, target_transform=None):
        super().__init__()
        self.data_wr = data_wr
        self.data_sp = data_sp
        self.target = target
        self.wr_transform = wr_transform
        self.sp_transform = sp_transform
        self.target_transform = target_transform
        
    def __len__(self):
        return self.data_wr.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:

        wr, sp, target = self.data_wr[index], self.data_sp[index], self.target[index]

        wr = Image.fromarray(wr)
        
        if self.wr_transform is not None:
            wr = self.transform(wr)

        if self.sp_transform is not None:
            sp = self.transform(sp)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return wr, sp, target