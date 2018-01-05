import torchvision.datasets as dset
import torchvision.transforms as transforms
import pickle
import torch.utils.data
import pandas as pd
from glob import glob

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, index):
        return torch.Tensor(self.imgs[index]).unsqueeze(0)

    def __len__(self):
        return len(self.imgs)

class ImgAndGroundTruthDataset(torch.utils.data.Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, index):
        return (torch.Tensor(self.imgs[index][0]).unsqueeze(0), torch.Tensor(self.imgs[index][1]).unsqueeze(0))

    def __len__(self):
        return len(self.imgs)

def loadHeat(size=32, boundary=True, test=False, ground_truth=False):
    if size not in [4, 8, 16, 32, 64]:
        raise ValueError("Dataset does not exist for size %d x %d" % (size, size))

    if ground_truth:
       bc_csv = sorted(glob("datasets/%d*/BC/*.csv"%size))
       eq_csv = sorted(glob("datasets/%d*/field/*.csv"%size))
       dataset = [(pd.read_csv(f,delimiter=",",header=None),pd.read_csv(g,delimiter=",",header=None)) for f,g in zip(bc_csv,eq_csv)]
       return ImgAndGroundTruthDataset(dataset)

    csv_files = glob("datasets/%d*/BC/*.csv"%size)

    dataset = [pd.read_csv(f,delimiter=",", header=None) for f in csv_files]

    return ImgDataset(dataset)
