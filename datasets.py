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

def loadHeat(size=32, boundary=True, test=False, ground_truth=False):
    if size not in [4, 8, 16, 32, 64]:
        raise ValueError("Dataset does not exist for that size")

    if size in [4,8,16, 32]:
        csv_files = glob("datasets/%d*/BC/*.csv"%size)

    dataset = [pd.read_csv(f, delimiter=",", header=None) for f in csv_files]

    return ImgDataset(dataset)
