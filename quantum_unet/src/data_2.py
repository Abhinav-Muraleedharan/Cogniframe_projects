import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np 
import torch 

class TwoFolderDataset(Dataset):
    def __init__(self, folder1, folder2, transform):
        self.folder1 = folder1
        self.folder2 = folder2
        self.transform = transform
        # Load all file names
        self.folder1_files = [os.path.join(folder1, f) for f in os.listdir(folder1)]
        self.folder2_files = [os.path.join(folder2, f) for f in os.listdir(folder2)]
        self.file_names = self.folder1_files + self.folder2_files

        # Labels: 0 for folder1 and 1 for folder2
        self.labels = [0] * len(self.folder1_files) + [1] * len(self.folder2_files)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):

        image = cv2.imread(self.folder1_files[index], cv2.IMREAD_GRAYSCALE)
        # image = self.transform(image)
        image = cv2.resize(image, (512, 512))
        image = image/255.0
        image = np.expand_dims(image,axis = 0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        
        mask = cv2.imread(self.folder2_files[index],cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))
        # mask = self.transform(mask)              
        mask = mask/255.0
        mask = np.expand_dims(mask,axis = 0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        return image, mask

 
            


