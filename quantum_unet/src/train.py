import os
import time
from glob import glob
import pandas as pd
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms

from ..data import DriveDataset
from ..Qmodel_MK2 import build_unet
from ..loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
from torchvision import transforms
transform = transforms.ToPILImage()
create_dir("output_img")
def train(model, loader, optimizer, loss_fn, device, PATH, n):
    epoch_loss = 0.0
    model.train()
    
    v = 0
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
         
        optimizer.zero_grad()
        y_pred = model(x)
        torch.save(model, PATH)
        if (v==0):
            img = transform(y_pred[0])
            name = "output_img"+str(n)+".png"
            v +=1
            paths = os.path.join("/gpfs/fs1/home/l/lcl_scp3162/lcl_scp3162u002/SOSCIP_training/output_img",name)
            img.save(paths)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":
    seeding(42)
    create_dir("filess")
    train_x = glob("/gpfs/fs1/home/l/lcl_scp3162/lcl_scp3162u002/SOSCIP_training/data_set/Pcd_512/*")
    train_y = glob("/gpfs/fs1/home/l/lcl_scp3162/lcl_scp3162u002/SOSCIP_training/data_set/Raw_512/*")
    print("x_len", len(train_x))
    # Hyperparameters
    size = (512, 512)
    batch_size = 13
    num_epochs = 18
    lr = 0.004
    checkpoint_path = "/gpfs/fs1/home/l/lcl_scp3162/lcl_scp3162u002/SOSCIP_training/filess/checkpoint.pth"
    
    train_dataset = DriveDataset(train_x, train_y)
    print(len(train_dataset))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False, sampler=None,
                              batch_sampler=None, num_workers=13, collate_fn=None,
                              pin_memory=False, drop_last=False, timeout=0,
                              worker_init_fn=None, persistent_workers=False)
    device = torch.device("cuda")
    model = build_unet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    
    training_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, loss_fn, device, checkpoint_path, epoch)
        print(epoch, train_loss)
        training_losses.append(train_loss)
    
    
