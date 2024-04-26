import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from MNIST import MNIST
from torchvision import transforms
from Configure import model_configs

def load_train_data(data_dir):

    x_train_wr = np.load(os.path.join(data_dir, f'x_train_wr.npy'))
    x_train_sp = np.load(os.path.join(data_dir, f'x_train_sp.npy'))

    y_train = pd.read_csv(os.path.join(data_dir, f'y_train.csv'))
    y_train.drop(['row_id'], axis=1, inplace=True)
    y_train = y_train.values

    return x_train_wr, x_train_sp, y_train

def train_valid_split(x_train_wr, x_train_sp, y_train, val_ratio=0.2):

    assert x_train_wr.shape[0] == x_train_sp.shape[0], 'Written and Spoken inputs need same number of samples.'

    split_index = int(val_ratio * x_train_wr.shape[0])

    x_valid_wr = x_train_wr[:split_index]
    x_valid_sp = x_train_sp[:split_index]
    y_valid = y_train[:split_index]

    x_train_wr = x_train_wr[split_index:]
    x_train_sp = x_train_sp[split_index:]
    y_train = y_train[split_index:]

    return x_train_wr, x_train_sp, y_train, x_valid_wr, x_valid_sp, y_valid

def load_testing_images(data_dir):

    x_test_wr = np.load(os.path.join(data_dir, f'x_test_wr.npy'))
    x_test_sp = np.load(os.path.join(data_dir, f'x_test_sp.npy'))

    return x_test_wr, x_test_sp

def custom_dataloader(data_wr, data_sp, label, batch_size, train):

    data_wr = data_wr.reshape(-1, 1, 28, 28)
    data_wr = data_wr.transpose((0, 2, 3, 1))

    train_wr_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        ])
    
    train_sp_transform = transforms.Compose([
        transforms.ToTensor()
        ])
    
    test_wr_transform = transforms.Compose([
        transforms.ToTensor()
        ])
    
    test_sp_transform = transforms.Compose([
        transforms.ToTensor()
        ])
    
    if train:
        train_dataset = MNIST(data_wr, data_sp, label, wr_transform=train_wr_transform, sp_transform = train_sp_transform)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=model_configs.num_workers)
        return train_loader

    elif not train:
        test_dataset = MNIST(data_wr, data_sp, wr_transform=test_wr_transform, sp_transform = test_sp_transform)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=model_configs.num_workers)
        return test_loader