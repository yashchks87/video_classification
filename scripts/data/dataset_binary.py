import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../data/')
from sklearn.model_selection import train_test_split

def prep_csv(df):
    df['path'] = df['youtube_id'].apply(lambda x: '../../kinetics_frames/' + x + '/')
    df['path_exists'] = df['path'].apply(lambda x: os.path.exists(x))
    df = df[df['path_exists'] == True]
    label_mapper = {
        'playing harp' : 0,
        'snowkiting' : 1
    }
    df['label_encoded'] = df['label'].map(label_mapper)
    return df

class VideoDataset(Dataset):
    def __init__(self, mapper, img_size):
        self.paths = [x[0] for x in mapper]
        self.labels = [x[1] for x in mapper]
        self.norm = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        frames = self.paths[index]
        label = torch.Tensor([self.labels[index]]).float()
        tensor_holder = torch.empty(20, 3, self.img_size[0], self.img_size[1], dtype=torch.float32)
        count = 0
        for path in range(len(frames)):
            img = torchvision.io.read_image(frames[path])
            img = torchvision.transforms.functional.resize(img, self.img_size)
            img = img / 255
            img = img.float()
            img = self.norm(img)
            tensor_holder[count] = img
            count += 1
        tensor_holder = tensor_holder.permute(1, 0, 2, 3)
        return tensor_holder, label


class GetData():
    def __init__(self, label_path):
        self.label_path = label_path
        self.df = pd.read_csv(label_path)
        self.df = prep_csv(self.df)
    
    def split_datasets(self, test_size = 0.2, random_state = 42):
        train, test = train_test_split(self.df, test_size = test_size, random_state = random_state)
        train, val = train_test_split(train, test_size = test_size, random_state = random_state)
        self.train, self.val, self.test = train, val, test
    
    def gather_frames(self, df):
        paths = df['path'].values.tolist()
        labels = df['label_encoded'].values.tolist()
        label_path_mapper = []
        for x in range(len(paths)):
            frames = glob.glob(paths[x] + '*.jpg')
            label_path_mapper.append([frames, labels[x]])
        return label_path_mapper

    def perp_dataset(self):
        self.train_final = self.gather_frames(self.train)
        self.val_final = self.gather_frames(self.val)
        self.test_final = self.gather_frames(self.test)
    
    def create_dataset(self, img_size = (128, 128)):
        self.train_dataset = VideoDataset(self.train_final, img_size)
        self.val_dataset = VideoDataset(self.val_final, img_size)
        self.test_dataset = VideoDataset(self.test_final, img_size)
    
    def create_loaders(self, batch_size = 32, num_workers = 10, shuffle_train = True, shuffle_val = True):
        self.batch_size = batch_size
        self.num_workers = num_workers 
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = shuffle_train, num_workers = num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size = batch_size, shuffle = shuffle_val, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = shuffle_val, num_workers=num_workers)
    
    def return_reference_obj(self):
        return self

    def return_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader