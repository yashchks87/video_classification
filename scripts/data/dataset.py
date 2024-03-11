import torch
import torchvision
from torch.utils.data import Dataset
import glob
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../data/')
# from data.encode_labels import BinaryLabels, EncodeLabels
from sklearn.model_selection import train_test_split



class VideoDataset(Dataset):
    def __init__(self, paths, labels, img_size = (128, 128)):
        self.paths = paths
        self.labels = labels
        self.img_size = img_size
        self.norm = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = glob.glob(self.paths[idx] + '/*.jpg')
        temp = torch.zeros(20, 3, self.img_size[0], self.img_size[1])
        count = 0
        for frame in frames:
            img = torchvision.io.read_image(frame)
            img = torchvision.transforms.functional.resize(img, (self.img_size[0], self.img_size[1]))  
            img = img / 255
            img = self.norm(img)
            temp[count, :, :, :] = img
            count += 1
        temp = temp.reshape(3, 20, self.img_size[0], self.img_size[1])
        return temp, self.labels[idx]
    
def find_top_2_classes(df):
    most_labels = df['label'].value_counts().keys()[:2]
    df = df[df['label'].isin(most_labels)]
    label_dict = {
        most_labels[0] : 0,
        most_labels[1] : 1
    }
    df['label_encoded'] = df['label'].map(label_dict)
    return df, most_labels.tolist()

def read_and_fix_csv(df, root_path):
    df['path'] = df['youtube_id'].apply(lambda x: root_path + x + '/')
    df['path_exists'] = df['path'].apply(lambda x: os.path.exists(x))
    df = df[df['path_exists'] == True]
    return df
    
class GetData():
    def __init__(self, dir_path, label_path, is_binary = True, label_list = None):
        self.dir_path = dir_path
        self.label_path = label_path
        self.is_binary = is_binary
        self.label_list = label_list
        self.label_df = pd.read_csv(label_path)
        self.label_df = read_and_fix_csv(self.label_df, dir_path)
        if is_binary and label_list is None:
            self.label_df, self.most_labels = find_top_2_classes(self.label_df)

    def split_datasets(self, test_size = 0.01, random_state = 42):
        train, test = train_test_split(self.label_df, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train, test_size=0.01, random_state=random_state)
        self.train, self.val, self.test = train, val, test

    def list_all_frames(self):
        pass
    
    def create_datasets(self, img_size = (128, 128)):
        pass


# def fix_labels(label_path):
#     labels = np.loadtxt(label_path, delimiter=',', dtype=str)
#     labels = dict(zip(labels[1:][:, 1].tolist(), labels[1:][:, 0].tolist()))
#     return labels

# class VideoDataset(Dataset):
#     def __init__(self, paths, labels, img_size = (128, 128)):
#         self.paths = paths
#         self.labels = labels
#         self.img_size = img_size
#         self.norm = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   
        
#     def __len__(self):
#         return len(self.dirs)

#     def __getitem__(self, idx):
#         frames = glob.glob(self.dirs[idx] + '/*.jpg')
#         temp = torch.zeros(20, 3, self.img_size[0], self.img_size[1])
#         count = 0
#         for frame in frames:
#             img = torchvision.io.read_image(frame)
#             img = torchvision.transforms.functional.resize(img, (self.img_size[0], self.img_size[1]))  
#             img = img / 255
#             img = self.norm(img)
#             temp[count, :, :, :] = img
#             count += 1
#         temp = temp.reshape(3, 20, self.img_size[0], self.img_size[1])
#         label_finder = self.dirs[idx].split('/')[-1].split('_')[0]
#         return temp, label_finder


# class GetData():
#     def __init__(self, dir_path, label_path):
#         self.dir_path = dir_path
#         self.label_path = label_path

#     def preprocess(self, is_binary = True):
#         self.dirs = [d for d in os.listdir(self.dir_path) if os.path.isdir(os.path.join(self.dir_path, d))]
#         self.dirs = [self.dir_path + d for d in self.dirs]
#         if is_binary:
#             # pass
#             binary_obj = BinaryLabels(self.label_path)
#             binary_obj.process_values()
#             self.binary_df = binary_obj.get_values()
#             updated_dirs, updated_labels = [], []
#             # for x in self.dirs:
#             #     temp = x.split('/')[-1].split('_')[0]
#             #     if temp in self.binary_df['key'].tolist():
#             #         updated_dirs.append(x)
#             #         updated_labels.append(self.binary_df[temp])
#             # l_, c = {}, 0
#             # for x in set(updated_labels):
#             #     if x not in l_:
#             #         l_[x] = c
#             #         c += 1
#             # self.label_mapper = l_
#             # encoded_labels = [l_[x] for x in updated_labels]
#             # final = dict(zip(updated_dirs, encoded_labels))
#         # else:
#         #     encode_obj = EncodeLabels(self.label_path)
#         #     encode_obj.encode_labels()
#         #     self.id_labels = encode_obj.get_values()
#         #     final = {}
#         #     for x in self.dirs:
#         #         if x.split('/')[-1].split('_')[0] in self.id_labels:
#         #             final[x] = self.id_labels[x.split('/')[-1].split('_')[0]]
#         # self.key_label = pd.DataFrame({'key': list(final.keys()), 'label': list(final.values())})
    
#     def split_dataset(self):
#         train, test = train_test_split(self.key_label, test_size=0.01, random_state=42)
#         train, val = train_test_split(train, test_size=0.01, random_state=42)
#         self.train, self.val, self.test = train, val, test
    
#     def create_datasets(self):
#         self.train_dataset = VideoDataset(self.train['key'].tolist(), self.train['label'].tolist())
#         self.val_dataset = VideoDataset(self.val['key'].tolist(), self.val['label'].tolist())
#         self.test_dataset = VideoDataset(self.test['key'].tolist(), self.test['label'].tolist())
#         return self.train_dataset, self.val_dataset, self.test_dataset



# class DataStats():
#     def __init__(self, csv_path):
#         self.csv_path = csv_path
#         self.df = pd.read_csv(csv_path)
    
#     def remove_non_existing(self, df):
#         df = df[df['dir_exists'] == True]
#         return df
    
#     def check_weather_dir_exists(self, df):
#         df = df['paths'].apply(lambda x: os.path.exists(x))

#     def find_max_occurances(self):
#         pass




