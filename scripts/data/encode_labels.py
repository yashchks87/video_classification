import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class EncodeLabels():
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
    def InitializeLabelEncoder(self):
        self.le = LabelEncoder()

    def read_file(self):
        self.file = np.loadtxt(self.csv_path, delimiter=',', dtype=str) 
        self.labels = self.file[1:][:, 0].tolist()
        self.id = self.file[1:][:, 1].tolist()
        self.id_labels = dict(zip(self.id, self.labels))
    
    def encode_labels(self):
        self.InitializeLabelEncoder()
        self.read_file()
        self.le.fit(self.labels)
        self.encoded_labels = self.le.transform(self.labels)
        self.id_encoded_labels = dict(zip(self.id, self.encoded_labels))
    
    def get_values(self):
        return self.id_encoded_labels

def find_top_categories(labels):
    keys, counts = np.unique(labels, return_counts=True)
    sorted = np.argsort(counts)[::-1]
    return keys[sorted[0]], keys[sorted[1]]

class BinaryLabels():
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def read_file(self):
        self.file = np.loadtxt(self.csv_path, delimiter=',', dtype=str) 
        self.labels = [x.replace('"', '') for x in self.file[1:][:, 0].tolist()]
        self.first, self.second = find_top_categories(self.labels)
        self.id = self.file[1:][:, 1].tolist()
        # self.id_labels = dict(zip(self.id, self.labels))
        self.id_labels = pd.DataFrame({'key': self.id, 'label': self.labels})

    def process_values(self):
        self.read_file()
        tracker = [self.first, self.second]
        self.id_labels = self.id_labels[self.id_labels['label'].isin(tracker)]


    def get_values(self):
        return self.id_labels
    