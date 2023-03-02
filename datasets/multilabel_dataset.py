import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
import torch


class MultiLabelDataset(Dataset):
    def __init__(self, list_filename, training):
        self.training = training
        
        self.data = []
        self.one_labels = []
        self.two_labels = []
        
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        
        for split in splits:
            self.data.append(split[0])
            self.one_labels.append(int(split[1]))
            self.two_labels.append(int(split[2]))

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.load_image(self.data[index])
        
        processed = get_transform()
        
        image = processed(image)
        
        dict_data = {
            'img': image,
            'labels': {
                'one_label': self.one_labels[index],
                'two_label': self.two_labels[index]
            }
        }
        
        return dict_data