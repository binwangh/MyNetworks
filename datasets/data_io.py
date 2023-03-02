import torchvision.transforms as transforms

# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#     "test": transforms.Compose([transforms.Resize(256),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

## chestnut
Chestnut_transform = {
    "train": transforms.Compose([transforms.Resize([160, 160]),
                                #  transforms.RandomHorizontalFlip(p=0.5),
                                #  transforms.RandomVerticalFlip(p=0.5),
                                #  transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "test": transforms.Compose([transforms.Resize([160, 160]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

## Lemon
Lemon_transform = {
    "train": transforms.Compose([transforms.Resize([320, 320]),
                                #  transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "test": transforms.Compose([transforms.Resize([320, 320]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

## Lemon
JunZao_transform = {
    "train": transforms.Compose([transforms.Resize([160, 160]),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "test": transforms.Compose([transforms.Resize([160, 160]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}



import numpy as np
import re
import torchvision.transforms as transforms


def get_transform():
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    return transforms.Compose([
        transforms.Resize([160, 160]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines