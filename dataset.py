import os, glob
import numpy as np
import json
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import torch, torchvision
import torchvision.tv_tensors

class CityScapesDataset(Dataset):
    
    def __init__(self, images, masks, polygons, trainId2label, sample_frac = 10):

        # self.images = images.sample(frac = sample_frac)
        self.images = images[:sample_frac]
        self.masks = masks[:sample_frac]
        self.polygons = polygons[:sample_frac]
        self.label2id = trainId2label

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        labels = torch.Tensor([self.label2id[label] for label in labels])
        labels = labels.to(dtype=torch.int64)
        
        transforms = torchvision.transforms.PILToTensor()

        orig_image = Image.open(self.images[index])
        input_feature = transforms(orig_image)

        orig_mask = Image.open(self.masks[index])
        input_mask = transforms(orig_mask)
        
        orig_polygons = self.polygons[index]
        with open(orig_polygons) as json_file:
            polygon_coords = json.load(json_file)

        seg_polygons = pd.DataFrame(polygon_coords)

        # masks = torchvision.tv_tensors.Mask(self.masks)

        # bounding_box = torchvision.tv_tensors.BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=self.images[0].size[::-1])

        return input_feature, input_mask

class CityScapesFiles:

    def __init__(self):

        self.test_features = []
        self.train_features = []
        self.val_features = []

        self.test_masks = []
        self.train_masks = []
        self.val_masks = []

        self.test_labels = []
        self.train_labels = []
        self.val_labels = []

    def load_features(self, dir):

        #--------------------TRAIN---------------------------#

        split_dir = dir + '/' + 'train'
        self.cities = os.listdir(split_dir)

        self.train_features = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*.png"))
            self.train_features += city_files

        # print(len(self.train_features))

        #--------------------TEST---------------------------#

        split_dir = dir + '/' + 'test'
        self.cities = os.listdir(split_dir)

        self.test_features = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*.png"))
            self.test_features += city_files

        # print(len(self.test_features))

        #--------------------VAL---------------------------#

        split_dir = dir + '/' + 'val'
        self.cities = os.listdir(split_dir)

        self.val_features = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*.png"))
            self.val_features += city_files

        # print(len(self.val_features))

        return (self.train_features, self.test_features, self.val_features)

    def load_masks(self, dir):

        #--------------------TRAIN---------------------------#

        split_dir = dir + '/' + 'train'
        self.cities = os.listdir(split_dir)

        self.train_masks = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*_instanceTrainIds.png"))
            self.train_masks += city_files

        print(len(self.train_masks))

        #--------------------TEST---------------------------#

        split_dir = dir + '/' + 'test'
        self.cities = os.listdir(split_dir)

        self.test_masks = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*_instanceTrainIds.png"))
            self.test_masks += city_files

        print(len(self.test_masks))

        #--------------------VAL---------------------------#

        split_dir = dir + '/' + 'val'
        self.cities = os.listdir(split_dir)

        self.val_masks = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*_instanceTrainIds.png"))
            self.val_masks += city_files

        print(len(self.val_masks))

        return (self.train_masks, self.test_masks, self.val_masks)
    
    def load_labels(self, dir):

        #--------------------TRAIN---------------------------#

        split_dir = dir + '/' + 'train'
        self.cities = os.listdir(split_dir)

        self.train_labels = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*.json"))
            self.train_labels += city_files

        print(len(self.train_labels))

            # for file in file_list:
            #     with open(file) as json_file:

            #         label = json.load(json_file)
            #         label['id'] = Path(file).stem.replace('_gtFine_polygons', '')

            #         self.train_labels.append(label)

        #--------------------TEST---------------------------#

        split_dir = dir + '/' + 'test'
        self.cities = os.listdir(split_dir)

        self.test_labels = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*.json"))
            self.test_labels += city_files

        print(len(self.test_labels))

        #--------------------VAL---------------------------#

        split_dir = dir + '/' + 'val'
        self.cities = os.listdir(split_dir)

        self.val_labels = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*.json"))
            self.val_labels += city_files

        print(len(self.val_labels))

        # self.train_labels = pd.DataFrame(self.train_labels)
        # self.test_labels = pd.DataFrame(self.test_labels)
        # self.val_labels = pd.DataFrame(self.val_labels)

        return (self.train_labels, self.test_labels, self.val_labels)