import os, glob
import json
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import torch, torchvision
import torchvision.tv_tensors

class CityScapedDataset(Dataset):
    
    def __init__(self, images, masks, annotations, label2id, sample_frac = 10):

        # self.images = images.sample(frac = sample_frac)
        self.images = images.iloc[:sample_frac]
        self.masks = masks.iloc[:sample_frac]
        self.annotations = annotations.iloc[:sample_frac]
        self.label2id = label2id

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        labels = torch.Tensor([self.label2id[label] for label in labels])
        labels = labels.to(dtype=torch.int64)
        
        masks = torchvision.tv_tensors.Mask(self.masks)

        bounding_box = torchvision.tv_tensors.BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=self.images[0].size[::-1])

        return super().__getitem__(index)

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

            file_list = glob.glob(os.path.join(split_dir, city, "*.png"))

            for file in file_list:

                img = Image.open(file)

                self.train_features.append(img)
                img.close()

        #--------------------TEST---------------------------#

        split_dir = dir + '/' + 'test'
        self.cities = os.listdir(split_dir)

        self.test_features = []

        for city in self.cities:

            file_list = glob.glob(os.path.join(split_dir, city, "*.png"))

            for file in file_list:
                
                img = Image.open(file)

                self.test_features.append(img)
                img.close()

        #--------------------VAL---------------------------#

        split_dir = dir + '/' + 'val'
        self.cities = os.listdir(split_dir)

        self.val_features = []

        for city in self.cities:

            file_list = glob.glob(os.path.join(split_dir, city, "*.png"))

            for file in file_list:

                img = Image.open(file)

                self.val_features.append(img)
                img.close()

        return (self.train_features, self.test_features, self.val_features)

    def load_masks(self, dir):

        #--------------------TRAIN---------------------------#

        split_dir = dir + '/' + 'train'
        self.cities = os.listdir(split_dir)

        self.train_masks = []

        for city in self.cities:

            file_list = glob.glob(os.path.join(split_dir, city, "*_instanceTrainIds.png"))

            for file in file_list:

                img = Image.open(file)

                self.train_masks.append(img)
                # img.close()

        #--------------------TEST---------------------------#

        split_dir = dir + '/' + 'test'
        self.cities = os.listdir(split_dir)

        self.test_masks = []

        for city in self.cities:

            file_list = glob.glob(os.path.join(split_dir, city, "*_instanceTrainIds.png"))

            for file in file_list:
                
                img = Image.open(file)

                self.test_masks.append(img)
                img.close()

        #--------------------VAL---------------------------#

        split_dir = dir + '/' + 'val'
        self.cities = os.listdir(split_dir)

        self.val_masks = []

        for city in self.cities:

            file_list = glob.glob(os.path.join(split_dir, city, "*_instanceTrainIds.png"))

            for file in file_list:

                img = Image.open(file)

                self.val_masks.append(img)
                img.close()

        return (self.train_masks, self.test_masks, self.val_masks)
    
    def load_labels(self, dir):

        #--------------------TRAIN---------------------------#

        split_dir = dir + '/' + 'train'
        self.cities = os.listdir(split_dir)

        self.train_labels = []

        for city in self.cities:

            file_list = glob.glob(os.path.join(split_dir, city, "*.json"))

            for file in file_list:
                with open(file) as json_file:

                    label = json.load(json_file)
                    label['id'] = Path(file).stem.replace('_gtFine_polygons', '')

                    self.train_labels.append(label)

        #--------------------TEST---------------------------#

        split_dir = dir + '/' + 'test'
        self.cities = os.listdir(split_dir)

        self.test_labels = []

        for city in self.cities:

            file_list = glob.glob(os.path.join(split_dir, city, "*.json"))

            for file in file_list:
                with open(file) as json_file:

                    label = json.load(json_file)
                    label['id'] = Path(file).stem.replace('_gtFine_polygons', '')

                    self.test_labels.append(label)

        #--------------------VAL---------------------------#

        split_dir = dir + '/' + 'val'
        self.cities = os.listdir(split_dir)

        self.val_labels = []

        for city in self.cities:

            file_list = glob.glob(os.path.join(split_dir, city, "*.json"))

            for file in file_list:
                with open(file) as json_file:

                    label = json.load(json_file)
                    label['id'] = Path(file).stem.replace('_gtFine_polygons', '')

                    self.val_labels.append(label)

        # print(len(self.val_labels))
        # print(self.test_labels[0])

        self.train_labels = pd.DataFrame(self.train_labels)
        self.test_labels = pd.DataFrame(self.test_labels)
        self.val_labels = pd.DataFrame(self.val_labels)

        return (self.train_labels, self.test_labels, self.val_labels)