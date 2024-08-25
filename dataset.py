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
    
    def __init__(self, images, masks, polygons, train_labels, sample_frac = 10, instances_only = False):

        # self.images = images.sample(frac = sample_frac)
        self.images = images[:sample_frac]
        self.masks = masks[:sample_frac]
        self.polygons = polygons[:sample_frac]
        self.train_labels = train_labels
        self.instances_only = instances_only

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        # labels = torch.Tensor([self.label2id[label] for label in labels])
        # labels = labels.to(dtype=torch.int64)
        
        transforms = torchvision.transforms.ToTensor()
        # transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((512, 256)), 
        #                                              torchvision.transforms.PILToTensor()])

        # img_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
        #                                              torchvision.transforms.Normalize(mean=mean, 
        #                                                                               std=std)
        #                                              ])
        
        # mask_transforms = torchvision.transforms.Compose([
        #   torchvision.transforms.Lambda(lambda image: torch.tensor(np.array(resized_mask).astype(np.float32)).unsqueeze(0)),
        #   torchvision.transforms.Normalize((0.5,), (0.5,))])

        mask_transforms = torchvision.transforms.Lambda(lambda resized_mask: torch.from_numpy(np.array(resized_mask).astype(np.float32)).unsqueeze(0))

        # mask_transforms = torchvision.transforms.PILToTensor()

        orig_image = Image.open(self.images[index])
        resized_img = orig_image.resize((1024, 512))
        # input_feature = img_transforms(resized_img)
        input_feature = transforms(resized_img)

        mean, std = input_feature.mean([1,2]), input_feature.std([1,2])

        img_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                         ])

        input_feature = img_transforms(resized_img)

        # print(input_feature)

        orig_mask = Image.open(self.masks[index])
        resized_mask = orig_mask.resize((1024, 512))
        input_mask = mask_transforms(resized_mask)

        input_mask = np.array(input_mask)

        self.instances_only = True

        if self.instances_only:

            input_mask[input_mask < 1000] = 0

        else:

            input_mask[input_mask > 1000] = input_mask[input_mask > 1000] / 1000
            input_mask[input_mask == -1] = 19
            input_mask[input_mask == 255] = 20

        # print(input_mask)

        input_mask = torch.from_numpy(input_mask)

        input_mask = input_mask.type(torch.LongTensor)

        torch.set_printoptions(profile="full")
        print(input_mask)

        input_mask = torch.nn.functional.one_hot(input_mask, 21).transpose(0, 3).squeeze(-1)

        print(input_mask.shape)

        masks = torchvision.tv_tensors.Mask(torch.concat([torchvision.tv_tensors.Mask(input_mask, dtype=torch.bool)]))
        # print(masks[0])

        for mask in masks[11:18]:
            print(mask.unsqueeze(0).shape)
            mask = mask.unsqueeze(0)
            bounding_box = torchvision.tv_tensors.BoundingBoxes(data=torchvision.ops.masks_to_boxes(mask), format=torchvision.tv_tensors.BoundingBoxFormat("XYXY"), canvas_size=(1024, 512))

            print(bounding_box)

        # bounding_box = torchvision.tv_tensors.BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format=torchvision.tv_tensors.BoundingBoxFormat("XYXY"), canvas_size=input_mask.size[::-1])

        print("AAAAAAAAAA")

        # bounding_boxes = torchvision.ops.masks_to_boxes(input_mask)
        # print(bounding_boxes)

        # print(input_mask.shape)

        # torch.set_printoptions(profile="full")
        # print(input_mask)

        # input_mask = np.array(input_mask)
        # input_mask = torch.from_numpy(input_mask)
        # input_mask = input_mask.type(torch.LongTensor)
        
        # orig_polygons = self.polygons[index]
        # with open(orig_polygons) as json_file:
        #     polygon_coords = json.load(json_file)

        # seg_polygons = pd.DataFrame(polygon_coords)

        # masks = torchvision.tv_tensors.Mask(self.masks)

        # bounding_box = torchvision.tv_tensors.BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=self.images[0].size[::-1])

        # return input_feature, input_mask
        
        # torch.set_printoptions(profile="full")
        # print(torch.tensor(input_mask, dtype = torch.long))

        # print(input_feature.shape)
        # print(input_mask.shape)
    
        return {"image": torch.tensor(input_feature, dtype = torch.float32), 
                "mask": torch.tensor(input_mask, dtype = torch.float32)
                }

        # return {"image": input_feature,
        #         "mask": input_mask
        #         }

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

        # print(len(self.train_masks))

        #--------------------TEST---------------------------#

        split_dir = dir + '/' + 'test'
        self.cities = os.listdir(split_dir)

        self.test_masks = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*_instanceTrainIds.png"))
            self.test_masks += city_files

        # print(len(self.test_masks))

        #--------------------VAL---------------------------#

        split_dir = dir + '/' + 'val'
        self.cities = os.listdir(split_dir)

        self.val_masks = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*_instanceTrainIds.png"))
            self.val_masks += city_files

        # print(len(self.val_masks))

        return (self.train_masks, self.test_masks, self.val_masks)
    
    def load_labels(self, dir):

        #--------------------TRAIN---------------------------#

        split_dir = dir + '/' + 'train'
        self.cities = os.listdir(split_dir)

        self.train_labels = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*.json"))
            self.train_labels += city_files

        # print(len(self.train_labels))

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

        # print(len(self.test_labels))

        #--------------------VAL---------------------------#

        split_dir = dir + '/' + 'val'
        self.cities = os.listdir(split_dir)

        self.val_labels = []

        for city in self.cities:

            city_files = glob.glob(os.path.join(split_dir, city, "*.json"))
            self.val_labels += city_files

        # print(len(self.val_labels))

        # self.train_labels = pd.DataFrame(self.train_labels)
        # self.test_labels = pd.DataFrame(self.test_labels)
        # self.val_labels = pd.DataFrame(self.val_labels)

        return (self.train_labels, self.test_labels, self.val_labels)