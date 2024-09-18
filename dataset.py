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
    
    def __init__(self, images, masks, polygons, sample_frac = 10, instances_only = False):

        # self.images = images.sample(frac = sample_frac)
        self.images = images[:sample_frac]
        self.masks = masks[:sample_frac]
        self.polygons = polygons[:sample_frac]
        self.instances_only = instances_only
        self.resize_dim = (512, 256) #(1024, 512)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        transforms = torchvision.transforms.ToTensor()

        mask_transforms = torchvision.transforms.Lambda(lambda resized_mask: torch.from_numpy(np.array(resized_mask).astype(np.float32)).unsqueeze(0))

        # mask_transforms = torchvision.transforms.PILToTensor()

        orig_image = Image.open(self.images[index])
        resized_img = orig_image.resize(self.resize_dim)
        # input_feature = img_transforms(resized_img)
        input_feature = transforms(resized_img)

        mean, std = input_feature.mean([1,2]), input_feature.std([1,2])

        img_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                        #  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                         torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #(0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                         ])

        input_feature = img_transforms(resized_img)

        # print(input_feature)

        orig_mask = Image.open(self.masks[index])
        resized_mask = orig_mask.resize(self.resize_dim)
        input_mask = mask_transforms(resized_mask)

        input_mask = np.array(input_mask)

        if self.instances_only:

            input_mask[input_mask < 1000] = 0

            instances = np.unique(input_mask)
            instances = np.delete(instances, np.where(instances == 0.))
            # instances = np.concatenate([[0], instances])

            new_mask = torch.zeros(input_mask.shape)
            
            for id in instances:

                # print(id)
                temp_mask = np.empty(input_mask.shape)
                    
                temp_mask[input_mask != id] = 0
                temp_mask[input_mask == id] = id

                temp_mask = torch.from_numpy(temp_mask)
                temp_mask = temp_mask.type(torch.LongTensor)

                if torch.count_nonzero(new_mask) == 0:
                    new_mask = temp_mask.detach().clone()

                else:
                    new_mask = torch.cat((new_mask, temp_mask))

            input_mask = new_mask.detach().clone()

            instance_masks = torchvision.tv_tensors.Mask(torch.concat([torchvision.tv_tensors.Mask(input_mask, dtype=torch.bool)]))

            if list(instance_masks.shape) == [1, 512, 1024]:
                print("---A---")
                return (torch.tensor(0), {"boxes": "INVALID"})

            # torch.set_printoptions(profile="full")
            # print(instance_masks.to(torch.uint8))
            # print(instance_masks.shape)
            bb = torchvision.ops.masks_to_boxes(instance_masks)

            delete_idxs = []

            for idx, box in enumerate(bb):
                
                if bb[idx][0] == bb[idx][2]:

                    delete_idxs.append(idx)

                elif bb[idx][1] == bb[idx][3]:

                    delete_idxs.append(idx)

            if len(delete_idxs) > 0:

                for idx in reversed(delete_idxs):
                    bb = torch.cat((bb[:idx], bb[idx + 1:]))
                    instance_masks = torch.cat((instance_masks[:idx], instance_masks[idx + 1:]))
                    instances = np.delete(instances, idx)

            if list(bb.shape) == [4,]:
                print("---E---")
                return (torch.tensor(0), {"boxes": "INVALID"})

            labels = [(label / 1000) - 10 for label in instances]
            # labels.insert(0, 0)
            
            # torch.set_printoptions(profile="full")
            # print(torch.tensor(instance_masks, dtype = torch.int64))

            target = {}
            target["boxes"] = torchvision.tv_tensors.BoundingBoxes(bb, format="XYXY", canvas_size=(512, 1024))
            target["masks"] = torch.tensor(instance_masks, dtype = torch.uint8)
            target["labels"] = torch.tensor(labels, dtype = torch.int64)

            # print(target["labels"])
            
            # print(type(target["boxes"]))
            
            # print(type(target))

            # print(target["boxes"])

            return torch.tensor(input_feature, dtype = torch.float32), target

        else:

            # input_mask[input_mask > 1000] = input_mask[input_mask > 1000] / 1000
            # input_mask[input_mask == -1] = 19
            # input_mask[input_mask == 255] = 20

            # input_mask = torch.from_numpy(input_mask)

            # input_mask = input_mask.type(torch.LongTensor)

            # # torch.set_printoptions(profile="full")
            # # print(input_mask)

            # input_mask = torch.nn.functional.one_hot(input_mask, 21).transpose(0, 3).squeeze(-1)

            # # print(input_mask)

            # # print(input_mask.shape)

            #---------------------------------------------------------------------------------#

            input_mask[input_mask < 1000] = 0
            input_mask[input_mask > 1000] = (input_mask[input_mask > 1000] / 1000) - 10

            input_mask = torch.from_numpy(input_mask)

            input_mask = input_mask.type(torch.LongTensor)

            # torch.set_printoptions(profile="full")
            # print(input_mask)

            input_mask = torch.nn.functional.one_hot(input_mask, 9).transpose(0, 3).squeeze(-1)

            return {"image": torch.tensor(input_feature, dtype = torch.float32), 
                    "mask": torch.tensor(input_mask, dtype = torch.float32)
                    }

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