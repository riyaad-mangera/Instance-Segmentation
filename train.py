import numpy as np
import torchvision.tv_tensors
from torch.utils.data import DataLoader
from dataset import CityScapesFiles, CityScapesDataset
import models
from PIL import Image
import torch, torchvision
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels import labels, name2label

import models.UNet_Model

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64

def train(model, train_loader, val_loader, loss_function, epochs = 5):

    return 0

feature_dir = r"./dataset/features/leftImg8bit"
label_dir = r"./dataset/labels/gtFine_trainvaltest/gtFine"

dataset = CityScapesFiles()

# Get list of directories for all required files
train_imgs, test_imgs, val_imgs = dataset.load_features(feature_dir)
train_masks, test_masks, val_masks = dataset.load_masks(label_dir)
train_polygons, test_polygons, val_polygons = dataset.load_labels(label_dir)

# print(len(train_imgs) + len(test_imgs) + len(val_imgs))
# print(len(train_masks) + len(test_masks) + len(val_masks))
# print(len(train_anno) + len(test_anno) + len(val_anno))

# print(labels)
trainId2label   = { label.trainId : label for label in reversed(labels) }
# print(trainId2label)

train_dataset = CityScapesDataset(train_imgs, train_masks, train_polygons, trainId2label, sample_frac = 256)
test_dataset = CityScapesDataset(test_imgs, test_masks, test_polygons, trainId2label, sample_frac = 256)
val_dataset = CityScapesDataset(val_imgs, val_masks, val_polygons, trainId2label, sample_frac = 256)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': True,
                'pin_memory': True
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': True,
                'pin_memory': True
                }

val_params = {'batch_size': VAL_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': True,
                'pin_memory': True
                }

train_dataloader = DataLoader(train_dataset, **train_params)
test_dataloader = DataLoader(test_dataset, **test_params)
val_dataloader = DataLoader(val_dataset, **val_params)

model = models.UNet_Model.UNetModel(in_channels = 3, num_classes = 21)

# print(train_anno)

# print(name2label)

# transforms = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
# masks = transforms(train_masks)

# print(type(train_masks[0]))

# masks = torchvision.tv_tensors.Mask(torch.concat([torchvision.tv_tensors.Mask(torchvision.transforms.PILToTensor()(train_masks[0]), dtype=torch.bool)])) # train_masks[0])
# print(masks)

# bounding_box = torchvision.tv_tensors.BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format=torchvision.tv_tensors.BoundingBoxFormat("CXCYWH"), canvas_size=train_masks[0].size[::-1])

# print(bounding_box)

# print(labels)

# lbs = [11]
# lbs_tens = torch.Tensor(lbs)

# torchvision.utils.draw_bounding_boxes(masks.to(dtype=torch.uint8), bounding_box.to(dtype=torch.uint8), lbs_tens)

# img = Image.open("./test_masks/aachen_000000_000019_gtFine_instanceTrainIds.png")

# transforms = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
# transforms = torchvision.transforms.PILToTensor()
# mask = transforms(img)

# torch.set_printoptions(threshold=np.inf)
# print(mask)



# bb = torchvision.ops.masks_to_boxes(mask.to(dtype=torch.uint8))

# print(bb)

# boxes = torchvision.utils.draw_bounding_boxes(mask.to(dtype=torch.uint8), bb, colors="red")

# img.close()

# img2 = torchvision.transforms.ToPILImage()(boxes) 
  
# # display output 
# img2.show() 