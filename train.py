import torchvision.tv_tensors
from dataset import CityScapesFiles
from PIL import Image
import torch, torchvision
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels     import labels, name2label

feature_dir = r"./dataset/features/leftImg8bit"
label_dir = r"./dataset/labels/gtFine_trainvaltest/gtFine"

dataset = CityScapesFiles()

# train_imgs, test_imgs, val_imgs = dataset.load_features(feature_dir)
train_masks, test_masks, val_masks = dataset.load_masks(label_dir)
# train_anno, test_anno, val_anno = dataset.load_labels(label_dir)

# print(len(train_imgs) + len(test_imgs) + len(val_imgs))
print(len(train_masks) + len(test_masks) + len(val_masks))
# print(len(train_anno) + len(test_anno) + len(val_anno))

# print(train_anno)

# print(name2label)

# transforms = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
# masks = transforms(train_masks)

# print(type(train_masks[0]))

masks = torchvision.tv_tensors.Mask(torch.concat([torchvision.tv_tensors.Mask(torchvision.transforms.PILToTensor()(train_masks[0]), dtype=torch.bool)])) # train_masks[0])
print(masks)

bounding_box = torchvision.tv_tensors.BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format=torchvision.tv_tensors.BoundingBoxFormat("CXCYWH"), canvas_size=train_masks[0].size[::-1])

print(bounding_box)

# print(labels)

lbs = [11]
lbs_tens = torch.Tensor(lbs)

# torchvision.utils.draw_bounding_boxes(masks.to(dtype=torch.uint8), bounding_box.to(dtype=torch.uint8), lbs_tens)