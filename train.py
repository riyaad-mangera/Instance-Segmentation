import numpy as np
import torchvision.tv_tensors
from torch.utils.data import DataLoader
from dataset import CityScapesFiles, CityScapesDataset
import models
from PIL import Image
import torch, torchvision
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels import labels, name2label
from models import UNet_Model, MaskRCNN_Model

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f'Device: {device}')

TRAIN_BATCH_SIZE = 5
TEST_BATCH_SIZE = 1
VAL_BATCH_SIZE = 5

def dice_coefficient(y_pred, y_true, epsilon = 1e-07):
    y_pred_copy = y_pred.clone()

    y_pred_copy[y_pred_copy < 0] = 0
    y_pred_copy[y_pred_copy > 0] = 1

    intersection = abs(torch.sum(y_pred_copy * y_true))
    union = abs(torch.sum(y_pred_copy) + torch.sum(y_true))
    dice_coef = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice_coef

def train(model, train_loader, val_loader, loss_function, optimiser, epochs = 5):

    model.to(device)
    loss_function.to(device)

    average_losses = []
    average_dice_coef = []

    model.train()

    for epoch in range(epochs):

        print(f'Epoch: {epoch}')

        epoch_losses = []
        epoch_dice_coeffs = []

        for idx, batch in enumerate(train_loader):

            print(f'Epoch: {epoch}, Batch {idx} of {int(len(train_loader.dataset)/TRAIN_BATCH_SIZE)}')

            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            # print(image.shape)
            # print(mask.shape)

            y_pred = model(image)
            y_pred = y_pred.to(device)

            optimiser.zero_grad()
            # model.zero_grad()

            dice_coeff = dice_coefficient(y_pred, mask)

            # print(dice_coeff)

            # pred_labels = torch.argmax(y_pred, dim=1)
            # print(pred_labels)

            # print(y_pred.shape)
            # print(mask.shape)

            loss = loss_function(y_pred, mask)
            # optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # print(loss)

            epoch_losses.append(loss.item())
            epoch_dice_coeffs.append(dice_coeff.item())

            # print(epoch_losses)
            # print(epoch_dice_coeffs)

        average_losses.append(np.average(epoch_losses))
        average_dice_coef.append(np.average(epoch_dice_coeffs))

        print(average_losses)
        print(average_dice_coef)

    return average_losses, average_dice_coef

def test(model, test_loader, loss_function):

    model.eval()

    average_losses = []
    average_dice_coef = []

    for idx, batch in enumerate(test_loader):

        print(f'Testing Batch {idx} of {int(len(test_loader.dataset)/TEST_BATCH_SIZE)}')

        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        y_pred = model(image)
        y_pred = y_pred.to(device)

        dice_coeff = dice_coefficient(y_pred, mask)
        loss = loss_function(y_pred, mask)

        average_losses.append(np.average(loss.item()))
        average_dice_coef.append(np.average(dice_coeff.item()))

    return average_losses, average_dice_coef, y_pred

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
train_labels = []
trainId2label   = { label.trainId : label for label in reversed(labels) }
for label in labels:
    train_labels.append(label.trainId)

train_labels = list(set(train_labels))
# print(train_labels)

train_dataset = CityScapesDataset(train_imgs, train_masks, train_polygons, train_labels, sample_frac = 30)
test_dataset = CityScapesDataset(test_imgs, test_masks, test_polygons, train_labels, sample_frac = 1)
val_dataset = CityScapesDataset(val_imgs, val_masks, val_polygons, train_labels, sample_frac = 100)

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

model = UNet_Model.UNetModel(in_channels = 3, num_classes = 21) # 21)
# model_rcnn = MaskRCNN_Model.MaskRCNN_Model(model = None, checkpoint = None, num_classes = 21).get_model()

# print(model_rcnn)

# loss_function = torch.nn.BCEWithLogitsLoss()
loss_function = torch.nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr = 0.1, weight_decay = 0.0)
# optimiser = torch.optim.AdamW(model.parameters(), lr=0.1)

average_losses, average_dice_coef = train(model, train_dataloader, val_dataloader, loss_function, optimiser, epochs = 10)

print(average_losses)
print(average_dice_coef)

test_losses, test_dice_coefs, y_pred = test(model, test_dataloader, loss_function)

print(test_losses)
print(test_dice_coefs)

# unorm = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# pred_mask = unorm(y_pred)

# torch.set_printoptions(profile="full")
# print(y_pred)

# transforms = torchvision.transforms.ToPILImage()
# img = transforms(y_pred)

# img.show()

# print(torch.squeeze(y_pred, dim=0))
# print(torch.squeeze(y_pred, dim=0).shape)

# palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
# colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
# colors = (colors % 255).numpy().astype("uint8")

# # plot the semantic segmentation predictions of 21 classes in each color
# r = Image.fromarray(torch.squeeze(y_pred, dim=0).byte().cpu().numpy()).resize((512, 512))
# r.putpalette(colors)

# import matplotlib.pyplot as plt
# plt.imshow(r)

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

# img = Image.open("./test_masks/02_00_050.png")

# transforms = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
# transforms = torchvision.transforms.PILToTensor()
# mask = transforms(img)

# # torch.set_printoptions(threshold=np.inf)
# print(mask.shape)


# img = np.array(Image.open("./test_masks/02_00_050.png"))

# # transforms = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
# transforms = torchvision.transforms.ToTensor()
# mask = transforms(img)

# mask = torch.max(mask, dim=2)[0]

# # torch.set_printoptions(threshold=np.inf)
# print(mask.type(torch.long).shape)


# bb = torchvision.ops.masks_to_boxes(mask.to(dtype=torch.uint8))

# print(bb)

# boxes = torchvision.utils.draw_bounding_boxes(mask.to(dtype=torch.uint8), bb, colors="red")

# img.close()

# img2 = torchvision.transforms.ToPILImage()(boxes) 
  
# # display output 
# img2.show() 