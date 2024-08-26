import numpy as np
import torchvision.tv_tensors
from torch.utils.data import DataLoader
from dataset import CityScapesFiles, CityScapesDataset
from logger import Logger
import models
from PIL import Image
import torch, torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mat_img
import wandb
import pickle
from matplotlib.colors import ListedColormap
from cityscapesscripts.helpers.labels import labels, name2label
from models import UNet_Model, MaskRCNN_Model
from PIL import Image
from PIL import ImageDraw

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Device: {device}')

TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1

def dice_coefficient(y_pred, y_true, epsilon = 1e-07):
    y_pred_copy = y_pred.clone()

    y_pred_copy[y_pred_copy < 0] = 0
    y_pred_copy[y_pred_copy > 0] = 1

    intersection = abs(torch.sum(y_pred_copy * y_true))
    union = abs(torch.sum(y_pred_copy) + torch.sum(y_true))
    dice_coef = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice_coef

def train(model, train_loader, val_loader, loss_function, optimiser, logger, epochs = 5):

    model.to(device)
    loss_function.to(device)

    average_val_losses = []
    average_val_dice_coef = []

    model.train()

    for epoch in range(epochs):

        print(f'Epoch: {epoch}')

        train_loss = []
        batch_losses = []
        batch_dice_coeffs = []

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

            train_loss.append(loss.item())

            # print(loss)

            with torch.no_grad():
                model.eval()
                
                for val_idx, val_batch in enumerate(val_loader):

                    print(f'Validating Batch {val_idx} of {int(len(val_loader.dataset)/VAL_BATCH_SIZE)}')

                    val_image = val_batch["image"].to(device)
                    val_mask = val_batch["mask"].to(device)

                    val_y_pred = model(val_image)
                    val_y_pred = val_y_pred.to(device)

                    val_dice_coeff = dice_coefficient(val_y_pred, val_mask)
                    val_loss = loss_function(val_y_pred, val_mask)

                    batch_losses.append(val_loss.item())
                    batch_dice_coeffs.append(val_dice_coeff.item())

                    if (idx == TRAIN_BATCH_SIZE - 1) and (val_idx == VAL_BATCH_SIZE - 1):

                        y_pred_labels_again = torch.argmax(val_y_pred, dim=1)
                        y_pred_img_arr = y_pred_labels_again.detach().cpu().numpy()

                        # y_pred_img_arr = np.transpose(y_pred_img_arr,(1, 2, 0))

                        y_true_labels_again = torch.argmax(val_mask, dim=1)
                        y_true_img_arr = y_true_labels_again.detach().cpu().numpy()

                        # y_true_img_arr = np.transpose(y_true_img_arr,(1, 2, 0))

                        class_labels = {0: "road", 1: "sidewalk", 2: "building", 3:"wall", 4:"fence", 
                                        5:"pole", 6:"traffic_light", 7:"traffic_sign", 8:"vegetation", 
                                        9:"terrain", 10:"sky", 11:"person", 12:"rider", 
                                        13:"car", 14:"truck", 15:"bus", 16:"train", 
                                        17:"motorcycle", 18:"bicycle", 19:"license_plate", 20:"unlabeled"}
                        
                        invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                    torchvision.transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                        std = [ 1., 1., 1. ]),
                                ])

                        orig_img = invTrans(val_image)

                        # img_trans = torchvision.transforms.ToPILImage()

                        orig_img = orig_img.detach().cpu().numpy()

                        original_image = np.transpose(orig_img[0],(1, 2, 0))

                        if logger != '':
                            logger.log({"validation_predictions" : wandb.Image(original_image, masks={"predictions" : {"mask_data" : y_pred_img_arr[0], "class_labels" : class_labels}, "ground_truth" : {"mask_data" : y_true_img_arr[0], "class_labels" : class_labels}})})

            model.train()

            # print(epoch_losses)
            # print(epoch_dice_coeffs)

        average_val_losses.append(np.average(batch_losses))
        average_val_dice_coef.append(np.average(batch_dice_coeffs))

        print(average_val_losses)
        print(average_val_dice_coef)

        if logger != '':
            logger.log({'train_loss': np.sum(train_loss) / len(train_loss),
                        'validation_loss': np.sum(batch_losses) / len(batch_losses), 
                        'validation_dice_coefficient': np.sum(batch_dice_coeffs) / len(batch_dice_coeffs)})

        if epoch % 2 == 0:    
            with open(f'./checkpoints/unet_test_2_ep_{epoch}.pkl', 'wb') as file:
                pickle.dump(model, file)

    return average_val_losses, average_val_dice_coef

def test(model, test_loader, loss_function, logger):

    model.eval()

    dice_coefs = []
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

        dice_coefs.append(dice_coeff.item())

        average_losses.append(np.average(loss.item()))
        average_dice_coef.append(np.average(dice_coeff.item()))

        y_pred_labels_again = torch.argmax(y_pred, dim=1)
        y_pred_img_arr = y_pred_labels_again.detach().cpu().numpy()

        # y_pred_img_arr = np.transpose(y_pred_img_arr,(1, 2, 0))

        y_true_labels_again = torch.argmax(mask, dim=1)
        y_true_img_arr = y_true_labels_again.detach().cpu().numpy()

        # y_true_img_arr = np.transpose(y_true_img_arr,(1, 2, 0))

        class_labels = {0: "road", 1: "sidewalk", 2: "building", 3:"wall", 4:"fence", 
                        5:"pole", 6:"traffic_light", 7:"traffic_sign", 8:"vegetation", 
                        9:"terrain", 10:"sky", 11:"person", 12:"rider", 
                        13:"car", 14:"truck", 15:"bus", 16:"train", 
                        17:"motorcycle", 18:"bicycle", 19:"license_plate", 20:"unlabeled"}
        
        invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                        std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                    torchvision.transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                        std = [ 1., 1., 1. ]),
                ])

        orig_img = invTrans(image)

        # img_trans = torchvision.transforms.ToPILImage()

        orig_img = orig_img.detach().cpu().numpy()

        original_image = np.transpose(orig_img[0],(1, 2, 0))

        if logger != '':
            logger.log({"test_predictions" : wandb.Image(original_image, masks={"predictions": {"mask_data" : y_pred_img_arr[0], "class_labels" : class_labels}})})

    if logger != '':
        logger.log({'test_dice_coefficients': np.sum(dice_coefs) / len(dice_coefs)})

    return average_losses, average_dice_coef, y_pred

def train_with_instances(model, train_loader, val_loader, loss_function, optimiser, logger, epochs = 5):
    model.to(device)
    loss_function.to(device)

    average_val_losses = []
    average_val_dice_coef = []

    model.train()

    for epoch in range(epochs):

        print(f'Epoch: {epoch}')

        train_loss = []
        batch_losses = []
        batch_dice_coeffs = []

        for idx, (image, targets) in enumerate(train_loader):

            print(f'Epoch: {epoch}, Batch {idx} of {int(len(train_loader.dataset)/TRAIN_BATCH_SIZE)}')

            image = image.to(device)
            # image = [torch.squeeze(image)]

            target = [{k: v for k, v in targets.items()}]            

            for index, t in enumerate(target):

                target[index]["boxes"] = torch.squeeze(target[index]["boxes"]).to(device)
                target[index]["masks"] = torch.squeeze(target[index]["masks"]).to(device)
                target[index]["labels"] = torch.squeeze(target[index]["labels"]).to(device)

            y_pred = model(image, target)

            # print(y_pred)

            # y_pred = y_pred.to(device)

            optimiser.zero_grad()
            # model.zero_grad()

            # dice_coeff = dice_coefficient(y_pred, mask)

            # pred_labels = torch.argmax(y_pred, dim=1)

            # loss = loss_function(y_pred, mask)
            # optimiser.zero_grad()

            loss = sum([loss for loss in y_pred.values()])
            loss.backward()
            optimiser.step()

            train_loss.append(loss.item())
            # train_loss.append(y_pred)

            # print(train_loss)

            with torch.no_grad():
                model.eval()
                
                for val_idx, (val_image, val_targets) in enumerate(val_loader):

                    print(f'Validating Batch {val_idx} of {int(len(val_loader.dataset)/VAL_BATCH_SIZE)}')

                    val_image = val_image.to(device)
                    # val_image = [torch.squeeze(val_image)]

                    val_target = [{k: v for k, v in val_targets.items()}]            

                    for index, t in enumerate(val_target):

                        val_target[index]["boxes"] = torch.squeeze(val_target[index]["boxes"]).to(device)
                        val_target[index]["masks"] = torch.squeeze(val_target[index]["masks"]).to(device)
                        val_target[index]["labels"] = torch.squeeze(val_target[index]["labels"]).to(device)

                    val_y_pred = model(val_image)

                    # print(val_y_pred)
                    # val_loss = sum([loss for loss in val_y_pred.values()])

                    # print(val_loss)

                    # val_y_pred = val_y_pred.to(device)

                    # val_dice_coeff = dice_coefficient(val_y_pred, val_mask)
                    # val_loss = loss_function(val_y_pred, val_mask)

                    # batch_losses.append(val_loss.item())
                    # batch_dice_coeffs.append(val_dice_coeff.item())

                    # if (idx == TRAIN_BATCH_SIZE - 1) and (val_idx == VAL_BATCH_SIZE - 1):

                    #     y_pred_labels_again = torch.argmax(val_y_pred, dim=1)
                    #     y_pred_img_arr = y_pred_labels_again.detach().cpu().numpy()

                    #     # y_pred_img_arr = np.transpose(y_pred_img_arr,(1, 2, 0))

                    #     y_true_labels_again = torch.argmax(val_mask, dim=1)
                    #     y_true_img_arr = y_true_labels_again.detach().cpu().numpy()

                    #     # y_true_img_arr = np.transpose(y_true_img_arr,(1, 2, 0))

                    #     class_labels = {0: "road", 1: "sidewalk", 2: "building", 3:"wall", 4:"fence", 
                    #                     5:"pole", 6:"traffic_light", 7:"traffic_sign", 8:"vegetation", 
                    #                     9:"terrain", 10:"sky", 11:"person", 12:"rider", 
                    #                     13:"car", 14:"truck", 15:"bus", 16:"train", 
                    #                     17:"motorcycle", 18:"bicycle", 19:"license_plate", 20:"unlabeled"}
                        
                    #     invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                    #                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                    #                 torchvision.transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                    #                                     std = [ 1., 1., 1. ]),
                    #             ])

                    #     orig_img = invTrans(val_image)

                    #     # img_trans = torchvision.transforms.ToPILImage()

                    #     orig_img = orig_img.detach().cpu().numpy()

                    #     original_image = np.transpose(orig_img[0],(1, 2, 0))

                    #     if logger != '':
                    #         logger.log({"validation_predictions" : wandb.Image(original_image, masks={"predictions" : {"mask_data" : y_pred_img_arr[0], "class_labels" : class_labels}, "ground_truth" : {"mask_data" : y_true_img_arr[0], "class_labels" : class_labels}})})

            model.train()

            # print(epoch_losses)
            # print(epoch_dice_coeffs)

        average_val_losses.append(val_y_pred[0])
        # average_val_dice_coef.append(np.average(batch_dice_coeffs))

        # print(average_val_losses)
        # print(average_val_dice_coef)

        if logger != '':
            logger.log({'train_loss': np.sum(train_loss) / len(train_loss),
                        'validation_loss': np.sum(batch_losses) / len(batch_losses), 
                        'validation_dice_coefficient': np.sum(batch_dice_coeffs) / len(batch_dice_coeffs)})

        if epoch % 2 == 0:    
            with open(f'./checkpoints/mask_rcnn_test_ep_{epoch}.pkl', 'wb') as file:
                pickle.dump(model, file)

    return average_val_losses, train_loss

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

instances_only = True

train_dataset = CityScapesDataset(train_imgs, train_masks, train_polygons, train_labels, sample_frac = 20, instances_only = instances_only) # 500)
test_dataset = CityScapesDataset(test_imgs, test_masks, test_polygons, train_labels, sample_frac = 20, instances_only = instances_only)
val_dataset = CityScapesDataset(val_imgs, val_masks, val_polygons, train_labels, sample_frac = 5, instances_only = instances_only) # 5)

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

logger = ''
# wandb_logger = Logger(f"unet_test", project='instance-segmentation-project')
# logger = wandb_logger.get_logger()

# model = UNet_Model.UNetModel(in_channels = 3, num_classes = 21) # 21)
model_rcnn = MaskRCNN_Model.MaskRCNN_Model(model = None, checkpoint = None, num_classes = 8).get_model()

# print(model_rcnn)

loss_function = torch.nn.BCEWithLogitsLoss()
# loss_function = torch.nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model_rcnn.parameters(), lr = 0.1, weight_decay = 0.05)
# optimiser = torch.optim.AdamW(model.parameters(), lr=0.1)

# average_losses, average_dice_coef = train(model, train_dataloader, val_dataloader, loss_function, optimiser, logger, epochs = 1)
average_losses, train_loss = train_with_instances(model_rcnn, train_dataloader, val_dataloader, loss_function, optimiser, logger, epochs = 25)

print(average_losses)
# print(average_dice_coef)
print("--------------------AAAAAAAAAA----------------------")
print(train_loss)

# with open(f'./checkpoints/unet_test_ep_30.pkl', 'rb') as file:
#     model = pickle.load(file)

# test_losses, test_dice_coefs, y_pred = test(model, test_dataloader, loss_function, logger)

# print(test_losses)
# print(test_dice_coefs)

# labels_again = torch.argmax(y_pred, dim=1)
# print(labels_again)

# img_arr = labels_again.detach().cpu().numpy()
# print(img_arr[0])


#-------------------------------------------------------------------

# cmap = ListedColormap([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), 
#                        (190,153,153), (153,153,153), (250,170, 30), (220,220,  0), 
#                        (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), 
#                        (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100), 
#                        (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142), 
#                        (  0,  0,  0) ])

# cmap.set_bad("black")
# plt.imshow(img_arr[0], cmap=cmap)
# plt.show()

# img = torchvision.transforms.ToPILImage(mode="I")(labels_again.to(torch.int))
# img = img.resize((2048, 1024))

# img = img.point(lambda i: (i * 1000) * (1 / 255)).convert("RGB").save("test_masks/test_out.png")

# def visualise_mask(mask):

#     labels_again = torch.argmax(mask, dim=1)

#     img_arr = labels_again.detach().cpu().numpy()

#     cmap = ListedColormap([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), 
#                            (190,153,153), (153,153,153), (250,170, 30), (220,220,  0), 
#                            (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), 
#                            (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100), 
#                            (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142), 
#                            (  0,  0,  0) ])
    
#     cmap.set_bad("black")
#     new_mask = plt.imshow(img_arr[0], cmap=cmap)

#     return new_mask

# for idx, batch in enumerate(val_dataloader):
#     mask = batch["mask"].to(device)

#     test = visualise_mask(mask)
#     plt.show()

#-------------------------------------------------------------------------------------------------------#

# img.show()

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