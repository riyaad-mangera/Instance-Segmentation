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
from DiceLoss import DiceLoss
from PIL import Image
from PIL import ImageDraw
import os

# os.environ['https_proxy'] = 'http://hpc-proxy00.city.ac.uk:3128'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Device: {device}')

TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1

def dice_coefficient(y_pred, y_true, epsilon = 1e-6):
    y_pred_copy = y_pred.clone()

    y_pred_copy[y_pred_copy < 0] = 0
    y_pred_copy[y_pred_copy > 0] = 1

    intersection = abs(torch.sum(y_pred_copy * y_true))
    union = abs(torch.sum(y_pred_copy) + torch.sum(y_true))
    dice_coef = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice_coef

# Altered Sorensen–Dice coefficient with epsilon for smoothing
def dice_coefficient_generalised(y_pred, y_true, epsilon=1e-6):

    # y_pred = y_pred.detach().cpu()
    # y_true = y_true.detach().cpu()

    # y_true_flatten = np.asarray(y_true).astype(np.bool)
    # y_pred_flatten = np.asarray(y_pred).astype(np.bool)

    # if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
    #     return 1.0

    # return (2. * np.sum(y_true_flatten * y_pred_flatten)) / (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + epsilon)

    prediction = y_pred.view(-1)
    target = y_true.view(-1)

    intersection = torch.sum(prediction * target)
    union = torch.sum(prediction) + torch.sum(target)

    dice_coef = 1 - ((2. * intersection + epsilon) / (union + epsilon))

    return dice_coef

def train(model, train_loader, val_loader, loss_function, optimiser, logger, epochs = 5, start_epoch = 0):

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

            # dice_coeff = dice_coefficient(y_pred, mask)

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

                    print(f'\tValidating Batch {val_idx} of {int(len(val_loader.dataset)/VAL_BATCH_SIZE)}')

                    val_image = val_batch["image"].to(device)
                    val_mask = val_batch["mask"].to(device)

                    val_y_pred = model(val_image)
                    val_y_pred = val_y_pred.to(device)

                    val_dice_coeff = dice_coefficient(val_y_pred, val_mask)
                    val_loss = loss_function(val_y_pred, val_mask)

                    batch_losses.append(val_loss.item())
                    batch_dice_coeffs.append(val_dice_coeff.item())

                    if (idx == TRAIN_BATCH_SIZE - 1) and (val_idx == VAL_BATCH_SIZE - 1):

                        # print(val_y_pred.shape)
                        # print(val_mask.shape)

                        y_pred_labels_again = torch.argmax(val_y_pred, dim=1)
                        y_pred_img_arr = y_pred_labels_again.detach().cpu().numpy()

                        # y_pred_img_arr = np.transpose(y_pred_img_arr,(1, 2, 0))

                        y_true_labels_again = torch.argmax(val_mask, dim=1)
                        y_true_img_arr = y_true_labels_again.detach().cpu().numpy()

                        # print(y_pred_img_arr.shape)
                        # print(y_true_img_arr.shape)

                        # y_true_img_arr = np.transpose(y_true_img_arr,(1, 2, 0))

                        # class_labels = {0: "road", 1: "sidewalk", 2: "building", 3:"wall", 4:"fence", 
                        #                 5:"pole", 6:"traffic_light", 7:"traffic_sign", 8:"vegetation", 
                        #                 9:"terrain", 10:"sky", 11:"person", 12:"rider", 
                        #                 13:"car", 14:"truck", 15:"bus", 16:"train", 
                        #                 17:"motorcycle", 18:"bicycle", 19:"license_plate", 20:"unlabeled"}

                        # class_labels = {1: "road", 2: "sidewalk", 3: "building", 4:"wall", 5:"fence", 
                        #                 6:"pole", 7:"traffic_light", 8:"traffic_sign", 9:"vegetation", 
                        #                 10:"terrain", 11:"sky", 12:"person", 13:"rider", 
                        #                 14:"car", 15:"truck", 16:"bus", 17:"train", 
                        #                 18:"motorcycle", 19:"bicycle", 20:"license_plate"
                        #                 }

                        class_labels = {1: "person", 2: "rider", 3: "car", 4: "truck",
                                        5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"
                                        }
                        
                        # class_labels = {0: "person", 1: "rider", 2: "car", 3: "truck",
                        #                 4: "bus", 5: "train", 6: "motorcycle", 7: "bicycle"
                        #                 }

                        invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                    torchvision.transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                        std = [ 1., 1., 1. ]),
                                ])

                        orig_img = invTrans(val_image)

                        # img_trans = torchvision.transforms.ToPILImage()

                        orig_img = orig_img.detach().cpu().numpy()

                        original_image = np.transpose(orig_img[0],(1, 2, 0))

                        # if logger != '':
                        #     logger.log({"validation_predictions" : wandb.Image(original_image, masks={"predictions" : {"mask_data" : y_pred_img_arr[0], "class_labels" : class_labels}, "ground_truth" : {"mask_data" : y_true_img_arr[0], "class_labels" : class_labels}})})

            model.train()

            # print(epoch_losses)
            # print(epoch_dice_coeffs)

        # average_val_losses.append(np.average(batch_losses))
        # average_val_dice_coef.append(np.average(batch_dice_coeffs))

        # print(average_val_losses)
        # print(average_val_dice_coef)

        if logger != '':
            logger.log({"validation_predictions" : wandb.Image(original_image, masks={"predictions" : {"mask_data" : y_pred_img_arr[0], "class_labels" : class_labels}, "ground_truth" : {"mask_data" : y_true_img_arr[0], "class_labels" : class_labels}}),
                        'train_loss': np.sum(train_loss) / len(train_loss),
                        'validation_loss': np.sum(batch_losses) / len(batch_losses), 
                        'validation_dice_coefficient': np.sum(batch_dice_coeffs) / len(batch_dice_coeffs)})

        if epoch % 10 == 0 or epoch == epochs - 1:    
            with open(f'./checkpoints/{model.name}_ep_{epoch + start_epoch}.pkl', 'wb') as file:
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

        # class_labels = {0: "road", 1: "sidewalk", 2: "building", 3:"wall", 4:"fence", 
        #                 5:"pole", 6:"traffic_light", 7:"traffic_sign", 8:"vegetation", 
        #                 9:"terrain", 10:"sky", 11:"person", 12:"rider", 
        #                 13:"car", 14:"truck", 15:"bus", 16:"train", 
        #                 17:"motorcycle", 18:"bicycle", 19:"license_plate", 20:"unlabeled"}

        # class_labels = {1: "road", 2: "sidewalk", 3: "building", 4:"wall", 5:"fence", 
        #                 6:"pole", 7:"traffic_light", 8:"traffic_sign", 9:"vegetation", 
        #                 10:"terrain", 11:"sky", 12:"person", 13:"rider", 
        #                 14:"car", 15:"truck", 16:"bus", 17:"train", 
        #                 18:"motorcycle", 19:"bicycle", 20:"license_plate"
        #                 }

        class_labels = {1: "person", 2: "rider", 3: "car", 4: "truck", 
                        5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"
                        }
        
        # class_labels = {0: "person", 1: "rider", 2: "car", 3: "truck",
        #                 4: "bus", 5: "train", 6: "motorcycle", 7: "bicycle"
        #                 }

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
            logger.log({"test_predictions" : wandb.Image(original_image, masks={"predictions": {"mask_data" : y_pred_img_arr[0], "class_labels" : class_labels}, "ground_truth" : {"mask_data" : y_true_img_arr[0], "class_labels" : class_labels}})})

    if logger != '':
        logger.log({'test_dice_coefficients': np.sum(dice_coefs) / len(dice_coefs)})

    return average_losses, average_dice_coef, y_pred

def bounding_boxes(boxes, labels, class_id_to_label, pred = True):
    
    all_boxes = []

    # print(labels)

    # plot each bounding box for this image
    for idx, box in enumerate(boxes):
        # get coordinates and labels

        if pred:
            box_data = {"position" : {
                "minX" : box[0],
                "maxX" : box[2],
                "minY" : box[1],
                "maxY" : box[3]
                },
                "class_id" : labels[idx],
                "domain": "pixel",
                "box_caption": class_id_to_label[labels[idx]],
                "scores": {"acc": 0.0, "loss": 0.0}
                }

        else:

            box_data = {"position" : {
                "minX" : box[0].item(),
                "maxX" : box[2].item(),
                "minY" : box[1].item(),
                "maxY" : box[3].item()
                },
                "class_id" : labels[idx],
                "domain": "pixel",
                "box_caption": class_id_to_label[labels[idx]],
                "scores": {"acc": 0.0, "loss": 0.0}
                }
        
        # print(labels[idx])
        
        all_boxes.append(box_data)

    # print(all_boxes[0])

    return all_boxes

def iou_score(y_pred, y_true):

    # print(type(y_pred))
    # print(y_true)

    intersection = (y_pred * y_true).sum()

    if intersection == 0:
        return 0.0
    
    union = torch.logical_or(y_pred, y_true).to(torch.int).sum()

    score = intersection / union

    del y_pred
    del y_true
    del intersection
    del union

    return score

def train_with_instances(model, train_loader, val_loader, loss_function, optimiser, logger, epochs = 5, start_epoch = 0):
    model.to(device)
    loss_function.to(device)

    average_val_losses = []

    model.train()

    for epoch in range(epochs):

        print(f'Epoch: {epoch}')

        train_loss = []
        batch_losses = []
        batch_iou = []

        for idx, (image, targets) in enumerate(train_loader):

            print(f'Epoch: {epoch}, Batch {idx} of {int(len(train_loader.dataset)/TRAIN_BATCH_SIZE)}')

            # print(image)

            # print(list(torch.tensor(0).shape))
            if list(image.shape) == [1]:
                print("Invalid inputs, skipping...")
                continue

            else:

                # print(list(image.shape))

                image = image.to(device)
                # image = [torch.squeeze(image)]

                target = [{k: v for k, v in targets.items()}]            

                for index, t in enumerate(target):

                    target[index]["boxes"] = torch.squeeze(target[index]["boxes"]).to(device)
                    target[index]["masks"] = torch.squeeze(target[index]["masks"]).to(device)
                    target[index]["labels"] = torch.squeeze(target[index]["labels"]).to(device)

                # # Skip images with no boxes
                # if list(target[0]["boxes"].shape) == [4,]:
                #     print("No boxes")
                #     continue

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

                        print(f'\tValidating Batch {val_idx} of {int(len(val_loader.dataset)/VAL_BATCH_SIZE)}')

                        if list(val_image.shape) == [1]:
                            print("Invalid inputs, skipping...")
                            continue

                        else:

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

                            batch_losses.append(val_y_pred)

                            # iou = iou_score(val_y_pred[0]["masks"], val_target[0]["masks"])
                            iou = -1

                            batch_iou.append(iou)
                            
                            if (idx == TRAIN_BATCH_SIZE - 1) and (val_idx == VAL_BATCH_SIZE - 1):

                                invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                                                            std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                            torchvision.transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                                            std = [ 1., 1., 1. ]),
                                        ])

                                orig_img = invTrans(val_image)

                                # img_trans = torchvision.transforms.ToPILImage()

                                orig_img = orig_img.detach().cpu().numpy()

                                original_image = np.transpose(orig_img[0],(1, 2, 0))

                                display_ids = {"person" : 1, "rider" : 2, "car" : 3, "truck" : 4,
                                            "bus" : 5, "train" : 6, "motorcycle": 7, "bicycle" : 8
                                            }


                                class_id_to_label = {1: "person", 2: "rider", 3: "car", 4: "truck",
                                                    5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"
                                                    }
                                
                                # print(val_y_pred[0]["boxes"])

                                # print(val_y_pred[0]["masks"])

                                boxes = val_y_pred[0]["boxes"].tolist()
                                labels = val_y_pred[0]["labels"].tolist()

                                true_labels = val_target[0]["labels"].tolist()

                                # print(val_y_pred[0]["scores"])
                                prob_threshold = 0.0

                                if len(boxes) > 0:

                                    thres_prob_masks = val_y_pred[0]["masks"] > prob_threshold

                                    # print(val_y_pred[0]["masks"])
                                    # print(thres_prob_masks)

                                    # print(val_y_pred[0]["masks"])

                                    # print(val_y_pred[0]["masks"].shape)

                                    y_pred_labels_again = torch.argmax(torch.squeeze(thres_prob_masks.to(torch.int64), 1), dim=0)
                                    y_true_labels_again = torch.argmax(val_target[0]["masks"], dim=0)

                                    # print(type(y_pred_labels_again))

                                    # y_pred_img_arr = y_pred_labels_again.detach().cpu().numpy()
                                    masks = y_pred_labels_again.detach().cpu().numpy()
                                    true_masks = y_true_labels_again.detach().cpu().numpy()

                                    # print(len(true_labels))

                                    for true_idx in range(1, len(true_labels)):
                                        # print(idx)
                                        true_masks[true_masks == true_idx] = true_labels[true_idx]

                                    for pred_idx in range(1, len(labels)):
                                        masks[masks == pred_idx] = labels[pred_idx]

                                    # print(true_masks.shape)
                                    # print(masks.shape)

                                    true_labels = val_target[0]["labels"].tolist()
                                    # print(labels)

                                    bb = bounding_boxes(boxes, labels, class_id_to_label, pred=True)
                                    true_bb = bounding_boxes(val_target[0]["boxes"], true_labels, class_id_to_label, pred=False)

                                    logger.log({"validation_predictions" : wandb.Image(original_image, 
                                                                                    boxes = {"predictions": {"box_data": bb, "class_labels" : class_id_to_label}, "ground_truth" : {"box_data": true_bb, "class_labels" : class_id_to_label}},
                                                                                    masks={"predictions": {"mask_data" : masks, "class_labels" : class_id_to_label}, "ground_truth": {"mask_data" : true_masks, "class_labels" : class_id_to_label}}
                                                                                    )})
                                    
                            del val_image
                            del val_targets
                            torch.cuda.empty_cache()
                    
                    # print(torch.cuda.memory_summary())
                    torch.cuda.empty_cache()

                model.train()

        if logger != '':
            logger.log({'train_loss': np.sum(train_loss) / len(train_loss),
                        # 'validation_loss': np.sum(batch_losses) / len(batch_losses),
                        'validation_iou': np.sum(batch_iou) / len(batch_iou) #np.sum(batch_iou) / len(batch_iou)
                        })

        if epoch != 0 and epoch % 1 == 0:    
            with open(f'./checkpoints/{model.name}_test_ep_{epoch + start_epoch}.pkl', 'wb') as file:
                pickle.dump(model, file)

    return "average_val_losses", train_loss

def test_with_instances(model, test_loader, loss_function, logger):
    model.eval()

    batch_losses = []
    batch_iou = []
                    
    for idx, (image, targets) in enumerate(test_loader):

        print(f'Testing Batch {idx} of {int(len(test_loader.dataset)/TEST_BATCH_SIZE)}')

        if list(image.shape) == [1]:
            print("Invalid inputs, skipping...")
            continue

        else:

            image = image.to(device)

            target = [{k: v for k, v in targets.items()}]            

            for index, t in enumerate(target):

                target[index]["boxes"] = torch.squeeze(target[index]["boxes"]).to(device)
                target[index]["masks"] = torch.squeeze(target[index]["masks"]).to(device)
                target[index]["labels"] = torch.squeeze(target[index]["labels"]).to(device)

            y_pred = model(image)

            batch_losses.append(y_pred)

            # iou = iou_score(y_pred[0]["masks"], target[0]["masks"])
            iou = -1

            batch_iou.append(iou)

            invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                                                            std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                            torchvision.transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                                            std = [ 1., 1., 1. ]),
                                        ])

            orig_img = invTrans(image)

            orig_img = orig_img.detach().cpu().numpy()

            original_image = np.transpose(orig_img[0],(1, 2, 0))

            class_id_to_label = {1: "person", 2: "rider", 3: "car", 4: "truck",
                                5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"
                                }
            
            # print(val_y_pred[0]["boxes"])

            # print(val_y_pred[0]["masks"])

            boxes = y_pred[0]["boxes"].tolist()
            labels = y_pred[0]["labels"].tolist()

            true_labels = target[0]["labels"].tolist()

            prob_threshold = 0

            if len(boxes) > 0:

                thres_prob_masks = y_pred[0]["masks"] > prob_threshold

                y_pred_labels_again = torch.argmax(torch.squeeze(thres_prob_masks.to(torch.int64), 1), dim=0)
                y_true_labels_again = torch.argmax(target[0]["masks"], dim=0)

                masks = y_pred_labels_again.detach().cpu().numpy()
                true_masks = y_true_labels_again.detach().cpu().numpy()

                for true_idx in range(1, len(true_labels)):
                    # print(idx)
                    true_masks[true_masks == true_idx] = true_labels[true_idx]

                for pred_idx in range(1, len(labels)):
                    masks[masks == pred_idx] = labels[pred_idx]

                true_labels = target[0]["labels"].tolist()

                bb = bounding_boxes(boxes, labels, class_id_to_label, pred=True)
                true_bb = bounding_boxes(target[0]["boxes"], true_labels, class_id_to_label, pred=False)

                logger.log({"validation_predictions" : wandb.Image(original_image, 
                                                                boxes = {"predictions": {"box_data": bb, "class_labels" : class_id_to_label}, "ground_truth" : {"box_data": true_bb, "class_labels" : class_id_to_label}},
                                                                masks={"predictions": {"mask_data" : masks, "class_labels" : class_id_to_label}, "ground_truth": {"mask_data" : true_masks, "class_labels" : class_id_to_label}}
                                                                )})
                
                del y_pred_labels_again
                del y_true_labels_again
                del masks
                del true_masks
                del true_labels
                del bb
                del true_bb
                torch.cuda.empty_cache()
                
            del image
            del orig_img
            del original_image
            del targets
            del target
            del y_pred
            torch.cuda.empty_cache()

def load_checkpoint(file_name):
    with open(f'./checkpoints/{file_name}.pkl', 'rb') as file:
        model = pickle.load(file)

    return model

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

instances_only = False

train_dataset = CityScapesDataset(train_imgs, train_masks, train_polygons, train_labels, sample_frac = 300, instances_only = instances_only) # 500)
test_dataset = CityScapesDataset(test_imgs, test_masks, test_polygons, train_labels, sample_frac = 20, instances_only = instances_only)
val_dataset = CityScapesDataset(val_imgs, val_masks, val_polygons, train_labels, sample_frac = 20, instances_only = instances_only) # 5)

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

lr = 1e-3 # 3e-5 # 1e-3 for mask rcnn (maybe 1e-5?)
weight_decay = 0

model = UNet_Model.UNetModel(in_channels = 3, num_classes = 9) # 9) # 21)

model = load_checkpoint("unet_model_2169_ep_49")

# model_rcnn = MaskRCNN_Model.MaskRCNN_Model(model = None, checkpoint = None, num_classes = 9).get_model()

# with open(f'./checkpoints/mask_rcnn_test_ep_15.pkl', 'rb') as file:
#     model_rcnn = pickle.load(file)

start_epoch = 0

# pos_weight = torch.ones([9, 512, 1024])
# pos_weight[0] = 0
# loss_function = torch.nn.BCEWithLogitsLoss() #pos_weight=pos_weight)

# weights = torch.tensor([0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
weights = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                        5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 
                        0.0, 0.0])
loss_function = torch.nn.CrossEntropyLoss()
# loss_function = DiceLoss()
# loss_function = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay) #1e-3, weight_decay = 0)
# mask_rcnn_optimiser = torch.optim.Adam(model_rcnn.parameters(), lr = lr, weight_decay = weight_decay)

# optimiser = torch.optim.AdamW(model.parameters(), lr=0.1)

logger = ''
wandb_logger = Logger(f"{model.name}_test", project='instance-segmentation-project')
logger = wandb_logger.get_logger()

print(model.name)
print(type(loss_function))
print(f"lr: {lr}")
print(f"weight_decay: {weight_decay}")

print(f'Device: {device}')

# average_losses, average_dice_coef = train(model, train_dataloader, val_dataloader, loss_function, optimiser, logger, epochs = 30, start_epoch = start_epoch)
# average_losses, train_loss = train_with_instances(model_rcnn, train_dataloader, val_dataloader, loss_function, mask_rcnn_optimiser, logger, epochs = 30, start_epoch = start_epoch)

# print(average_losses)
# print(average_dice_coef)
print("--------------------AAAAAAAAAA----------------------")
# print(train_loss)

# with open(f'./checkpoints/unet_model_9701_ep_199.pkl', 'rb') as file:
#     model = pickle.load(file)

test_losses, test_dice_coefs, y_pred = test(model, test_dataloader, loss_function, logger)

# with open(f'./checkpoints/mask_rcnn_test_ep_25.pkl', 'rb') as file:
#     model_rcnn = pickle.load(file)

# test_with_instances(model_rcnn, val_dataloader, loss_function, logger)

# print(test_losses)
# print(test_dice_coefs)

# labels_again = torch.argmax(y_pred, dim=1)
# print(labels_again)

# img_arr = labels_again.detach().cpu().numpy()
# print(img_arr[0])


#-------------------------------------------------------------------