import numpy as np
import torchvision.tv_tensors
from torch.utils.data import DataLoader
from dataset import CityScapesFiles, CityScapesDataset
from logger import Logger
import models
import torch, torchvision
import wandb
import pickle
from torchmetrics import Dice, JaccardIndex
from models import UNet_Model, UNet3PlusAttn_Model, MaskRCNN_Model, UNetR3PlusAttn_Model, SwinNet3PlusModel

# Uncomment when running on hyperion
# os.environ['https_proxy'] = 'http://hpc-proxy00.city.ac.uk:3128'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Device: {device}')

TRAIN_BATCH_SIZE = 3
TEST_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1

def train(model, train_loader, val_loader, loss_function, optimiser, logger, epochs = 5, start_epoch = 0):

    model.to(device)
    loss_function.to(device)

    average_val_losses = []
    average_val_dice_coef = []

    dice = Dice(num_classes = 9, ignore_index = 0).to(device)
    iou = JaccardIndex(task = "multiclass", num_classes = 9, ignore_index = 0).to(device)

    model.train()

    for epoch in range(epochs):

        print(f'Epoch: {epoch}')

        train_loss = []
        batch_losses = []
        batch_dice_coeffs = []
        batch_ious = []

        for idx, batch in enumerate(train_loader):

            print(f'Epoch: {epoch}, Batch {idx} of {int(len(train_loader.dataset)/TRAIN_BATCH_SIZE)}')

            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            y_pred = model(image)

            optimiser.zero_grad()

            loss1 = loss_function(y_pred, mask)

            loss = loss1

            # Uncomment when using deep supervision
            # loss1 = loss_function(y_pred[0], mask, weight = 1 / (2 ** 0))
            # loss2 = loss_function(y_pred[1], mask, weight = 1 / (2 ** 1))
            # loss3 = loss_function(y_pred[2], mask, weight = 1 / (2 ** 2))
            # loss4 = loss_function(y_pred[3], mask, weight = 1 / (2 ** 3))
            # loss5 = loss_function(y_pred[4], mask, weight = 1 / (2 ** 4))

            # loss = loss1 + loss2 + loss3 + loss4 + loss5

            print(f"Loss: {loss}")
            
            loss.backward()
            optimiser.step()

            train_loss.append(loss.item())

            with torch.no_grad():
                model.eval()
                
                print(f"Validating {int(len(val_loader.dataset)/VAL_BATCH_SIZE)} batches")

                for val_idx, val_batch in enumerate(val_loader):

                    val_image = val_batch["image"].to(device)
                    val_mask = val_batch["mask"].to(device)

                    val_y_pred = model(val_image)

                    val_dice_coeff = dice(torch.argmax(val_y_pred, dim=1), torch.argmax(val_mask, dim=1))
                    val_iou = iou(torch.nn.functional.softmax(val_y_pred, dim=1), torch.argmax(val_mask, dim = 1))

                    val_loss1 = loss_function(val_y_pred, val_mask)

                    val_loss = val_loss1

                    # Uncomment when using deep supervision
                    # val_loss1 = loss_function(val_y_pred[0], val_mask, weight = 1 / (2 ** 0))
                    # val_loss2 = loss_function(val_y_pred[1], val_mask, weight = 1 / (2 ** 1))
                    # val_loss3 = loss_function(val_y_pred[2], val_mask, weight = 1 / (2 ** 2))
                    # val_loss4 = loss_function(val_y_pred[3], val_mask, weight = 1 / (2 ** 3))
                    # val_loss5 = loss_function(val_y_pred[4], val_mask, weight = 1 / (2 ** 4))

                    # val_loss = val_loss1 + val_loss2 + val_loss3 + val_loss4 + val_loss5

                    batch_losses.append(val_loss.item())
                    batch_dice_coeffs.append(val_dice_coeff.item())
                    batch_ious.append(val_iou.item())

                    if (idx == TRAIN_BATCH_SIZE - 1) and (val_idx == VAL_BATCH_SIZE - 1):

                        y_pred_labels_again = torch.argmax(val_y_pred, dim=1)
                        y_pred_img_arr = y_pred_labels_again.detach().cpu().numpy()

                        y_true_labels_again = torch.argmax(val_mask, dim=1)
                        y_true_img_arr = y_true_labels_again.detach().cpu().numpy()

                        # Semantic Labels
                        # class_labels = {0: "road", 1: "sidewalk", 2: "building", 3:"wall", 4:"fence", 
                        #                 5:"pole", 6:"traffic_light", 7:"traffic_sign", 8:"vegetation", 
                        #                 9:"terrain", 10:"sky", 11:"person", 12:"rider", 
                        #                 13:"car", 14:"truck", 15:"bus", 16:"train", 
                        #                 17:"motorcycle", 18:"bicycle", 19:"license_plate", 20:"unlabeled"}

                        #Instance Labels
                        class_labels = {1: "person", 2: "rider", 3: "car", 4: "truck",
                                        5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"
                                        }

                        invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])

                        orig_img = invTrans(val_image)
                        orig_img = orig_img.detach().cpu().numpy()

                        original_image = np.transpose(orig_img[0],(1, 2, 0))

            model.train()

        if logger != '':
            logger.log({"validation_predictions" : wandb.Image(original_image, masks={"predictions" : {"mask_data" : y_pred_img_arr[0], "class_labels" : class_labels}, "ground_truth" : {"mask_data" : y_true_img_arr[0], "class_labels" : class_labels}}),
                        'train_loss': np.sum(train_loss) / len(train_loss),
                        'validation_loss': np.sum(batch_losses) / len(batch_losses), 
                        'validation_dice_coefficient': np.sum(batch_dice_coeffs) / len(batch_dice_coeffs),
                        'validation_iou': np.sum(batch_ious) / len(batch_ious)})

        if epoch % 5 == 0 or epoch == epochs - 1:    
            with open(f'./checkpoints/{model.name}_ep_{epoch + start_epoch}.pkl', 'wb') as file:
                pickle.dump(model, file)

    return average_val_losses, average_val_dice_coef

def test(model, test_loader, loss_function, logger):

    model.eval()

    dice_coefs = []
    average_losses = []
    average_dice_coef = []
    average_iou = []

    dice = Dice(num_classes = 9, ignore_index = 0).to(device)
    iou = JaccardIndex(task = "multiclass", num_classes = 9, ignore_index = 0).to(device)

    for idx, batch in enumerate(test_loader):

        print(f'Testing Batch {idx} of {int(len(test_loader.dataset)/TEST_BATCH_SIZE)}')

        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        y_pred = model(image)

        dice_coeff = dice(torch.argmax(y_pred, dim=1), torch.argmax(mask, dim=1))
        test_iou = iou(torch.nn.functional.softmax(y_pred, dim=1), torch.argmax(mask, dim = 1))

        loss = loss_function(y_pred, mask)

        dice_coefs.append(dice_coeff.item())
        average_iou.append(test_iou.item())

        average_losses.append(np.average(loss.item()))
        average_dice_coef.append(np.average(dice_coeff.item()))

        y_pred_labels_again = torch.argmax(y_pred, dim=1)
        y_pred_img_arr = y_pred_labels_again.detach().cpu().numpy()

        y_true_labels_again = torch.argmax(mask, dim=1)
        y_true_img_arr = y_true_labels_again.detach().cpu().numpy()

        class_labels = {1: "person", 2: "rider", 3: "car", 4: "truck", 
                        5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"
                        }

        invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])

        orig_img = invTrans(image)
        orig_img = orig_img.detach().cpu().numpy()

        original_image = np.transpose(orig_img[0],(1, 2, 0))

        if logger != '':
            logger.log({"test_predictions" : wandb.Image(original_image, masks={"predictions": {"mask_data" : y_pred_img_arr[0], "class_labels" : class_labels}, "ground_truth" : {"mask_data" : y_true_img_arr[0], "class_labels" : class_labels}})})

    if logger != '':
        logger.log({'test_dice_coefficients': np.sum(dice_coefs) / len(dice_coefs),
                    'test_iou': np.sum(average_iou) / len(average_iou)})

    return average_losses, average_dice_coef, 0

# Convert bounding boxes into a Wandb compatible format
def bounding_boxes(boxes, labels, class_id_to_label, scores = None, pred = True):
    
    all_boxes = []

    for idx, box in enumerate(boxes):

        if scores is None:
            acc = 0

        else:
            acc = scores[idx]

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
                "scores": {"acc": acc, "loss": 0.0}
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
        
        all_boxes.append(box_data)

    return all_boxes

def train_with_instances(model, train_loader, val_loader, loss_function, optimiser, logger, epochs = 5):
    model.to(device)
    loss_function.to(device)

    model.train()

    for epoch in range(epochs):

        print(f'Epoch: {epoch}')

        train_loss = []
        batch_losses = []
        batch_iou = []

        for idx, (image, targets) in enumerate(train_loader):

            print(f'Epoch: {epoch}, Batch {idx} of {int(len(train_loader.dataset)/TRAIN_BATCH_SIZE)}')

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

                y_pred = model(image, target)

                optimiser.zero_grad()

                loss = sum([loss for loss in y_pred.values()])
                loss.backward()
                optimiser.step()

                train_loss.append(loss.item())

                with torch.no_grad():
                    model.eval()
                    
                    for val_idx, (val_image, val_targets) in enumerate(val_loader):

                        print(f'\tValidating Batch {val_idx} of {int(len(val_loader.dataset)/VAL_BATCH_SIZE)}')

                        val_image = val_image.to(device)
                        # val_image = [torch.squeeze(val_image)]

                        val_target = [{k: v for k, v in val_targets.items()}]            

                        for index, t in enumerate(val_target):

                            val_target[index]["boxes"] = torch.squeeze(val_target[index]["boxes"]).to(device)
                            val_target[index]["masks"] = torch.squeeze(val_target[index]["masks"]).to(device)
                            val_target[index]["labels"] = torch.squeeze(val_target[index]["labels"]).to(device)

                        val_y_pred = model(val_image)

                        batch_losses.append(val_y_pred)

                        # Placeholder as iou was unable to be calculated
                        iou = -1

                        batch_iou.append(iou)
                        
                        if (idx == TRAIN_BATCH_SIZE - 1) and (val_idx == VAL_BATCH_SIZE - 1):

                            invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])

                            class_id_to_label = {1: "person", 2: "rider", 3: "car", 4: "truck",
                                                5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"
                                                }

                            boxes = val_y_pred[0]["boxes"].tolist()
                            labels = val_y_pred[0]["labels"].tolist()
                            scores = val_y_pred[0]["scores"].tolist()

                            true_labels = val_target[0]["labels"].tolist()

                            prob_threshold = 0.0

                            # Only log masks and boxes if there is at least one box
                            # Wandb logger will crash if there are no boxes
                            if len(boxes) > 0:

                                orig_img = invTrans(val_image)

                                orig_img = orig_img.detach().cpu().numpy()

                                original_image = np.transpose(np.squeeze(orig_img),(1, 2, 0))

                                thres_prob_masks = val_y_pred[0]["masks"] > prob_threshold

                                y_pred_labels_again = torch.argmax(torch.squeeze(thres_prob_masks.to(torch.int64), 1), dim=0)
                                y_true_labels_again = torch.argmax(val_target[0]["masks"], dim=0)

                                masks = y_pred_labels_again.detach().cpu().numpy()
                                true_masks = y_true_labels_again.detach().cpu().numpy()

                                for true_idx in range(1, len(true_labels)):
                                    true_masks[true_masks == true_idx] = true_labels[true_idx]

                                for pred_idx in range(1, len(labels)):
                                    masks[masks == pred_idx] = labels[pred_idx]

                                true_labels = val_target[0]["labels"].tolist()

                                bb = bounding_boxes(boxes, labels, class_id_to_label, scores, pred=True)
                                true_bb = bounding_boxes(val_target[0]["boxes"], true_labels, class_id_to_label, pred=False)

                model.train()

        if logger != '':
            logger.log({'train_loss': np.sum(train_loss) / len(train_loss),
                        'validation_iou': np.sum(batch_iou) / len(batch_iou), #np.sum(batch_iou) / len(batch_iou)
                        'validation_predictions': wandb.Image(original_image, 
                                                               boxes = {"predictions": {"box_data": bb, "class_labels" : class_id_to_label}, "ground_truth" : {"box_data": true_bb, "class_labels" : class_id_to_label}},
                                                               masks={"predictions": {"mask_data" : masks, "class_labels" : class_id_to_label}, "ground_truth": {"mask_data" : true_masks, "class_labels" : class_id_to_label}}
                                                               )
                        })

        if epoch != 0 and epoch % 5 == 0:    
            with open(f'./checkpoints/mask_rcnn_test_ep_{epoch}.pkl', 'wb') as file:
                pickle.dump(model, file)

    return "average_val_losses", train_loss

def test_with_instances(model, test_loader, logger):
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

            # Placeholder as iou was unable to be calculated
            iou = -1

            batch_iou.append(iou)

            invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])

            class_id_to_label = {1: "person", 2: "rider", 3: "car", 4: "truck",
                                5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"
                                }

            boxes = y_pred[0]["boxes"].tolist()
            labels = y_pred[0]["labels"].tolist()
            scores = y_pred[0]["scores"].tolist()

            true_labels = target[0]["labels"].tolist()

            prob_threshold = 0

            # Only log masks and boxes if there is at least one box
            # Wandb logger will crash if there are no boxes
            if len(boxes) > 0:

                orig_img = invTrans(image)

                orig_img = orig_img.detach().cpu().numpy()

                original_image = np.transpose(np.squeeze(orig_img),(1, 2, 0))

                thres_prob_masks = y_pred[0]["masks"] > prob_threshold

                y_pred_labels_again = torch.argmax(torch.squeeze(thres_prob_masks.to(torch.int64), 1), dim=0)
                y_true_labels_again = torch.argmax(target[0]["masks"], dim=0)

                masks = y_pred_labels_again.detach().cpu().numpy()
                true_masks = y_true_labels_again.detach().cpu().numpy()

                for true_idx in range(1, len(true_labels)):
                    true_masks[true_masks == true_idx] = true_labels[true_idx]

                for pred_idx in range(1, len(labels)):
                    masks[masks == pred_idx] = labels[pred_idx]

                true_labels = target[0]["labels"].tolist()

                bb = bounding_boxes(boxes, labels, class_id_to_label, scores, pred=True)
                true_bb = bounding_boxes(target[0]["boxes"], true_labels, class_id_to_label, pred=False)

                logger.log({"test_predictions" : wandb.Image(original_image, 
                                                                boxes = {"predictions": {"box_data": bb, "class_labels" : class_id_to_label}, "ground_truth" : {"box_data": true_bb, "class_labels" : class_id_to_label}},
                                                                masks={"predictions": {"mask_data" : masks, "class_labels" : class_id_to_label}, "ground_truth": {"mask_data" : true_masks, "class_labels" : class_id_to_label}}
                                                                )})
                

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

# Change to True if running a Mask R-CNN Model
masks_and_boxes = False

train_dataset = CityScapesDataset(train_imgs, train_masks, train_polygons, sample_frac = 300, masks_and_boxes = masks_and_boxes) #2400, instances_only = instances_only) # 500)
test_dataset = CityScapesDataset(test_imgs, test_masks, test_polygons, sample_frac = 40, masks_and_boxes = masks_and_boxes)
val_dataset = CityScapesDataset(val_imgs, val_masks, val_polygons, sample_frac = 10, masks_and_boxes = masks_and_boxes) # 5)

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

lr = 1e-6
weight_decay = 1e-7

# model = UNet_Model.UNetModel(in_channels = 3, num_classes = 9)
# model = UNet3PlusAttn_Model.UNet3PlusAttnModel(in_channels = 3, num_classes = 9)
model = SwinNet3PlusModel.SwinNet3PlusModel(in_channels = 3, num_classes = 9)
# model = load_checkpoint("unet3+Attn_model_2164_ep_49")

# model_rcnn = MaskRCNN_Model.MaskRCNN_Model(model = None, checkpoint = None, num_classes = 9).get_model()
# model = load_checkpoint("mask_rcnn_test_ep_15")

start_epoch = 0

loss_function = torch.nn.CrossEntropyLoss()
# loss_function = UNet3PlusLoss(device = device)

optimiser = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay) #1e-3, weight_decay = 0)
# mask_rcnn_optimiser = torch.optim.Adam(model_rcnn.parameters(), lr = lr, weight_decay = weight_decay)

logger = ''

# Uncomment to log runs to Wandb
# wandb_logger = Logger(f"{model.name}_test_hyp", project='instance-segmentation-project')
# logger = wandb_logger.get_logger()

print(model.name)
print(f"loss_fn: {type(loss_function)}")
print(f"lr: {lr}")
print(f"weight_decay: {weight_decay}")
print(f"star_ep: {start_epoch}")
print(f'Device: {device}')

average_losses, average_dice_coef = train(model, train_dataloader, val_dataloader, loss_function, optimiser, logger, epochs = 50, start_epoch = start_epoch)
# average_losses, train_loss = train_with_instances(model_rcnn, train_dataloader, val_dataloader, loss_function, mask_rcnn_optimiser, logger, epochs = 10)

# Load checpoint if testing
# load_checkpoint("model_name")

test_losses, test_dice_coefs, y_pred = test(model, test_dataloader, loss_function, logger)
# test_with_instances(model_rcnn, test_dataloader, loss_function, logger)

# Get number of trainable parameters
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Trainable Params: {pytorch_total_params}")

# Visualise structure of the model
# model_graph = draw_graph(model, input_size=(1, 3, 512, 1024), roll = True)
# graph = model_graph.visual_graph
# # print(graph)
# graph.view("SwinNet3+_Graph", directory="test_masks")