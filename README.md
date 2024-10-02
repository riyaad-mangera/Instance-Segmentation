# Instance-Segmentation

This repository contains code which was used to evaluate different Image Segmentaion models on the Cityscapes Dataset. Additionally, this repository contains the Pytorch Implementation of the SwinNet3+ model.

The dataset used can be obtained from the Cityscapes Website below, with the following files needing to be downloaded:
    - gtFine_trainvaltest.zip
    - leftImg8bit_trainvaltest.zip

    https://www.cityscapes-dataset.com/

Please note that before training this dataset, a few steps are recommended:

    - Firstly, it is recommended to run the createTrainIdInstanceImgs.py helper file to generate the correct ground truth labels for the models.

    - Secondly, the structure of the dataset should be in the following format:
        - Train features example: root/dataset/features/leftImg8bit/train/*city_name*/*image.png*
        - Train labels example: root/dataset/labels/gtFine_trainvaltest/gtFine/train/*city_name*/*image.png*

To train or test this model, first install the required libraries listed in the requirements.txt file, or requirements_yolo.txt to train the dataset on a YOLOv8 model, after which the model the train.py file can be edited to perform the required train or testing task.

Logs of previous training runs of these models can be found in the following Wandb loggers:
    https://wandb.ai/riyaad-mangera/instance-segmentation-project
    https://wandb.ai/riyaad-mangera/YOLOv8/


# References

AdeelH (2020). pytorch-multi-class-focal-loss/focal_loss.py at master · AdeelH/pytorch-multi-class-focal-loss. [online] GitHub. Available at: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

Fang, G., (2019). pytorch-msssim/pytorch_msssim/ssim.py at master · VainF/pytorch-msssim. [online] GitHub. Available at: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py

FernandoPC25 (2024). Mastering U-Net: A Step-by-Step Guide to Segmentation from Scratch with PyTorch. [online] Medium. Available at: https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114

Hyounjun-Oh (2024). GitHub - Hyounjun-Oh/YOLOv8_cityscapes: How to implement the Cityscapes dataset in YOLOv8. [online] GitHub. Available at: https://github.com/Hyounjun-Oh/YOLOv8_cityscapes/

mcordts (2016). cityscapesScripts/cityscapesscripts/helpers/labels.py at master · mcordts/cityscapesScripts. [online] GitHub. Available at: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

mcordts (2016). cityscapesScripts/cityscapesscripts/preparation/createTrainIdInstanceImgs.py at master · mcordts/cityscapesScripts. [online] GitHub. Available at: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdInstanceImgs.py

mcordts (2016). cityscapesScripts/cityscapesscripts/preparation/createTrainIdInstanceImgs.py at master · mcordts/cityscapesScripts. [online] GitHub. Available at: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py

pytorch.org. (n.d.). TorchVision Object Detection Finetuning Tutorial — PyTorch Tutorials 1.5.0 documentation. [online] Available at: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

sfczekalski (2020). attention_unet/model.py at master · sfczekalski/attention_unet. [online] GitHub. Available at: https://github.com/sfczekalski/attention_unet/blob/master/model.py

ZJUGiveLab (2020). UNet-Version/models/UNet_3Plus.py at master · ZJUGiveLab/UNet-Version. [online] GitHub. Available at: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/UNet_3Plus.py