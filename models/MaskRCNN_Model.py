import torchvision
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN_Model():
    def __init__(self, model = None, checkpoint = None, num_classes = 8):

        self.checkpoint = checkpoint

        if model == None:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained = True)

        else:
            self.model = model

        self.model.name = f"mask_rcnn_model_{''.join(random.sample([str(x) for x in range(10)], 4))}"

        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_channels = self.in_features, num_classes = num_classes)

        self.in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.hidden_layer = 256

        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(self.in_features_mask, self.hidden_layer, num_classes)

    def get_model(self):
        
        return self.model
