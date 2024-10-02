from ultralytics import YOLO

class YOLOModel():

    def __init__(self):
        self.model = YOLO("YOLOv8n-seg.pt")

    def get_model(self):
        return self.model