from ultralytics import YOLO

class YOLOModel():

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def get_model(self):
        return self.model