from models import YOLO_Model

yolo_model = YOLO_Model.YOLOModel().get_model()

print(yolo_model)

results = yolo_model.train(data="data.yaml", epochs = 5, batch = 1, device = device)

print("train done")

results = yolo_model.val()

print("val done")

results = yolo_model("https://ultralytics.com/images/bus.jpg")

print("predict done")

success = yolo_model.export(format="onnx")