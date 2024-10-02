from models import YOLO_Model

yolo_model = YOLO_Model.YOLOModel().get_model()

# print(yolo_model)

results = yolo_model.train(data="data.yaml", epochs = 50, batch = 1) #, device = 'cpu')

print("train done")

results = yolo_model.val()

print("val done")

results = yolo_model("datasets/datasets/test/images")

print("predict done")

for i, result in enumerate(results):
    result.save(f"runs/segment/tests/test_{i}.png")

# print(results)

# success = yolo_model.export(format="onnx")