from ultralytics import YOLO

#Trainig a model from scratch

train_model = YOLO("yolov8n.yaml")

results = train_model.train(data="config.yaml", epochs=100)

