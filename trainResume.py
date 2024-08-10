from ultralytics import YOLO

# Load a model
model = YOLO("./runs/detect/train/weights/last.pt")  

# Resume training
results = model.train(resume=True)