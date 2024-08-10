from ultralytics import YOLO

# Load a model
model = YOLO("/home/anhle/workspace/iuh_tgmt/YOLO/runs/detect/train5/weights/best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
arr = []
for i in range(700, 800):
    arr.append("./dataset/images/img%d.jpg"%i)
    
results = model(arr)  # return a list of Results objects

# Process results list
i =0
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="./results/ret%d.jpg"%i)  # save to disk
    i+=1