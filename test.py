from ultralytics import YOLO
import torch

## check gpu availability
print("GPU available: ", torch.cuda.is_available())

## detection
checkpoint = "./runs/detect/train/weights/last.pt"
input_image = "./dataset/test/images/image (95).jpg"
model = YOLO(checkpoint)
results = model(source=input_image, conf=0.3, show=True, save=True)

## results is an object of type 'Results' which is a list of dictionaries containing boxes, labels, scores, and keypoints
print('Results: ', results)

## access boxes and keypoints
print('boxes ', results[0].boxes)

print('Done!')