import cv2
import cvzone
import math
from ultralytics import YOLO

# path to your test video
video = "video.mp4"

cap = cv2.VideoCapture(video)

# path to your model
model = YOLO("model.pt")

classnames = ['cars']

while True:
    ret, frame = cap.read()

    # Replay video
    if not ret:
        cap = cv2.VideoCapture(video)
        continue

    frame = cv2.resize(frame, (640, 480))

    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            confidence = box.conf[0]
            conf = math.ceil(confidence * 100)

            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]

            if conf > 40:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()



