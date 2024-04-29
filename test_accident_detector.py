from ultralytics import YOLO
import pafy
import cv2
import numpy as np

ID2CLASS = {0: 'bike_bike_accident', 1: 'bike_object_accident', 2: 'bike_person_accident', 3: 'car_bike_accident', 4:'car_car_accident', 5: 'car_object_accident', 6: 'car_person_accident'}

def detect_accident(source):
    model = YOLO('accident_detection.pt')
    
    pafyVideoUrl = pafy.new(source).getbest(preftype='mp4').url
    cap = cv2.VideoCapture(pafyVideoUrl)
    assert cap.isOpened(), f"{pafyVideoUrl} can't be opened..."
    
    while True:
        _, frame = cap.read()
        
        result = model(frame, imgsz=640, verbose=False)
        
        boxes_obj = result[0].boxes
        for ind, bbox_conf in enumerate(boxes_obj.conf):
            if bbox_conf > 0.8:
                class_id = int(boxes_obj.cls[ind])
                class_name = ID2CLASS[class_id]
                print(f"{class_name} is detected!")
        
        cv2.imshow("Accident Detection", result[0].plot())
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
if __name__=="__main__":
    detect_accident(source="https://www.youtube.com/watch?v=J0qU5lwTcRI")