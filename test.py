import cv2
from ultralytics import YOLO

ALPR = YOLO('license_plate_detector.pt')

resultALPR = ALPR.predict("test2.jpg")[0]

box = resultALPR.boxes

img = cv2.imread('test2.jpg') 

cords_xyxy_ALPR = box.xyxy[0].tolist()
cords_xyxy_ALPR = [round(x) for x in cords_xyxy_ALPR]

cv2.rectangle(img, (cords_xyxy_ALPR[0],cords_xyxy_ALPR[1]), (cords_xyxy_ALPR[2],cords_xyxy_ALPR[3]), (255, 0, 0), 2)

cv2.imshow('window_name', img) 
cv2.waitKey(0)
cv2.destroyAllWindows() 