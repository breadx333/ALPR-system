import cv2
from ultralytics import YOLO
import easyocr

model = YOLO('yolov8x.pt')
ALPR = YOLO('license_plate_detector.pt')

result = model.predict("test.jpg")[0]

img = cv2.imread('test.jpg') 

reader = easyocr.Reader(['en'], gpu=True)

for box in result.boxes:
    class_id = result.names[box.cls[0].item()]
    print(class_id)
    if (class_id == 'car'):
        conf = round(box.conf[0].item(), 2)
        if (conf >= 0.8):
            cords_xyxy = box.xyxy[0].tolist()
            cords_xyxy = [round(x) for x in cords_xyxy]

            cv2.rectangle(img, (cords_xyxy[0],cords_xyxy[1]), (cords_xyxy[2],cords_xyxy[3]), (255, 0, 0), 2)

            cropped_img = img[cords_xyxy[1]:cords_xyxy[3], cords_xyxy[0]:cords_xyxy[2]]

            #cv2.imshow("Cropped image", cropped_img) 
            #cv2.waitKey(0)

            resultALPR = ALPR.predict(cropped_img)[0]

            boxALPR = resultALPR.boxes

            confALPR = round(boxALPR.conf[0].item(), 2) if boxALPR.conf.nelement() else 0

            if (confALPR > 0.5):
                cords_xyxy_ALPR = boxALPR.xyxy[0].tolist()
                cords_xyxy_ALPR = [round(x) for x in cords_xyxy_ALPR]

                cropped_img_plate = cropped_img[cords_xyxy_ALPR[1]:cords_xyxy_ALPR[3], cords_xyxy_ALPR[0]:cords_xyxy_ALPR[2]]

                resultOCR = reader.readtext(cropped_img_plate)
                print(resultOCR)

                cv2.rectangle(cropped_img, (cords_xyxy_ALPR[0],cords_xyxy_ALPR[1]), (cords_xyxy_ALPR[2],cords_xyxy_ALPR[3]), (255, 0, 0), 2)

                cv2.imshow("Cropped image", cropped_img_plate) 
                cv2.waitKey(0)

            
        '''
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        print("Object type:", class_id)
        print("Coordinates:", cords)
        print("Probability:", conf)
        print("---")

        cv2.rectangle(img, (cords[0],cords[1]), (cords[2],cords[3]), (255, 0, 0), 2)

        '''

#print(model.model.names)

#1262, 243, 1499, 450

cv2.imshow('window_name', img) 
cv2.waitKey(0)
cv2.destroyAllWindows() 