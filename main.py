import cv2
from ultralytics import YOLO
import easyocr
import sqlite3
from flask import Flask, jsonify, request, send_file, render_template
from PIL import Image
import base64
import io

model = YOLO('yolov8x.pt')
ALPR = YOLO('license_plate_detector.pt')

reader = easyocr.Reader(['en'], gpu=False)

app = Flask(__name__)

@app.route("/")
def root():
    return render_template('index.html')
    
@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image_file"]

    image = cv2.imread(file.stream)

    result = model.predict(image)[0].boxes

    classes = result.cls.tolist()
    classes = [round(x) for x in classes]

    index = 0

    for cls in classes:
        if (cls == 2):
            conf = round(result.conf[index].item(), 2)
            if (conf >= 0.5):
                cords_xyxy = result.xyxy[index].tolist()
                cords_xyxy = [round(x) for x in cords_xyxy]

                cv2.rectangle(image, (cords_xyxy[0],cords_xyxy[1]), (cords_xyxy[2],cords_xyxy[3]), (255, 0, 0), 2)

                cropped_img = image[cords_xyxy[1]:cords_xyxy[3], cords_xyxy[0]:cords_xyxy[2]]

                resultALPR = ALPR.predict(cropped_img)[0].boxes

                #classesALPR = resultALPR.cls.tolist()
                #classesALPR = [round(x) for x in classesALPR]

                #confALPR = round(resultALPR.conf[0].item(), 2) if resultALPR.conf.nelement() else 0

                confsALPR = resultALPR.conf.tolist()
                
                if len(confsALPR) > 0:
                    indexALPR = 0
                    confsALPR = [round(x, 2) for x in confsALPR]
                    if len(confsALPR) > 1:
                        indexALPR = confsALPR.index(max(confsALPR))
                    confALPR = confsALPR[indexALPR]
                    
                    if (confALPR >= 0.5):
                        cords_xyxyALPR = resultALPR.xyxy[indexALPR].tolist()
                        cords_xyxyALPR = [round(x) for x in cords_xyxyALPR]

                        cropped_img_plate = cropped_img[cords_xyxyALPR[1]:cords_xyxyALPR[3], cords_xyxyALPR[0]:cords_xyxyALPR[2]]

                        resultOCR = reader.readtext(cropped_img_plate, allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                        if len(resultOCR) > 0:
                            characterizedResultOCR = "".join([x[1] for x in resultOCR])
                            cv2.putText(cropped_img, str(characterizedResultOCR), (cords_xyxyALPR[0], cords_xyxyALPR[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)

                        cv2.rectangle(cropped_img, (cords_xyxyALPR[0],cords_xyxyALPR[1]), (cords_xyxyALPR[2],cords_xyxyALPR[3]), (255, 0, 0), 2)
                        cv2.putText(cropped_img, str(confALPR), (cords_xyxyALPR[0], cords_xyxyALPR[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
                        #cv2.putText(cropped_img, str(resultOCR), (cords_xyxyALPR[0], cords_xyxyALPR[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)

                # indexALPR = 0

                # print("already cropped", classesALPR)

                # for clsALPR in classesALPR:
                #     if clsALPR == 0:
                #         confALPR = round(result.conf[indexALPR].item(), 2)
                #         if (confALPR >= 0.5):
                #             cords_xyxyALPR = resultALPR.xyxy[0].tolist()
                #             cords_xyxyALPR = [round(x) for x in cords_xyxyALPR]

                #             cv2.rectangle(cropped_img, (cords_xyxyALPR[0],cords_xyxyALPR[1]), (cords_xyxyALPR[2],cords_xyxyALPR[3]), (255, 0, 0), 2)

                #             cropped_img_plate = cropped_img[cords_xyxyALPR[1]:cords_xyxyALPR[3], cords_xyxyALPR[0]:cords_xyxyALPR[2]]

                #             resultOCR = reader.readtext(cropped_img_plate, allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                #     indexALPR += 1

        index += 1
                # classesALPR = resultALPR.cls.tolist()
                # classesALPR = [round(x) for x in classesALPR]

                #indexALPR = 0

                # for clsALPR in classesALPR:
                #     if (clsALPR == 0):
                #         #confALPR = round(resultALPR.conf[indexALPR].item(), 2)
                #         confALPR = round(resultALPR.conf[0].item(), 2) if resultALPR.conf.nelement() else 0
                #         if (confALPR >= 0.5):
                #             cords_xyxyALPR = resultALPR.xyxy[indexALPR].tolist()
                #             cords_xyxyALPR = [round(x) for x in cords_xyxyALPR]

                #             cv2.rectangle(cropped_img, (cords_xyxyALPR[0],cords_xyxyALPR[1]), (cords_xyxyALPR[2],cords_xyxyALPR[3]), (255, 0, 0), 2)

    # cv2.imshow("Cropped image", image) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    #return "hi"

    is_success, buffer = cv2.imencode(".jpg", image)
    io_buffer = io.BytesIO(buffer)

    data = io_buffer.read()
    data = base64.b64encode(data).decode()

    return jsonify({
                'msg': 'success', 
                'size': [image.shape[1], image.shape[0]], 
                'format': "jpg",
                'img': data
           })
"""
con = sqlite3.connect("entrance.db")
cur = con.cursor()
print(cur.execute("SELECT * FROM allowed").fetchall())

model = YOLO('yolov8x.pt')
ALPR = YOLO('license_plate_detector.pt')

result = model.predict("test_many.jpg")[0]

img = cv2.imread('test_many.jpg') 

reader = easyocr.Reader(['en'], gpu=True)

for box in result.boxes:
    class_id = result.names[box.cls[0].item()]
    #print(class_id)
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

                resultOCR = reader.readtext(cropped_img_plate, allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if (len(resultOCR)):
                    data_from_db = cur.execute("SELECT * FROM allowed")
                    characterizedResultOCR = "".join([x[1] for x in resultOCR])
                    print(characterizedResultOCR)
                    for item in data_from_db:
                        if (characterizedResultOCR in item):
                            print(f"Access is allowed for {characterizedResultOCR}")
                            break

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
"""