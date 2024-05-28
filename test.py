import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import subprocess
import psutil
import time

from watchpoints import watch

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/nice"
counter = 0

key=0
checker=0
labels = ["call", "ILY", "nice","peace","punch","spiderman","thumbsup"]

last_gesture=None
start_time = None
same_gesture_count=0

a = 4
if a == 4:
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        try:
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                if not start_time:
                    start_time = time.time()

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

                gesture=labels[index]

                if gesture==last_gesture:
                    same_gesture_count+=1
                else:
                    last_gesture=gesture
                    same_gesture_count=0
                    start_time=time.time()

                if same_gesture_count >=3 and time.time()-start_time>=3:
                    if last_gesture == 'ILY':
                        key = 1
                        print("ILY")

                    elif labels[index] == 'spiderman':
                        key = 2
                        print("spiderman")

                    elif labels[index] == 'thumbsup':
                        key = 3
                        print("thumbsup")

                    elif labels[index] == 'punch':
                        key = 4
                        print("punch")

                    elif labels[index] == 'call':
                        key = 5
                        print("call")

                    elif labels[index] == 'peace':
                        key = 6
                        print("peace")

                    elif labels[index] == 'nice':
                        key = 7
                        print("nice")


                    if checker != key:

                        print(checker, " ", key)
                        checker = key
                        if key == 1:
                            if "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe" not in (p.name() for p in psutil.process_iter()):
                                subprocess.run("C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe")
                        elif key == 2:
                            if "C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE" not in (p.name() for p in psutil.process_iter()):
                                subprocess.run("C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE")
                        elif key == 3:
                            if "C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.EXE" not in (p.name() for p in psutil.process_iter()):
                                subprocess.run("C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.EXE")
                        elif key == 4:
                            if "C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE" not in (p.name() for p in psutil.process_iter()):
                                subprocess.run("C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE")
                        elif key == 5:
                            if "C:\\Program Files\\JetBrains\\PyCharm Community Edition 2022.3.2\\bin\\pycharm64.exe" not in (p.name() for p in psutil.process_iter()):
                                subprocess.run("C:\\Program Files\\JetBrains\\PyCharm Community Edition 2022.3.2\\bin\\pycharm64.exe")
                        elif key == 6:
                            if "C:\\Windows\\explorer.exe" not in (p.name() for p in psutil.process_iter()):
                                subprocess.run("C:\\Windows\\explorer.exe")
                        elif key == 7:
                            if "C:\\Windows\\System32\\cmd.exe" not in (p.name() for p in psutil.process_iter()):
                                subprocess.run("C:\\Windows\\System32\\cmd.exe")



            cv2.imshow("Image", imgOutput)
            cv2.waitKey(1)
        except:
            pass
