from pipe import *
import cv2
import mediapipe as mp
import time
import math
import numpy as np
from random import shuffle
import reconize_knn
from sklearn import model_selection, preprocessing
import out
import winsound

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF

dataset = []
labels  = []
ds = []

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

allAngles   = []
leftAngles  = []
rightAngles = []

pTime = 0
cTime = 0


#
# 生成训练集并训练
#

traindb = []

dataset = Dataset(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13'])
dataset.makeDataset(1.0)

for i in dataset.trainSet:

    traindb.append(i)

    #print(traindb)

reconize_knn.knn_input(np.asarray(traindb), np.asarray(dataset.trainLabel))

leftIdentity  = None
rightIdentity = None

handIdentity  = None

handNumber    = 0

n = 0

stopTime = Timer()
isTiming = False

temp_TranslatedList = []

while True:

    _ , img = cap.read()

    # 左右反转视频
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)

    if isTiming == True and stopTime.currentTime() >= 1:
        out.voice(temp_TranslatedList)
        print(temp_TranslatedList)
        temp_TranslatedList = []

    if results.multi_hand_landmarks:

        isTiming = False

        # 有两只手
        if len(results.multi_hand_landmarks) == 2:

            # 左右手判断
            if (results.multi_hand_landmarks[0].landmark[0].x < results.multi_hand_landmarks[1].landmark[0].x):
                lefthand  = Hand(results.multi_hand_landmarks[0], mpDraw, img, mpHands)
                righthand = Hand(results.multi_hand_landmarks[1], mpDraw, img, mpHands)
            else:
                lefthand  = Hand(results.multi_hand_landmarks[1], mpDraw, img, mpHands)
                righthand = Hand(results.multi_hand_landmarks[0], mpDraw, img, mpHands)

            lefthand.drawPoints()
            righthand.drawPoints()

            leftAngles  = lefthand.getAngles()
            rightAngles = righthand.getAngles()

            # 分类
            leftres   = reconize_knn.knn.predict(np.asarray([leftAngles]))
            leftrate  = reconize_knn.knn.predict_proba(np.asarray([leftAngles]))[0]

            rightres  = reconize_knn.knn.predict(np.asarray([rightAngles]))
            rightrate = reconize_knn.knn.predict_proba(np.asarray([rightAngles]))[0]

            if isGuesture(leftrate) and isGuesture(rightrate):
                
                if leftres == leftIdentity and rightres == rightIdentity:
                    n += 1
                    transed = translate([str(leftres[0]), lefthand.getDirection()], [str(rightres[0]), righthand.getDirection()])

                    if n == 10 and (len(temp_TranslatedList) == 0 or transed != temp_TranslatedList[-1]):
                        temp_TranslatedList.append(transed)
                        print(leftres)
                        print(rightres)
                        #stopTime.start()

                else:
                    leftIdentity  =  leftres
                    rightIdentity =  rightres
                    n = 0

        # 有一只手
        elif len(results.multi_hand_landmarks) == 1:

            hand = Hand(results.multi_hand_landmarks[0], mpDraw, img, mpHands)
            hand.drawPoints()
            allAngles = hand.getAngles()

            handedness = results.multi_handedness[0].classification[0].label

            # 分类
            handres   = reconize_knn.knn.predict(np.asarray([allAngles]))
            handrate  = reconize_knn.knn.predict_proba(np.asarray([allAngles]))[0]

            if isGuesture(handrate):
                
                if handres == handIdentity:

                    n += 1
                    transed = translateSingle([str(handres[0]), hand.getDirection()], handedness)

                    if n == 10 and (len(temp_TranslatedList) == 0 or transed != temp_TranslatedList[-1]):
                        print(translateSingle([handres, hand.getDirection()], handedness))
                        #print(handres)
                        if transed == 'a':
                            for i in range(5):
                                for i in range(5):
                                    winsound.Beep(600, 200)
                                    time.sleep(0.1)
                                    winsound.Beep(600, 200)
                                    time.sleep(0.1)
                                    winsound.Beep(600, 200)
                                    time.sleep(0.1)
                                    winsound.Beep(600, 200)
                                    time.sleep(0.3)
                                out.speak("我需要帮助")
                        else:
                            temp_TranslatedList.append(transed)
                        #stopTime.start()

                else:
                    handIdentity  =  handres
                    n = 0

            key = cv2.waitKey(50) & 0xFF
            
            Key   = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z' , 'x' , 'c' , 'v' ]
            
            Label = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

            Key = list(map(ord, Key))

            
            if key in Key:

                labels.append(Label[Key.index(key)])
                ds.append(allAngles)

                '''
                r = open('13.txt', 'a')

                for i in allAngles:
                    r.write(str(i)+',')
                
                r.write("\n")
                r.close()
                '''

                print(ds)
                print(labels)

            if key == ord('t'):
                
                #print(dataset.trainSet)
                '''
                db = []

                for i in dataset.trainSet:
                    db.append(i)

                print(db)
                '''

                reconize_knn.knn_input(np.asarray(ds), np.asarray(labels))

                print(ds)
                print(labels)

            if key == ord('r'):
                res = reconize_knn.knn.predict(np.asarray([allAngles]))
                rate = reconize_knn.knn.predict_proba(np.asarray([allAngles]))
                print(rate[0][1])
                print(res)
                print(rate)

    else:
        if isTiming == False:
            isTiming = True
            stopTime.start()

        # 不知道干嘛用的，先注释了
        '''
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                # print(cx,cy)
                #if id ==0:
                #cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
        '''
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(temp_TranslatedList), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)