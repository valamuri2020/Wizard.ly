# CV Swiss knife, tool #1
import cv2
import numpy as np
import mediapipe as mp
import time
import modules.hand_tracking_module as htm
import math 
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from colors import colors

wCam, hCam = 640, 480
prevTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detectionConfidence=0.75)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax, _ = volume.GetVolumeRange()

barVol = 400
vol = 0
volPercent = 0
    

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0: 
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        # center of line
        cx, cy = (x1+x2) // 2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 12, colors["purple"], cv2.FILLED)
        cv2.circle(img, (x2,y2), 12, colors["purple"], cv2.FILLED)
        cv2.circle(img, (cx,cy), 12, colors["purple"], cv2.FILLED)
        cv2.line(img, (x1, y1), (x2,y2), colors["purple"], thickness=3)
        
        length = math.hypot(x2-x1, y2-y1)
        # hand range is 20 - 180
        # vol range is -65.25 - 0
        vol = np.interp(length, [20,125], [volMin, volMax])
        barVol = np.interp(vol, [20, 125], [400,150])
        volPercent = np.interp(vol, [20, 125], [0, 100])

        print(length, vol)
        fingers = detector.fingersOpen()
        
        # volume changing activated when pinky down and other fingers open so changes do not occur when a fist is made
        if not(fingers[4]) and sum(fingers[1:4]) == 3:
            volume.SetMasterVolumeLevel(vol, None)
            cv2.line(img, (x1, y1), (x2,y2), colors["blue"], thickness=3)    
        if length < 20:
            cv2.circle(img, (cx,cy), 12, colors["blue"], cv2.FILLED)
            
            
    # cv2.rectangle(img, (50,150), (85,400), (0, 200, 0), 3)
    # cv2.rectangle(img, (50,150), (85, int(vol)), (0, 200, 0), cv2.FILLED)

    # cv2.putText(img, f"{int(volPercent)}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, colors["blue"], 2)
    
    currentTime = time.time()
    fps = 1/(currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, colors["blue"], 2)
    cv2.imshow("image", img)
    cv2.waitKey(1)

