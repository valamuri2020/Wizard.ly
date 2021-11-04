# TODO: reorder imports
# TODO: add page scrolling
# TODO: add zoom images 
# TODO: fix audio input screen freeze

import cv2
import numpy as np 
import keyboard
import mouse
import math 
from screeninfo import get_monitors

from colors import colors
import hand_tracking_module as htm

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import speech_detector as sr

cap = cv2.VideoCapture(0)

# webcam settings
camWidth, camHeight = 640, 480
scrWidth, scrHeight = get_monitors()[0].width, get_monitors()[0].height 
# 1920, 1080
cap.set(3, camWidth)
cap.set(4, camHeight)

# hand detector variables
detector = htm.HandDetector(detectionConfidence=0.85, maxHands=1)
smootheningFactor = 7
prevX, prevY = 0, 0
currX, currY = 0, 0

# audio control variables
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax, _ = volume.GetVolumeRange()

barVol = 400
vol = 0
volPercent = 0

# speech recognition listener
listener = sr.SpeechDetector()

while True:
    success, img = cap.read()
    
    img = detector.findHands(img)
    landmarks, boundingBox = detector.findPosition(img)
    # if hand detected, len(landmarks) will be nonzero
    
    """ Virtual Mouse """
    if len(landmarks) != 0:
        x1, y1 = landmarks[8][1:]
        x2, y2 = landmarks[12][1:]

        # returns a list of fingers that are open; thumb is at 0th index 
        fingers = detector.fingersOpen()
        # constant to reduce mousepad size
        frameR = 120
        
        # when middle, ring, and pinky fingers closed
        if sum(fingers[2:]) == 0:     
            # reduce virtual mousepad area, then draw
            cv2.rectangle(img, (frameR, frameR), (camWidth - frameR, camHeight - frameR), colors["purple"], 2)   
            # convert camera coordinates to screen coordinates
            x3 = np.interp(x1, (frameR, camWidth - frameR), (0, scrWidth))
            y3 = np.interp(y1, (frameR, camHeight - frameR), (0, scrHeight))
            
            # smoothen values to prevent mouse from shaking 
            currX = prevX + (x3 - prevX) / smootheningFactor
            currY = prevY + (y3 - prevY) / smootheningFactor
            
            # flipping x value to make cursor move in the same direction as hand by doing scrWidth - currX
            mouse.move(scrWidth - currX, currY)
            cv2.circle(img, (x1, y1), 15, colors["purple"], cv2.FILLED)
            prevX, prevY = currX, currY
        
        # if index and middle fingers are open
        if fingers[1] == 1 and fingers[2] == 1:  
            # draw mousepad
            cv2.rectangle(img, (frameR, frameR), (camWidth - frameR, camHeight - frameR), colors["purple"], 2)
                  
            length, img, lineInfo = detector.calculateDistance(8, 12, img)
            # center of line connecting index and middle fingers
            cLineX, cLineY = lineInfo[4], lineInfo[5]
            if length < 16:
                cv2.circle(img, (cLineX, cLineY), 15, colors["green"], cv2.FILLED)
                mouse.click()
               
    """ Gesture Control Volume """
    if len(landmarks) != 0: 
        x1, y1 = landmarks[4][1:]
        x2, y2 = landmarks[8][1:]
        # center of line
        cx, cy = (x1+x2) // 2, (y1+y2)//2
        
        length = math.hypot(x2-x1, y2-y1)
        # hand range: 20 - 180
        # vol range: -65.25 - 0
        vol = np.interp(length, [20,125], [volMin, volMax])
        barVol = np.interp(vol, [20, 125], [400,150])
        volPercent = np.interp(vol, [20, 125], [0, 100])

        fingers = detector.fingersOpen()
        
        # volume changing activated when pinky down and other fingers open so changes do not occur when a fist is made
        if not(fingers[4]) and sum(fingers[1:4]) == 3:
            cv2.circle(img, (x1,y1), 12, colors["purple"], cv2.FILLED)
            cv2.circle(img, (x2,y2), 12, colors["purple"], cv2.FILLED)
            cv2.circle(img, (cx,cy), 12, colors["purple"], cv2.FILLED)
            cv2.line(img, (x1, y1), (x2,y2), colors["purple"], thickness=3)
            
            volume.SetMasterVolumeLevel(vol, None)
            cv2.line(img, (x1, y1), (x2,y2), colors["blue"], thickness=3)    
        
        if length < 20:
            cv2.circle(img, (cx,cy), 12, colors["blue"], cv2.FILLED)
    
    """ Speech to text """
    if len(landmarks) != 0:
        fingers = detector.fingersOpen()
        
        x1, y1 = landmarks[4][1:]
        x2, y2 = landmarks[20][1:]
        
        if sum(fingers[1:4]) == 0 and fingers[4]:
            cv2.putText(img, "Listening...", (camWidth//2 - 30, camHeight), cv2.FONT_HERSHEY_PLAIN, 2, colors["blue"], 2)
            cv2.circle(img, (x1,y1), 12, colors["red"], cv2.FILLED)
            cv2.circle(img, (x2,y2), 12, colors["red"], cv2.FILLED)
            
            print("speech to text mode")
            text = listener.getAudio()
            print("received :", text)
            if type(text) != None :
                keyboard.write(text=text)
                


    cv2.imshow("skadoosh ✨", img)
    
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()