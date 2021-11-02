import cv2
import numpy as np 
import mouse
from screeninfo import get_monitors
import hand_tracking_module as htm
from colors import colors

# Process
# 1. Find hand landmarks
# 2. Get tip of index and middle fingers 
# 3. Check which fingers are open, decide click vs scroll mode 
# 4. Moving mode if only index finger open
# 5. Convert camera coordinates to screen coordinates
# 6. Smoothen 
# 7. Move mouse 
# 8. Clicking mode if both index and middle finger open clicking mode 
# 9. Calculate distance between index and middle fingers; click allowed if distance sufficiently small

# read in webcam input and get screen dimensions
cap = cv2.VideoCapture(0)

camWidth, camHeight = 640, 480
scrWidth, scrHeight = get_monitors()[0].width, get_monitors()[0].height 
# 1920, 1080
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = htm.HandDetector(detectionConfidence=0.85, maxHands=1)
smootheningFactor = 7
prevX, prevY = 0, 0
currX, currY = 0, 0

while True:
    success, img = cap.read()
    
    img = detector.findHands(img)
    landmarks, boundingBox = detector.findPosition(img)
    
    # if hand detected, landmarks will be nonzero
    if len(landmarks) != 0:
        x1, y1 = landmarks[8][1:]
        x2, y2 = landmarks[12][1:]

        # returns a list of fingers that are open; thumb is at 0th index 
        fingers = detector.fingersOpen()
        
        # reduce virtual mousepad area, then draw
        frameR = 120
        cv2.rectangle(img, (frameR, frameR), (camWidth - frameR, camHeight - frameR), colors["purple"], 2)
        
        # when middle, ring, and pinky fingers closed
        if sum(fingers[2:]) == 0:        
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
            length, img, lineInfo = detector.calculateDistance(8, 12, img)
            # center of line connecting index and middle fingers
            cLineX, cLineY = lineInfo[4], lineInfo[5]
            if length < 16:
                cv2.circle(img, (cLineX, cLineY), 15, colors["green"], cv2.FILLED)
                mouse.click()

    cv2.imshow("img", img)
    cv2.waitKey(1)
    