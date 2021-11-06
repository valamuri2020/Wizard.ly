import time
import cv2
import mediapipe as mp
import math

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, 
                                         min_detection_confidence=self.detectionConfidence, 
                                         min_tracking_confidence=self.trackConfidence)      
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.fingerTips = [4,8,12,16,20]
        

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, landmark_list=handLms, connections=self.mpHands.HAND_CONNECTIONS)   

        return img

    def findPosition(self, img, handNumber=0, draw=True):
        xList = []
        yList = []
        boundingBox = []
        self.landmarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, landmark in enumerate(myHand.landmark):
                h, w, c = img.shape
                # converting position ratios to pixel values
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                
                xList.append(cx)
                yList.append(cy)
                self.landmarkList.append([id, cx, cy])
                
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            boundingBox = xMin, yMin, xMax, yMax
            
            if draw:
                cv2.circle(img, (cx,cy), 7, (255, 0, 255), cv2.FILLED)
                cv2.rectangle(img, (xMin -20, yMin - 20), (xMax + 20, yMax + 20), (0, 255, 0), 2)
                
        return self.landmarkList, boundingBox

    def fingersOpen(self):
        fingers = []
        # Thumb
        if self.landmarkList[self.fingerTips[0]][1] > self.landmarkList[self.fingerTips[0] - 1][1]: 
            fingers.append(1)
        else: fingers.append(0)
    
        for id in range(1,5):
            if self.landmarkList[self.fingerTips[id]][2] < self.landmarkList[self.fingerTips[id] - 2][2]:
                fingers.append(1)
            else : fingers.append(0)
        
        return fingers
    
    def calculateDistance(self, p1, p2, img, draw=True, circle_r=15, thickness=3):
        x1, y1 = self.landmarkList[p1][1:]
        x2, y2 = self.landmarkList[p2][1:]
        cx, cy = (x1+x2) // 2, (y1+y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), thickness)
            cv2.circle(img, (x1, y1), circle_r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), circle_r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), circle_r, (255, 0, 255), cv2.FILLED)
        
        return length, img, [x1, y1, x2, y2, cx, cy]
    
def main():
    prevTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img=img)
        # lmList[index], index is the hand landmark number. 4 is tip of thumb
        lmList, _ = detector.findPosition(img)
        if len(lmList) != 0: 
            print(lmList[4])
            fingers = detector.fingersOpen()
            print(fingers)
        # fps calculation
        currentTime = time.time()
        fps = 1/(currentTime - prevTime)
        prevTime = currentTime
        # cast to int to get whole numbers for fps
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv2.imshow("Video Camera", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()