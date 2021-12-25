import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(
        self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complexity = complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # to draw points and connections on hand
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNumber=0, draw=True):
        # landmark list
        lmlist = []
        # below line would get a particular hand and landmarks on that hand only

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            # to get index for each point in hand
            for id, lm in enumerate(myHand.landmark):
                # to convert point on X-Y plane into pixels by multiplying wiht height and width of screen
                height, width, channels = img.shape
                # cx, cy points on the plane X and Y
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmlist


def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    currTime = 0
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # to calculate FPS
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # to display fps, convert into string, position, font, scale, color, height
        cv2.putText(
            img,
            str(int(fps)),
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),
            3,
        )

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
