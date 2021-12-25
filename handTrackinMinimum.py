import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # to get index for each point in hand
            for id, lm in enumerate(handLms.landmark):
                # to convert point on X-Y plane into pixels by multiplying wiht height and width of screen
                height, width, channels = img.shape
                # cx, cy points on the plane X and Y
                cx, cy = int(lm.x * width), int(lm.y * height)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            # to draw points and connections on hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

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
