import cv2
import mediapipe as mp

# accessing webcam
cap = cv2.VideoCapture(0)

# accessing hands positions
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# accessing drawing functionalities
mpDraw = mp.solutions.drawing_utils

# displaying data from webcam
while True:
    success, img = cap.read()
    imgToRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting image to rgb
    handImg = hands.process(imgToRGB)
    results = handImg.multi_hand_landmarks
    
    if results is not None:
        for h in results:
            mpDraw.draw_landmarks(img, h) # drawing points on the image

    cv2.imshow("Image", img) # displaying the image
    cv2.waitKey(1)