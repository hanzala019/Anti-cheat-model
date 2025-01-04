import cv2
import mediapipe as mp

# Accessing webcam
cap = cv2.VideoCapture(0)

# Accessing hands positions
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Accessing drawing functionalities
mpDraw = mp.solutions.drawing_utils

# Displaying data from webcam
while True:
    success, img = cap.read()
    if not success:
        break
    
    imgToRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting image to RGB
    handImg = hands.process(imgToRGB)
    results = handImg.multi_hand_landmarks

    if results:
        for h in results:
            # Get the bounding box
            h, w, c = img.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in h.landmark:
                x, y = int(lm.x * w), int(lm.y * h) # converting to pixels
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw hand landmarks
            mpDraw.draw_landmarks(img, h, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)  # Displaying the image
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing ESC
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



