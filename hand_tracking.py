import cv2
import mediapipe as mp

# Accessing webcam
cap = cv2.VideoCapture(0)


# positions on the screen. has to be tuples
# (0,0) --> top left corner
# (max_width - 1, 0) --> top right corner
# (0, max_height - 1) --> bottom left corner
# (max_width - 1, max_height - 1) --> bottom right corner 

#                   x1  y1   x2  y2
restrictedArea = [(100,100),(400,400)] 

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
        for handlms in results:
            # Get the bounding box
            h, w, c = img.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in handlms.landmark:
                x, y = int(lm.x * w), int(lm.y * h) # converting to pixels
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

                # tracking the hands position
                if x_min >= restrictedArea[0][0] and x_max <= restrictedArea[1][0] and y_min >= restrictedArea[0][1] and y_max <= restrictedArea[1][1]:
                    print("Hand is inside the box")
                else:
                    if x_min < restrictedArea[0][0]:
                        print("Gone to the left")
                    if x_max > restrictedArea[1][0]:
                        print("Gone to the right")
                    if y_min < restrictedArea[0][1]:
                        print("Gone to the top")
                    if y_max > restrictedArea[1][1]:
                        print("Gone to the bottom")

            
            
            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw hand landmarks
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    # Draw the fixed bounding box
    cv2.rectangle(img, restrictedArea[0], restrictedArea[1], (0,225,0), 1, 4)
    cv2.imshow("Image", img)  # Displaying the image

    if cv2.waitKey(100) & 0xFF == 27:  # Exit on pressing ESC
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



