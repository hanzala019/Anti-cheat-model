import cv2
import mediapipe as mp
import math

# Accessing full body Positions
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mp_draw = mp.solutions.drawing_utils

# Access webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB
    imgToRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image
    results = pose.process(imgToRGB)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get landmarks for nose and shoulders
        nose = landmarks[mpPose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mpPose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER]

        # Convert to pixel 
        h, w, c = img.shape
        nose_point = (int(nose.x * w), int(nose.y * h))
        left_shoulder_point = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_point = (int(right_shoulder.x * w), int(right_shoulder.y * h))    # // means integer division
        midpoint_of_shoulders_point = ((left_shoulder_point[0] + right_shoulder_point[0]) // 2, (left_shoulder_point[1] + right_shoulder_point[1]) // 2  )

        # Draw points
        cv2.circle(img, nose_point, 5, (255, 0, 0), -1)
        cv2.circle(img, left_shoulder_point, 5, (0, 255, 0), -1)
        cv2.circle(img, right_shoulder_point, 5, (0, 255, 0), -1)
        cv2.circle(img, midpoint_of_shoulders_point, 5, (0, 0, 255), -1)


        # Calculate distances
        left_shoulder_to_nose_distance = math.sqrt((nose_point[0] - left_shoulder_point[0])**2 +(nose_point[1] - left_shoulder_point[1])**2)

        right_shoulder_to_nose_distance = math.sqrt((nose_point[0] - right_shoulder_point[0])**2 +(nose_point[1] - right_shoulder_point[1])**2)
        
        midpoint_shoulder_to_nose_distance = math.sqrt((nose_point[0] - midpoint_of_shoulders_point[0])**2 +(nose_point[1] - midpoint_of_shoulders_point[1])**2)

        midpoint_shoulder_to_shoulder_distance = math.sqrt((right_shoulder_point[0] - midpoint_of_shoulders_point[0])**2 +(right_shoulder_point[1] - midpoint_of_shoulders_point[1])**2)

        # Calculate angles
        # left_angle = math.degrees(math.atan2(left_shoulder_point[1] - nose_point[1],
        #                                      left_shoulder_point[0] - nose_point[0]))
        # right_angle = math.degrees(math.atan2(right_shoulder_point[1] - nose_point[1],
        #                                       right_shoulder_point[0] - nose_point[0]))
        
        right_angle = math.degrees(math.acos((midpoint_shoulder_to_nose_distance**2 + midpoint_shoulder_to_shoulder_distance**2 -  right_shoulder_to_nose_distance**2)/(2*midpoint_shoulder_to_nose_distance*midpoint_shoulder_to_shoulder_distance)))

        left_angle = math.degrees(math.acos((midpoint_shoulder_to_nose_distance**2 + midpoint_shoulder_to_shoulder_distance**2 -  left_shoulder_to_nose_distance**2)/(2*midpoint_shoulder_to_nose_distance*midpoint_shoulder_to_shoulder_distance)))
        
        # Display measurements
        cv2.putText(img, f"Left Distance: {int(left_shoulder_to_nose_distance)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Right Distance: {int(right_shoulder_to_nose_distance)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Midpoint: {int(midpoint_shoulder_to_shoulder_distance)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (55, 255, 255), 2)
        cv2.putText(img, f"Left Angle: {int(left_angle)}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (155, 255, 255), 2)
        cv2.putText(img, f"Right Angle: {int(right_angle)}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw connections
        mp_draw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Show the image
    cv2.imshow("Pose Detection", img)

    # Break on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
