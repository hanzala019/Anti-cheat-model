import cv2
import mediapipe as mp
import numpy as np

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Access webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        h, w, c = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Get the relevant points (scaled to pixel coordinates)
        nose = (int(landmarks[mp_pose.PoseLandmark.NOSE].x * w), int(landmarks[mp_pose.PoseLandmark.NOSE].y * h))
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))

        # Calculate midpoint of shoulders
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2)

        # Draw points and lines
        cv2.circle(frame, nose, 5, (0, 0, 255), -1)
        cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_shoulder, 5, (0, 255, 0), -1)
        cv2.circle(frame, midpoint, 5, (255, 0, 0), -1)

        cv2.line(frame, nose, midpoint, (255, 255, 0), 2)
        cv2.line(frame, midpoint, left_shoulder, (255, 255, 0), 2)
        cv2.line(frame, midpoint, right_shoulder, (255, 255, 0), 2)

        # Calculate vectors
        nose_to_mid = np.array([midpoint[0] - nose[0], midpoint[1] - nose[1]])
        mid_to_left = np.array([left_shoulder[0] - midpoint[0], left_shoulder[1] - midpoint[1]])
        mid_to_right = np.array([right_shoulder[0] - midpoint[0], right_shoulder[1] - midpoint[1]])

        # Calculate angles using dot product
        def calculate_angle(vec1, vec2):
            unit_vec1 = vec1 / np.linalg.norm(vec1)
            unit_vec2 = vec2 / np.linalg.norm(vec2)
            dot_product = np.dot(unit_vec1, unit_vec2)
            return np.degrees(np.arccos(dot_product))

        angle_nose_left = calculate_angle(nose_to_mid, mid_to_left)
        angle_nose_right = calculate_angle(nose_to_mid, mid_to_right)

        # Display angles
        cv2.putText(frame, f"Left Angle: {int(angle_nose_left)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 25, 255), 2)
        cv2.putText(frame, f"Right Angle: {int(angle_nose_right)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 55, 255), 2)

    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
