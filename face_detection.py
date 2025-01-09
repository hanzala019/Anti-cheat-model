import cv2
import mediapipe as mp
import math

# Accessing full body Positions
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mp_draw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh()

# Access webcam
cap = cv2.VideoCapture(1)

# flag to initialize nose bounding box
first_pose_detected = False

while True:
    try:
        success, img = cap.read()
        if not success:
            break

        # Convert the image to RGB
        imgToRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image for pose and face landmarks
        pose_results = pose.process(imgToRGB)
        face_results = face_mesh.process(imgToRGB)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # Get landmarks for nose and shoulders
            nose = landmarks[mpPose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mpPose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER]

            # Convert to pixel
            h, w, c = img.shape
            nose_point = (int(nose.x * w), int(nose.y * h))
            left_shoulder_point = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_shoulder_point = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            midpoint_of_shoulders_point = ((left_shoulder_point[0] + right_shoulder_point[0]) // 2, (left_shoulder_point[1] + right_shoulder_point[1]) // 2)

            # Draw points and lines [commented ones not necessary since mp_draw.draw_landmarks() does their job]
            # cv2.circle(img, nose_point, 5, (255, 0, 0), 2)
            # cv2.circle(img, left_shoulder_point, 5, (0, 255, 0), 2)
            # cv2.circle(img, right_shoulder_point, 5, (0, 255, 0), 2)
            # cv2.circle(img, midpoint_of_shoulders_point, 5, (0, 0, 255), 2)
            # cv2.line(img, midpoint_of_shoulders_point, nose_point, (255,255,255), 1)

            # Calculate distances and angles
            # left_shoulder_to_nose_distance = math.sqrt((nose_point[0] - left_shoulder_point[0])**2 + (nose_point[1] - left_shoulder_point[1])**2)
            # right_shoulder_to_nose_distance = math.sqrt((nose_point[0] - right_shoulder_point[0])**2 + (nose_point[1] - right_shoulder_point[1])**2)
            # midpoint_shoulder_to_nose_distance = math.sqrt((nose_point[0] - midpoint_of_shoulders_point[0])**2 + (nose_point[1] - midpoint_of_shoulders_point[1])**2)
            midpoint_shoulder_to_shoulder_distance = math.sqrt((right_shoulder_point[0] - midpoint_of_shoulders_point[0])**2 + (right_shoulder_point[1] - midpoint_of_shoulders_point[1])**2)

            # right_angle = math.degrees(math.acos((midpoint_shoulder_to_nose_distance**2 + midpoint_shoulder_to_shoulder_distance**2 - right_shoulder_to_nose_distance**2) / (2 * midpoint_shoulder_to_nose_distance * midpoint_shoulder_to_shoulder_distance)))
            
            right_shoulder_to_nose_x = abs(right_shoulder_point[0] - nose_point[0])
            shoulder_to_shoulder_distance = midpoint_shoulder_to_shoulder_distance * 2
            percentage = (right_shoulder_to_nose_x / shoulder_to_shoulder_distance) * 100


            # create nose bounding box if first detection
            if not first_pose_detected:
                nose_restriction_box = [(nose_point[0] - int(shoulder_to_shoulder_distance * 0.8), nose_point[1] + int(shoulder_to_shoulder_distance * 0.50)),
                                        (nose_point[0] + int(shoulder_to_shoulder_distance * 0.8), nose_point[1] - int(shoulder_to_shoulder_distance * 0.50))]
                first_pose_detected = True 
            cv2.rectangle(img, nose_restriction_box[0], nose_restriction_box[1], (0, 255, 0), 1)
            
            lean_direction = ""
            if nose_point[0] < nose_restriction_box[0][0]:
                lean_direction += "right "
            if nose_point[0] > nose_restriction_box[1][0]:
                lean_direction += "left "
            if nose_point[1] > nose_restriction_box[0][1]:
                lean_direction += "forward "
            if nose_point[1] < nose_restriction_box[1][1]:
                lean_direction += "back "

            # adjustment_angle = 180 - math.degrees(math.asin( (right_shoulder_point[1]-midpoint_of_shoulders_point[1])/midpoint_shoulder_to_shoulder_distance ))
            # cv2.ellipse(img, midpoint_of_shoulders_point, (25,25), adjustment_angle, 0, right_angle, (255, 255, 255), 1)

            # Display measurements
            # cv2.putText(img, f"Left Distance: {int(left_shoulder_to_nose_distance)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # cv2.putText(img, f"Right Distance: {int(right_shoulder_to_nose_distance)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # cv2.putText(img, f"Midpoint: {int(midpoint_shoulder_to_shoulder_distance)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # cv2.putText(img, f"{int(right_angle)}", (midpoint_of_shoulders_point[0] - 10 ,midpoint_of_shoulders_point[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"{int(percentage)}%", (midpoint_of_shoulders_point[0], midpoint_of_shoulders_point[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Leaning: {str(lean_direction)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw pose landmarks
            mp_draw.draw_landmarks(img, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            intercept_point_y = int(right_shoulder_point[1] + (percentage / 100) * (left_shoulder_point[1]-right_shoulder_point[1]))
            cv2.line(img, right_shoulder_point, (nose_point[0], intercept_point_y), (0, 255, 0), 4)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Indices for upper and lower lips (approximate)
                lower_lip_index = 17  # Approximate lower lip landmark
                upper_lip_index = 13  # Approximate upper lip landmark
                
                lower_lip = face_landmarks.landmark[lower_lip_index]
                upper_lip = face_landmarks.landmark[upper_lip_index]

                lower_lip_point = (int(lower_lip.x * w), int(lower_lip.y * h))
                upper_lip_point = (int(upper_lip.x * w), int(upper_lip.y * h))

                # Draw lip points
                cv2.circle(img, lower_lip_point, 3, (0, 0, 255), -1)
                cv2.circle(img, upper_lip_point, 3, (255, 0, 255), -1)
                cv2.line(img, lower_lip_point, upper_lip_point, (255,255,255), 1)

                # Calculate lip distance
                lip_distance = math.sqrt((upper_lip_point[0] - lower_lip_point[0])**2 + (upper_lip_point[1] - lower_lip_point[1])**2)

                cv2.putText(img, f"Lip Distance: {int(lip_distance)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show the image
        cv2.imshow("Pose and Lip Movement Detection", img)

        # Break on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        print(e)

cap.release()
cv2.destroyAllWindows()