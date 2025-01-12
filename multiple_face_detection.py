import cv2
import numpy as np
from ultralytics import YOLO
import math
import mediapipe as mp
import sys, os


# DRY functions
def set_nose_bound_box(boundbox_dict, id, nose_point, shoulder_to_shoulder_distance):
    if not (id in boundbox_dict):
        boundbox_dict.update({
            id : [
                    (nose_point[0] - int(shoulder_to_shoulder_distance * 0.8), nose_point[1] + int(shoulder_to_shoulder_distance * 0.50)),
                    (nose_point[0] + int(shoulder_to_shoulder_distance * 0.8), nose_point[1] - int(shoulder_to_shoulder_distance * 0.50))
                 ]
        })


def check_nose_bound_box(id, boundbox_dict, nose_point):
    # accessing specific person's nose boundbox
    nose_restriction_box = boundbox_dict[id]

    # Draw nose bounding box
    cv2.rectangle(img, nose_restriction_box[0], nose_restriction_box[1], (0, 255, 0), 1)

    # Check if nose is out of bounding box
    lean_direction = ""
    if nose_point[0] < nose_restriction_box[0][0]:
        lean_direction += "right "
    if nose_point[0] > nose_restriction_box[1][0]:
        lean_direction += "left "
    if nose_point[1] > nose_restriction_box[0][1]:
        lean_direction += "forward "
    if nose_point[1] < nose_restriction_box[1][1]:
        lean_direction += "back "

    cv2.putText(img, f"id: {str(id)}", (nose_restriction_box[0][0], nose_restriction_box[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, f"Leaning: {str(lean_direction)}", (nose_restriction_box[0][0], nose_restriction_box[0][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# Accessing full body Positions
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mp_draw = mp.solutions.drawing_utils
model = YOLO('../yoloweigth/yolov8n.pt')
cap = cv2.VideoCapture('D:\SA_Documents\Downloads\\vid.mp4')

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float height
print(width , height)
boundbox_dict = {}
wait_delay = 500

# create a mask
# mask = np.zeros((height,width), np.uint8)

while True:
        success , img = cap.read()
        if not success:
            break
        
        results = model.track(img, stream = True, persist = True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                inty = int(box.cls[0])
                if model.names[inty] == 'person':
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    print(x1, x2, y1, y2)

                    crop_img = img[y1:y2, x1:x2].copy()
                    # masking = cv2.rectangle(mask.copy(), (x1, y1), (x2, y2), 255, -1)
                    
                    # compute the bitwise AND using the mask
                    # masked_img = cv2.bitwise_and(img, img, mask = masking)
                    cv2.imshow('cropped person', crop_img)
                    cv2.waitKey(wait_delay)
                    
                    # Convert the image to RGB
                    imgToRGB = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                    # Process the image for pose and face landmarks
                    pose_results = pose.process(imgToRGB)
                    
                    if pose_results.pose_landmarks:
                        try:

                            print("landmarks detected")
                            landmarks = pose_results.pose_landmarks.landmark

                            # Get landmarks for nose and shoulders
                            nose = landmarks[mpPose.PoseLandmark.NOSE]
                            left_shoulder = landmarks[mpPose.PoseLandmark.LEFT_SHOULDER]
                            right_shoulder = landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER]

                            # Convert to pixel
                            h, w, c = crop_img.shape
                            nose_point = (int(nose.x * w), int(nose.y * h))
                            left_shoulder_point = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                            right_shoulder_point = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                            midpoint_of_shoulders_point = ((left_shoulder_point[0] + right_shoulder_point[0]) // 2, (left_shoulder_point[1] + right_shoulder_point[1]) // 2)
                            print(nose_point, left_shoulder_point, right_shoulder_point, midpoint_of_shoulders_point)

                            # Calculate distances
                            midpoint_shoulder_to_shoulder_distance = math.sqrt((right_shoulder_point[0] - midpoint_of_shoulders_point[0])**2 + (right_shoulder_point[1] - midpoint_of_shoulders_point[1])**2)
                            right_shoulder_to_nose_x = abs(right_shoulder_point[0] - nose_point[0])
                            shoulder_to_shoulder_distance = midpoint_shoulder_to_shoulder_distance * 2

                            percentage = (right_shoulder_to_nose_x / shoulder_to_shoulder_distance) * 100

                            # setting (if needed) and checking nose bounding box
                            # set_nose_bound_box(boundbox_dict, box.id, nose_point, shoulder_to_shoulder_distance)
                            # check_nose_bound_box(box.id, boundbox_dict, nose_point)


                            # Draw pose landmarks
                            mp_draw.draw_landmarks(crop_img, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                            # Display measurements
                            cv2.putText(crop_img, f"{int(percentage)}%", (midpoint_of_shoulders_point[0], midpoint_of_shoulders_point[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            intercept_point_y = int(right_shoulder_point[1] + (percentage / 100) * (left_shoulder_point[1]-right_shoulder_point[1]))
                            cv2.line(crop_img, right_shoulder_point, (nose_point[0], intercept_point_y), (0, 255, 0), 4)
                        
                        except ValueError as e:
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                        except ZeroDivisionError as e:
                            print(e)

                    # Show the image
                    cv2.imshow("Multi-person pose detection", crop_img)
                    cv2.waitKey(wait_delay)
                else :
                    continue
                
    
cap.release()
cv2.destroyAllWindows()