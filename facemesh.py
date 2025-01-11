import cv2
import mediapipe as pp
import time
import mediapipe.python.solutions.face_mesh as fc
import mediapipe.python.solutions.drawing_utils as dc

cap = cv2.VideoCapture('imgandvid/face.jpg')
mpDraw = dc
mpFacemesh = fc
facemesh = mpFacemesh.FaceMesh(max_num_faces= 3)
ptime=0
while True:
    success, img = cap.read()
    if not success:
        break
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = facemesh.process(image)
    if results.multi_face_landmarks:
        for res in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,res,mpFacemesh.FACEMESH_CONTOURS)
            
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f'FPS {int(fps)}',(20,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,230,23),3)
    cv2.imshow('yoyo',img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
