
import cv2 

cap = cv2.VideoCapture('imgandvid/dancecrop.mp4')
from ultralytics import NAS

# Load a COCO-pretrained YOLO-NAS-s model
model = NAS("yolo_nas_s.pt")

# Display model information (optional)
model.info()

# Validate the model on the COCO8 example dataset
results = model.val(data="coco8.yaml")

# Run inference with the YOLO-NAS-s model on the 'bus.jpg' image
results = model(cap)
cv2.imshow('negro',results)
cv2.waitKey(10)

'''
while True:
    success , img = cap.read()
    
    if not success:
        break
    results = model(img,stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cvzone.putTextRect(img,f'{conf}',(x1,y1))
    cv2.imshow('yolo',img)
    if cv2.waitKey(10) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()'''

