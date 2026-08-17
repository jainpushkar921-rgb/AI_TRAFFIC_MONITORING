from ultralytics import YOLO
import cv2
model=YOLO("yolov8n.pt")
cap=cv2.videocapture("datatraffic.mp4")
while True:
    ret,frame=cap.read()
    if not ret:
        break
    result=model(frame)
    annotated_frame=result[0].plot()
    cv2.imshow("traffic detection",annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
