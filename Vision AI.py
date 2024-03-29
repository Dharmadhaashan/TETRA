import cv2
import imutils

def pre_process(frame):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)
  return gray

def detect_vehicles(frame):
  car_cascade = cv2.CascadeClassifier('cars_3.xml')
  cars = car_cascade.detectMultiScale(frame, 1.1, 3)
  return cars

video_path = 'traffic.mp4'
cap = cv2.VideoCapture(video_path)

while True:
  count = 0
  ret, frame = cap.read()
  if not ret:
    break

  gray = pre_process(frame)

  cars = detect_vehicles(gray)

  for (x, y, w, h) in cars:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    count += 1

  cv2.putText(frame, f"Vehicles: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  cv2.imshow('Frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

