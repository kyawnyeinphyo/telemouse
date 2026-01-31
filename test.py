import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import urllib.request
import os


model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera. Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        exit(1)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            h, w, _ = frame.shape
            
            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            for connection in HAND_CONNECTIONS:
                start = hand_landmarks[connection[0]]
                end = hand_landmarks[connection[1]]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
            
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()