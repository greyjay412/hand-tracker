import cv2
import numpy as np
import os
import pyautogui
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as mp_Image
from mediapipe import ImageFormat
from mediapipe.tasks.python.vision import HandLandmarksConnections

def download_model_if_needed():
    """Download the hand landmarker model if not present"""
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        try:
            import urllib.request
            import ssl
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            print(f"Downloading hand landmarker model to {model_path}...")
            # Create SSL context that doesn't verify certificates (for macOS SSL issues)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(model_url, context=ssl_context) as response, open(model_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download the model manually from:")
            print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            print(f"And save it as {model_path}")
            raise
    return model_path

palmIDs = [0, 5, 9, 13, 17]

screenW, screenH = pyautogui.size()

deadZone = 100
pinchStart = 30

def run_hand_tracking_on_webcam():
    prevX, prevY = screenW//2, screenH//2
    model_path = download_model_if_needed()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    detector = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame...")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp_Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands
        detection_result = detector.detect(mp_image)
        
        # Draw the hand annotations on the image
        if detection_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in detection_result.hand_landmarks:

                indexTip = hand_landmarks[8]
                thumbTip = hand_landmarks[4]

                # Draw landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                #control cursor
                cursX = sum(hand_landmarks[id].x for id in palmIDs) / len(palmIDs)
                cursY = sum(hand_landmarks[id].y for id in palmIDs) / len(palmIDs)

                #map to screen
                screenX = int(screenW-(cursX*screenW))
                screenY = int(cursY*screenH)

                dx = screenX-prevX
                dy = screenY-prevY

                if math.hypot(dx, dy) > deadZone: #not sure if this solved the shakiness but im leaving it here cause Im too lazy to change the variables back
                    pyautogui.moveTo(screenX,screenY,duration=0)

                cv2.circle(frame, (int(cursX*w),int(cursY*h)), 5, (0,255,0), -1)  
                
                # Draw connections
                connections = HandLandmarksConnections.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection.start
                    end_idx = connection.end
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start_point = (int(hand_landmarks[start_idx].x * w), 
                                      int(hand_landmarks[start_idx].y * h))
                        end_point = (int(hand_landmarks[end_idx].x * w), 
                                    int(hand_landmarks[end_idx].y * h))
                        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

            #pinch select
            indexX, indexY = int(indexTip.x*w), int(indexTip.y*h)
            thumbX, thumbY =  int(thumbTip.x*w), int(thumbTip.y*h)

            indexPinchDist = math.hypot(indexX-thumbX, indexY-thumbY)

            if indexPinchDist < pinchStart:
                pyautogui.mouseDown(button="left")
            elif indexPinchDist > pinchStart:
                pyautogui.mouseUp(button="left")
                
                

        cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hand_tracking_on_webcam()
