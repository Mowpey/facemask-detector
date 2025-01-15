from tensorflow import keras
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os
import pygame
import threading
import time

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.mp3')
alarm_active = False

ANALYSIS_TIME = 3 
RESULT_DISPLAY_TIME = 3  
MAX_FACE_DISTANCE = 50  

mask_counter = 0
no_mask_counter = 0

def play_alarm():
    global alarm_active
    
    if not alarm_active:
        alarm_active = True
        alarm_sound.play()
        time.sleep(1.5)  
        alarm_sound.stop()
        alarm_active = False

def detect_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = keras.utils.img_to_array(face)
            face = keras.applications.mobilenet_v2.preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def get_face_center(box):
    """Calculate center point of face box"""
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

def get_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_matching_face(new_face_center, existing_faces):
    """Find the closest matching face from existing faces"""
    closest_distance = float('inf')
    closest_face_id = None
    
    for face_id, face_data in existing_faces.items():
        distance = get_distance(new_face_center, face_data['center'])
        if distance < closest_distance and distance < MAX_FACE_DISTANCE:
            closest_distance = distance
            closest_face_id = face_id
            
    return closest_face_id

prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = keras.models.load_model("mask_detector.h5")

print("[INFO] starting video stream...")
url = "http://192.168.254.100:8080/video"
vs = VideoStream(src=url).start()

face_data = {} 
next_face_id = 0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detect_mask(frame, faceNet, maskNet)
    
    no_mask_detected = False
    current_faces = set()
    
    # Reset counters for this frame
    current_mask_count = 0
    current_no_mask_count = 0

    for (box, pred) in zip(locs, preds):
        face_center = get_face_center(box)
        matching_face_id = find_matching_face(face_center, face_data)
        
        if matching_face_id is None:
            # New face detected
            matching_face_id = f"face_{next_face_id}"
            next_face_id += 1
            face_data[matching_face_id] = {
                'center': face_center,
                'start_time': time.time(),
                'status': 'analyzing',
                'last_status_change': time.time(),
                'counted': False  # Track if this face has been counted
            }
        else:
            # Update existing face position
            face_data[matching_face_id]['center'] = face_center
            
        current_faces.add(matching_face_id)
        face_info = face_data[matching_face_id]
        
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        current_time = time.time()
        time_since_last_change = current_time - face_info['last_status_change']

        # Check if we need to reset to analyzing state
        if face_info['status'] != 'analyzing' and time_since_last_change >= RESULT_DISPLAY_TIME:
            face_info['status'] = 'analyzing'
            face_info['last_status_change'] = current_time
            face_info['counted'] = False  # Reset counted flag
        
        # During analysis period
        if face_info['status'] == 'analyzing':
            time_analyzing = current_time - face_info['last_status_change']
            if time_analyzing < ANALYSIS_TIME:
                color = (255, 165, 0)  # Blue
                label = f"Analyzing... {int(ANALYSIS_TIME - time_analyzing)}s"
            else:
                # Analysis complete, set new status
                new_status = "mask" if mask > withoutMask else "no_mask"
                if not face_info['counted']:  # Only increment counter if not already counted
                    if new_status == "mask":
                        mask_counter += 1
                    else:
                        no_mask_counter += 1
                    face_info['counted'] = True
                
                face_info['status'] = new_status
                face_info['final_prediction'] = (mask, withoutMask)
                face_info['last_status_change'] = current_time
        
        # Display current status
        if face_info['status'] == "analyzing":
            color = (255, 165, 0)  # Blue
        elif face_info['status'] == "mask":
            label = f"Mask: {mask * 100:.2f}%"
            color = (0, 255, 0)  # Green
            current_mask_count += 1
        else:  # no_mask
            label = f"No Mask: {withoutMask * 100:.2f}%"
            color = (0, 0, 255)  # Red
            current_no_mask_count += 1
            no_mask_detected = True

        # Draw on frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Clean up faces that haven't been seen recently
    face_ids = list(face_data.keys())
    for face_id in face_ids:
        if face_id not in current_faces:
            del face_data[face_id]

    if no_mask_detected and not alarm_active:
        threading.Thread(target=play_alarm, daemon=True).start()
    
    # Display counters on frame
    counter_text = f"With Mask: {mask_counter} | Without Mask: {no_mask_counter}"
    cv2.putText(frame, counter_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()