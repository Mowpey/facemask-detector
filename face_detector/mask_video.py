from tensorflow import keras
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os
import pygame
import threading
import time

class MaskDetector:
    def __init__(self):
        # Initialize pygame for alarm
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound('alarm.mp3')
        self.alarm_active = False
        
        # Load face detection model
        prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        
        # Load mask detection model
        self.maskNet = keras.models.load_model("mask_detector.h5")
        
        # Initialize video stream
        self.vs = None
        self.is_running = False
        self.current_frame = None
        self.stats = {'with_mask': 0, 'without_mask': 0}
        
        # Thread for processing
        self.thread = None
    
    def play_alarm(self):
        if not self.alarm_active:
            self.alarm_active = True
            self.alarm_sound.play()
            time.sleep(3)
            self.alarm_sound.stop()
            self.alarm_active = False
    
    def detect_mask(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        
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
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = keras.utils.img_to_array(face)
                face = keras.applications.mobilenet_v2.preprocess_input(face)
                
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=32)
        
        return (locs, preds)
    
    def process_frames(self):
        while self.is_running:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=400)
            
            (locs, preds) = self.detect_mask(frame)
            
            with_mask = 0
            without_mask = 0
            no_mask_detected = False
            
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                if label == "Mask":
                    with_mask += 1
                else:
                    without_mask += 1
                    no_mask_detected = True
                
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            self.stats['with_mask'] = with_mask
            self.stats['without_mask'] = without_mask
            
            if no_mask_detected and not self.alarm_active:
                threading.Thread(target=self.play_alarm, daemon=True).start()
            
            self.current_frame = frame
    
    def start(self):
        self.vs = VideoStream(src=0).start()
        self.is_running = True
        self.thread = threading.Thread(target=self.process_frames, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.vs:
            self.vs.stop()
    
    def get_frame(self):
        return self.current_frame
    
    def get_stats(self):
        return self.stats