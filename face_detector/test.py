import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
from tensorflow import keras
import pygame
import threading
import time
from math import sqrt
from datetime import datetime
import os

class MaskDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Mask Alert")
        self.window.state('zoomed') 
        
  
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        self.style = ttk.Style()
        self.style.configure('Stats.TFrame', background='#29254d')
        self.style.configure('Header.TLabel', 
                           background='#332f59',
                           foreground='white',
                           font=('Arial', 12, 'bold'),
                           padding=5)
        self.style.configure('Value.TLabel',
                           background='#2d2a4d',
                           foreground='white',
                           font=('Arial', 24),
                           padding=5)
        
        
        self.stats_frame = ttk.Frame(window, style='Stats.TFrame')
        self.stats_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.video_frame = ttk.Frame(window)
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
       
        self.detection_active = False
        self.session_start_time = None
        self.mask_counter = 0
        self.no_mask_counter = 0
        self.face_data = {}
        self.next_face_id = 0
        self.alarm_active = False
        
       
        self.ANALYSIS_TIME = 3
        self.RESULT_DISPLAY_TIME = 3
        self.MAX_FACE_DISTANCE = 50
        
       
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound('alarm.mp3')
        
       
        self.load_models()
        
    
        self.create_stats_panel()
        self.create_video_panel()
        self.create_control_buttons()
    
        self.update_time()
    
    def load_models(self):
        prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        self.maskNet = keras.models.load_model("mask_detector.h5")
    
    def create_stats_panel(self):
        
        stats = [
            ('Total Detections', '0'),
            ('With Mask', '0'),
            ('Without Mask', '0'),
            ('Compliance Rate', '0%'),
            ('Active Faces', '0'),
            ('Session Duration', '00:00')
        ]
        
        self.stat_labels = {}
        for i, (name, initial_value) in enumerate(stats):
            header = ttk.Label(self.stats_frame, text=name, style='Header.TLabel')
            header.grid(row=i, column=0, sticky="ew", padx=1, pady=1)
            
            value = ttk.Label(self.stats_frame, text=initial_value, style='Value.TLabel')
            value.grid(row=i, column=1, sticky="ew", padx=1, pady=1)
            
            self.stat_labels[name] = value
        
        self.time_label = ttk.Label(self.stats_frame, style='Header.TLabel')
        self.time_label.grid(row=len(stats), column=0, columnspan=2, sticky="ew", pady=10)
    
    def create_video_panel(self):
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill='both')
    
    def create_control_buttons(self):
        button_frame = ttk.Frame(self.stats_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="START", command=self.start_detection)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="STOP", command=self.stop_detection)
        self.stop_button.pack(side='left', padx=5)
    
    def update_time(self):
        current_time = datetime.now()
        self.time_label.config(text=f'Time: {current_time.strftime("%H:%M:%S")}\n'
                                  f'Date: {current_time.strftime("%B %d, %Y")}')
        self.window.after(1000, self.update_time)
    
    def start_detection(self):
        if not self.detection_active:
            url = "http://192.168.254.100:8080/video"
            self.capture = cv2.VideoCapture(0)
            self.detection_active = True
            self.session_start_time = time.time()
            self.update_video()
            self.update_session_duration()
    
    def stop_detection(self):
        if self.detection_active:
            self.detection_active = False
            if self.capture:
                self.capture.release()
            self.capture = None
            self.session_start_time = None
    
    def play_alarm(self):
        if not self.alarm_active:
            self.alarm_active = True
            self.alarm_sound.play()
            time.sleep(1.5)
            self.alarm_sound.stop()
            self.alarm_active = False
    
    def get_face_center(self, box):
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    
    def get_distance(self, p1, p2):
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def find_matching_face(self, new_face_center, existing_faces):
        closest_distance = float('inf')
        closest_face_id = None
        
        for face_id, face_data in existing_faces.items():
            distance = self.get_distance(new_face_center, face_data['center'])
            if distance < closest_distance and distance < self.MAX_FACE_DISTANCE:
                closest_distance = distance
                closest_face_id = face_id
        
        return closest_face_id
    
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
            preds = self.maskNet.predict(faces, batch_size=32)
        
        return (locs, preds)
    
    def update_session_duration(self):
        if self.session_start_time and self.detection_active:
            duration = int(time.time() - self.session_start_time)
            minutes = duration // 60
            seconds = duration % 60
            self.stat_labels['Session Duration'].config(text=f'{minutes:02d}:{seconds:02d}')
            self.window.after(1000, self.update_session_duration)
    
    def update_video(self):
        if self.detection_active and self.capture:
            ret, frame = self.capture.read()
            if ret:
                (locs, preds) = self.detect_mask(frame)
                
                no_mask_detected = False
                current_faces = set()
                
                for (box, pred) in zip(locs, preds):
                    face_center = self.get_face_center(box)
                    matching_face_id = self.find_matching_face(face_center, self.face_data)
                    
                    if matching_face_id is None:
                        matching_face_id = f"face_{self.next_face_id}"
                        self.next_face_id += 1
                        self.face_data[matching_face_id] = {
                            'center': face_center,
                            'start_time': time.time(),
                            'status': 'analyzing',
                            'last_status_change': time.time(),
                            'counted': False
                        }
                    else:
                        self.face_data[matching_face_id]['center'] = face_center
                    
                    current_faces.add(matching_face_id)
                    self.process_face(frame, box, pred, matching_face_id)
                    
                
                face_ids = list(self.face_data.keys())
                for face_id in face_ids:
                    if face_id not in current_faces:
                        del self.face_data[face_id]
                
          
                self.update_statistics(len(current_faces))
              
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(frame_rgb)
                imgtk = PIL.ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            if self.detection_active:
                self.window.after(30, self.update_video)
    
    def process_face(self, frame, box, pred, face_id):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        face_info = self.face_data[face_id]
        
        current_time = time.time()
        time_since_last_change = current_time - face_info['last_status_change']
        

        if face_info['status'] != 'analyzing' and time_since_last_change >= self.RESULT_DISPLAY_TIME:
            face_info['status'] = 'analyzing'
            face_info['last_status_change'] = current_time
            face_info['counted'] = False
        
        
        if face_info['status'] == 'analyzing':
            
            time_analyzing = current_time - face_info['last_status_change']
            if time_analyzing < self.ANALYSIS_TIME:
               
                color = (255, 165, 0)  
                label = f"Analyzing... {int(self.ANALYSIS_TIME - time_analyzing)}s"
            else:
               
                new_status = "mask" if mask > withoutMask else "no_mask"
                if not face_info['counted']:
                    if new_status == "mask":
                        self.mask_counter += 1
                    else:
                        self.no_mask_counter += 1
                    face_info['counted'] = True
                
                face_info['status'] = new_status
                face_info['final_prediction'] = (mask, withoutMask)
                face_info['last_status_change'] = current_time
                
                
                if new_status == "mask":
                    label = f"Mask: {mask * 100:.2f}%"
                    color = (0, 255, 0)
                else:
                    label = f"No Mask: {withoutMask * 100:.2f}%"
                    color = (0, 0, 255)
                    if not self.alarm_active:
                        threading.Thread(target=self.play_alarm, daemon=True).start()
        else:
            
            if face_info['status'] == "mask":
                label = f"Mask: {mask * 100:.2f}%"
                color = (0, 255, 0)
            else:
                label = f"No Mask: {withoutMask * 100:.2f}%"
                color = (0, 0, 255)
                if not self.alarm_active:
                    threading.Thread(target=self.play_alarm, daemon=True).start()
        
      
        cv2.putText(frame, label, (startX, startY - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    def update_statistics(self, active_faces):
        total_detections = self.mask_counter + self.no_mask_counter
        compliance_rate = (self.mask_counter / total_detections * 100) if total_detections > 0 else 0
        
        self.stat_labels['Total Detections'].config(text=str(total_detections))
        self.stat_labels['With Mask'].config(text=str(self.mask_counter))
        self.stat_labels['Without Mask'].config(text=str(self.no_mask_counter))
        self.stat_labels['Compliance Rate'].config(text=f'{compliance_rate:.1f}%')
        self.stat_labels['Active Faces'].config(text=str(active_faces))

def main():
    root = tk.Tk()
    app = MaskDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()