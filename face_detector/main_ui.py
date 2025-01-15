from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from tensorflow import keras
import numpy as np
import cv2
import os
import pygame
import threading
import time
from math import sqrt
from datetime import datetime

# Set window background color (#1D1A4E)
Window.clearcolor = (0.114, 0.102, 0.306, 1)

class MaskAlert(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10
        
        # Title at the top
        title_label = Label(
            text='Mask Alert',
            font_size='32sp',
            size_hint_y=0.1,
            bold=True
        )
        
        # Main content layout
        content_layout = BoxLayout(
            orientation='horizontal',
            padding=10,
            spacing=10,
            size_hint_y=0.8
        )
        
        # Analysis constants
        self.ANALYSIS_TIME = 3
        self.RESULT_DISPLAY_TIME = 3
        self.MAX_FACE_DISTANCE = 50
        
        # Left panel
        left_panel = BoxLayout(
            orientation='vertical',
            size_hint_x=0.4,
            spacing=10,
            padding=10
        )
        
        # Time and date with larger font
        self.time_label = Label(
            text=f'Time: {datetime.now().strftime("%H:%M:%S")}\n'
                 f'Date: {datetime.now().strftime("%B %d, %Y")}',
            size_hint_y=0.2,
            font_size='20sp'
        )
        Clock.schedule_interval(self.update_time, 1)
        
        # Mask counters with larger font
        self.mask_count = Label(
            text='With mask: 0',
            size_hint_y=0.1,
            font_size='20sp'
        )
        self.no_mask_count = Label(
            text='Without mask: 0',
            size_hint_y=0.1,
            font_size='20sp'
        )
        
        # Control buttons (smaller size)
        button_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.1,
            spacing=10,
            padding=[0, 10]
        )
        
        self.start_button = Button(
            text='START',
            background_color=(0, 1, 0, 1),
            size_hint=(0.5, None),
            height='40dp'
        )
        self.start_button.bind(on_press=self.start_detection)
        
        self.stop_button = Button(
            text='STOP',
            background_color=(1, 0, 0, 1),
            size_hint=(0.5, None),
            height='40dp'
        )
        self.stop_button.bind(on_press=self.stop_detection)
        
        button_layout.add_widget(self.start_button)
        button_layout.add_widget(self.stop_button)
        
        # Add widgets to left panel
        left_panel.add_widget(self.time_label)
        left_panel.add_widget(self.mask_count)
        left_panel.add_widget(self.no_mask_count)
        left_panel.add_widget(button_layout)
        
        # Right panel (video feed)
        self.image = Image(size_hint_x=0.6)
        
        # Add panels to content layout
        content_layout.add_widget(left_panel)
        content_layout.add_widget(self.image)
        
        # Footer label
        footer_label = Label(
            text='by LGTV',
            size_hint_y=0.1,
            font_size='18sp'
        )
        
        # Add all main sections to the root layout
        self.add_widget(title_label)
        self.add_widget(content_layout)
        self.add_widget(footer_label)
        
        # Initialize mask detection variables
        self.capture = None
        self.detection_active = False
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound('alarm.mp3')
        self.alarm_active = False
        
        # Load the face and mask detection models
        prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        self.maskNet = keras.models.load_model("mask_detector.h5")
        
        # Initialize counters and face tracking
        self.mask_counter = 0
        self.no_mask_counter = 0
        self.face_data = {}
        self.next_face_id = 0
        
    def update_time(self, dt):
        self.time_label.text = (f'Time: {datetime.now().strftime("%H:%M:%S")}\n'
                               f'Date: {datetime.now().strftime("%B %d, %Y")}')
    
    def start_detection(self, instance):
        if not self.detection_active:
            self.capture = cv2.VideoCapture(0)
            self.detection_active = True
            Clock.schedule_interval(self.update, 1.0/30.0)
    
    def stop_detection(self, instance):
        if self.detection_active:
            self.detection_active = False
            Clock.unschedule(self.update)
            if self.capture:
                self.capture.release()
            self.capture = None
    
    def play_alarm(self):
        if not self.alarm_active:
            self.alarm_active = True
            self.alarm_sound.play()
            time.sleep(1.5)
            self.alarm_sound.stop()
            self.alarm_active = False
    
    def get_face_center(self, box):
        """Calculate center point of face box"""
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    
    def get_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def find_matching_face(self, new_face_center, existing_faces):
        """Find the closest matching face from existing faces"""
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
    
    def update(self, dt):
        if self.capture is None:
            return
            
        ret, frame = self.capture.read()
        if ret:
            (locs, preds) = self.detect_mask(frame)
            
            no_mask_detected = False
            current_faces = set()
            
            for (box, pred) in zip(locs, preds):
                face_center = self.get_face_center(box)
                matching_face_id = self.find_matching_face(face_center, self.face_data)
                
                if matching_face_id is None:
                    # New face detected
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
                    # Update existing face position
                    self.face_data[matching_face_id]['center'] = face_center
                    
                current_faces.add(matching_face_id)
                face_info = self.face_data[matching_face_id]
                
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                
                current_time = time.time()
                time_since_last_change = current_time - face_info['last_status_change']
                
                # Check if we need to reset to analyzing state
                if face_info['status'] != 'analyzing' and time_since_last_change >= self.RESULT_DISPLAY_TIME:
                    face_info['status'] = 'analyzing'
                    face_info['last_status_change'] = current_time
                    face_info['counted'] = False
                
                # During analysis period
                if face_info['status'] == 'analyzing':
                    time_analyzing = current_time - face_info['last_status_change']
                    if time_analyzing < self.ANALYSIS_TIME:
                        color = (255, 165, 0)  # Orange
                        label = f"Analyzing... {int(self.ANALYSIS_TIME - time_analyzing)}s"
                    else:
                        # Analysis complete, set new status
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
                
                # Display current status
                if face_info['status'] == "analyzing":
                    color = (255, 165, 0)  # Orange
                elif face_info['status'] == "mask":
                    label = f"Mask: {mask * 100:.2f}%"
                    color = (0, 255, 0)  # Green
                    if not face_info['counted']:
                        self.mask_counter += 1
                        face_info['counted'] = True
                else:  # no_mask
                    label = f"No Mask: {withoutMask * 100:.2f}%"
                    color = (0, 0, 255)  # Red
                    if not face_info['counted']:
                        self.no_mask_counter += 1
                        face_info['counted'] = True
                    no_mask_detected = True
                
                # Draw on frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Clean up faces that haven't been seen recently
            face_ids = list(self.face_data.keys())
            for face_id in face_ids:
                if face_id not in current_faces:
                    del self.face_data[face_id]
            
            # Update counter labels
            self.mask_count.text = f'With mask: {self.mask_counter}'
            self.no_mask_count.text = f'Without mask: {self.no_mask_counter}'
            
            # Play alarm if needed
            if no_mask_detected and not self.alarm_active:
                threading.Thread(target=self.play_alarm, daemon=True).start()
            
            # Convert to texture
            buf = cv2.flip(frame, 0)
            buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            self.image.texture = texture

class MaskAlertApp(App):
    def build(self):
        return MaskAlert()

if __name__ == '__main__':
    MaskAlertApp().run()