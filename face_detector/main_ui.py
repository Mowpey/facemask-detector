import tkinter as tk
from tkinter import ttk
from datetime import datetime
import cv2
from PIL import Image, ImageTk
import threading
from mask_video import MaskDetector

class MaskAlert:
    def __init__(self, root):
        self.root = root
        self.root.title("MaskAlert")
        
        self.detector = MaskDetector()
        self.is_running = False
        
        self.with_mask_count = tk.StringVar(value="0")
        self.without_mask_count = tk.StringVar(value="0")
        
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.left_panel = ttk.Frame(self.main_frame, relief="groove", padding="5")
        self.left_panel.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.time_var = tk.StringVar()
        self.date_var = tk.StringVar()
        self.update_datetime()
        
        ttk.Label(self.left_panel, textvariable=self.time_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(self.left_panel, textvariable=self.date_var).grid(row=1, column=0, sticky=tk.W)
        
        ttk.Label(self.left_panel, text="With mask:").grid(row=2, column=0, sticky=tk.W, pady=(10,0))
        ttk.Label(self.left_panel, textvariable=self.with_mask_count).grid(row=2, column=1, sticky=tk.W, pady=(10,0))
        
        ttk.Label(self.left_panel, text="Without mask:").grid(row=3, column=0, sticky=tk.W)
        ttk.Label(self.left_panel, textvariable=self.without_mask_count).grid(row=3, column=1, sticky=tk.W)
        
        self.button_frame = ttk.Frame(self.left_panel)
        self.button_frame.grid(row=4, column=0, columnspan=2, pady=(10,0))
        
        self.start_button = ttk.Button(self.button_frame, text="START", style="Green.TButton", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=2)
        
        self.stop_button = ttk.Button(self.button_frame, text="STOP", style="Red.TButton", command=self.stop_detection)
        self.stop_button.grid(row=0, column=1, padx=2)
        
        self.right_panel = ttk.Label(self.main_frame)
        self.right_panel.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
      
        self.configure_styles()
        self.update_clock()
    
    def configure_styles(self):
        style = ttk.Style()
        style.configure("Green.TButton", background="green")
        style.configure("Red.TButton", background="red")
    
    def update_datetime(self):
        now = datetime.now()
        self.time_var.set(f"Time: {now.strftime('%H:%M:%S')}")
        self.date_var.set(f"Date: {now.strftime('%B %d, %Y')}")
    
    def update_clock(self):
        self.update_datetime()
        self.root.after(1000, self.update_clock)
    
    def update_frame(self):
        if self.is_running:
            frame = self.detector.get_frame()
            if frame is not None:

                stats = self.detector.get_stats()
                self.with_mask_count.set(str(stats['with_mask']))
                self.without_mask_count.set(str(stats['without_mask']))
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(image=frame)
                
                self.right_panel.imgtk = frame
                self.right_panel.configure(image=frame)
            
            self.root.after(10, self.update_frame)
    
    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self.detector.start()
            self.update_frame()
            self.start_button.state(['disabled'])
            self.stop_button.state(['!disabled'])
    
    def stop_detection(self):
        if self.is_running:
            self.is_running = False
            self.detector.stop()
            self.start_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
    
    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = MaskAlert(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()