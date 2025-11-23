import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import datetime
from crowd_detector import PeopleDetector
from crowd_analyzer import CrowdAnalyzer
from playsound import playsound

class CrowdManagementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Management System")
        self.root.geometry("1200x800")
        
        # Load environment variables
        load_dotenv()
        
        # Initialize paths
        self.base_path = Path(__file__).parent.parent
        self.data_dir = self.base_path / "data"
        self.models_dir = self.base_path / "models"
        self.utils_dir = self.base_path / "utils"
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.camera_source = int(os.getenv("CAMERA_SOURCE", 0))
        self.camera_location = os.getenv("CAMERA_LOCATION", "Main Entrance")
        self.detection_interval = float(os.getenv("DETECTION_INTERVAL", 2))
        self.people_count_threshold = int(os.getenv("PEOPLE_COUNT_THRESHOLD", 15))
        self.csv_path = Path(os.getenv("CSV_PATH", "data/people_count_log.csv"))
        self.model_path = self.models_dir / "yolov8n.pt"  # Use models dir from parent
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
        
        # Initialize state variables
        self.is_monitoring = False
        self.last_alert_time = 0
        self.detector = None
        self.analyzer = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        ttk.Label(self.control_frame, text="Control Panel", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Start/Stop button
        self.monitor_button = ttk.Button(self.control_frame, text="Start Monitoring", command=self.toggle_monitoring)
        self.monitor_button.pack(pady=5)
        
        # Threshold adjustment
        ttk.Label(self.control_frame, text="Overcrowding Threshold:").pack(pady=5)
        self.threshold_var = tk.StringVar(value=str(self.people_count_threshold))
        threshold_entry = ttk.Entry(self.control_frame, textvariable=self.threshold_var)
        threshold_entry.pack(pady=5)
        
        # Status indicators
        self.status_frame = ttk.LabelFrame(self.control_frame, text="Status")
        self.status_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.monitoring_status = ttk.Label(self.status_frame, text="Monitoring: Stopped")
        self.monitoring_status.pack(pady=5)
        
        self.count_status = ttk.Label(self.status_frame, text="Current Count: 0")
        self.count_status.pack(pady=5)
        
        # Video display
        self.video_label = ttk.Label(self.display_frame)
        self.video_label.pack(pady=10)
        
    def toggle_monitoring(self):
        if not self.is_monitoring:
            self.start_monitoring()
        else:
            self.stop_monitoring()
            
    def start_monitoring(self):
        try:
            # Check if model exists, if not download it
            if not self.model_path.exists():
                messagebox.showinfo("Download", "Downloading YOLOv8 model. This may take a few minutes...")
                from ultralytics import YOLO
                # Download model with weights_only
                model = YOLO('yolov8n')
                model.download(weights_only=True)
                
                # Move the downloaded model to our models directory
                import shutil
                downloaded_path = Path('yolov8n.pt')
                if downloaded_path.exists():
                    self.models_dir.mkdir(exist_ok=True)
                    shutil.move(str(downloaded_path), str(self.model_path))
            
            # Initialize detector with weights_only loading
            self.detector = PeopleDetector(
                model_path=str(self.model_path),
                camera_source=self.camera_source,
                camera_location=self.camera_location,
                confidence_threshold=self.confidence_threshold,
                csv_path=self.csv_path
            )
            self.analyzer = CrowdAnalyzer(csv_path=self.csv_path)
            
            self.detector.start_capture()
            self.is_monitoring = True
            self.monitor_button.configure(text="Stop Monitoring")
            self.monitoring_status.configure(text="Monitoring: Active")
            
            # Start update thread
            self.update_thread = threading.Thread(target=self.update_frame, daemon=True)
            self.update_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
            
    def stop_monitoring(self):
        if self.detector:
            self.detector.stop_capture()
        self.is_monitoring = False
        self.monitor_button.configure(text="Start Monitoring")
        self.monitoring_status.configure(text="Monitoring: Stopped")
        
    def update_frame(self):
        while self.is_monitoring:
            frame = self.detector.detect_people()
            if frame is not None:
                # Convert frame to PhotoImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                
                # Update video display
                self.video_label.configure(image=frame)
                self.video_label.image = frame
                
                # Update count status
                current_count = self.detector.last_count
                self.count_status.configure(text=f"Current Count: {current_count}")
                
                # Check for overcrowding
                if current_count > self.people_count_threshold:
                    self.handle_overcrowding()
                    
            time.sleep(0.03)  # Limit frame rate
            
    def handle_overcrowding(self):
        current_time = time.time()
        if current_time - self.last_alert_time >= 60:  # Alert once per minute
            self.last_alert_time = current_time
            alert_sound_path = self.utils_dir / "alert.wav"
            if alert_sound_path.exists():
                try:
                    playsound(str(alert_sound_path))
                except Exception:
                    pass
            messagebox.showwarning("Overcrowding Alert", 
                                 f"Current count ({self.detector.last_count}) exceeds threshold ({self.people_count_threshold})")
            
    def run(self):
        self.root.mainloop()
        
def launch_crowd_management():
    root = tk.Tk()
    app = CrowdManagementApp(root)
    app.run()
    
if __name__ == "__main__":
    launch_crowd_management() 