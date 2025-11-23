import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import datetime

class PoseDataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Data Collector")
        self.root.state('zoomed')  # Maximize window
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize variables
        self.camera_source = "rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101"  # IP camera
        self.is_collecting = False
        self.current_pose = "Standing"
        self.collected_data = []
        self.frame_size = (800, 600)  # Default frame size
        
        # Available poses
        self.poses = ["Standing", "Sitting", "Raise Hand", "Fold Hand", "Clapping"]
        
        # Setup UI
        self.setup_ui()
        
        # Start camera
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            self.status_label.config(text="Error: Failed to connect to IP camera")
        else:
            self.status_label.config(text="Connected to IP camera")
            # Get actual frame size from camera
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.frame_size = (int(width), int(height))
        
        self.update_frame()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create main container with padding
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for video
        self.left_panel = ttk.Frame(self.main_container)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right panel for controls
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Video frame
        self.video_frame = ttk.LabelFrame(self.left_panel, text="Camera Feed")
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(self.right_panel, text="Controls")
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Pose selection
        pose_frame = ttk.Frame(controls_frame)
        pose_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pose_frame, text="Current Pose:").pack(pady=5)
        self.pose_var = tk.StringVar(value=self.current_pose)
        pose_menu = ttk.OptionMenu(pose_frame, self.pose_var, self.current_pose, *self.poses)
        pose_menu.pack(fill=tk.X, padx=5)
        
        # Collection controls
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.collect_btn = ttk.Button(btn_frame, text="Start Collecting", 
                                    command=self.toggle_collection)
        self.collect_btn.pack(fill=tk.X, padx=5, pady=2)
        
        save_btn = ttk.Button(btn_frame, text="Save Data", 
                             command=self.save_data)
        save_btn.pack(fill=tk.X, padx=5, pady=2)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.right_panel, text="Status")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready", wraplength=200)
        self.status_label.pack(padx=5, pady=5)
    
    def extract_pose_features(self, landmarks):
        """Extract relevant features from pose landmarks."""
        points = np.zeros((33, 3))
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = [landmark.x, landmark.y, landmark.z]
        
        # Calculate angles and features
        def calculate_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return np.degrees(angle)
        
        # Extract key points
        left_shoulder = points[11]
        right_shoulder = points[12]
        left_elbow = points[13]
        right_elbow = points[14]
        left_wrist = points[15]
        right_wrist = points[16]
        left_hip = points[23]
        right_hip = points[24]
        
        # Calculate features
        features = [
            calculate_angle(left_shoulder, left_elbow, left_wrist),  # Left elbow angle
            calculate_angle(right_shoulder, right_elbow, right_wrist),  # Right elbow angle
            calculate_angle(left_shoulder, right_shoulder, 
                          [right_shoulder[0], right_shoulder[1] + 1, right_shoulder[2]]),  # Shoulder angle
            calculate_angle(left_hip, right_hip, 
                          [right_hip[0], right_hip[1] + 1, right_hip[2]]),  # Hip angle
            left_wrist[1] - left_shoulder[1],  # Left hand raise
            right_wrist[1] - right_shoulder[1],  # Right hand raise
            (left_hip[1] + right_hip[1])/2 - (left_shoulder[1] + right_shoulder[1])/2,  # Torso angle
        ]
        
        return features
    
    def toggle_collection(self):
        """Toggle data collection state."""
        self.is_collecting = not self.is_collecting
        if self.is_collecting:
            self.collect_btn.config(text="Stop Collecting")
            self.status_label.config(text=f"Collecting data for {self.pose_var.get()}")
        else:
            self.collect_btn.config(text="Start Collecting")
            self.status_label.config(text="Ready")
    
    def update_frame(self):
        """Update video frame and collect data if active."""
        ret, frame = self.cap.read()
        if ret:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(frame_rgb)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Collect data if active
                if self.is_collecting:
                    features = self.extract_pose_features(results.pose_landmarks)
                    self.collected_data.append({
                        'features': features,
                        'pose': self.pose_var.get()
                    })
                    self.status_label.config(
                        text=f"Collected {len(self.collected_data)} samples for {self.pose_var.get()}"
                    )
            
            # Resize frame to fit the window while maintaining aspect ratio
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            aspect_ratio = w/h
            
            # Calculate new size maintaining aspect ratio
            window_width = self.video_frame.winfo_width()
            window_height = self.video_frame.winfo_height()
            if window_width > 0 and window_height > 0:
                if window_width/window_height > aspect_ratio:
                    new_height = window_height
                    new_width = int(window_height * aspect_ratio)
                else:
                    new_width = window_width
                    new_height = int(window_width / aspect_ratio)
                
                if new_width > 0 and new_height > 0:
                    frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PhotoImage and display
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(30, self.update_frame)
    
    def save_data(self):
        """Save collected data to file."""
        if not self.collected_data:
            self.status_label.config(text="No data to save")
            return
        
        try:
            # Create organized directory structure
            data_dir = Path("data/pose_training")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_data_{timestamp}.json"
            
            # Save data with metadata
            data = {
                'metadata': {
                    'timestamp': timestamp,
                    'total_samples': len(self.collected_data),
                    'poses': list(set(d['pose'] for d in self.collected_data)),
                    'samples_per_pose': {
                        pose: len([d for d in self.collected_data if d['pose'] == pose])
                        for pose in self.poses
                    }
                },
                'features': [d['features'] for d in self.collected_data],
                'poses': [d['pose'] for d in self.collected_data]
            }
            
            # Save the file
            save_path = data_dir / filename
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            # Update status with save location
            self.status_label.config(
                text=f"Saved {len(self.collected_data)} samples to:\n{save_path}"
            )
            
        except Exception as e:
            self.status_label.config(text=f"Error saving data: {str(e)}")
    
    def on_closing(self):
        """Handle window closing."""
        if self.cap:
            self.cap.release()
        self.root.quit()

def main():
    root = tk.Tk()
    app = PoseDataCollector(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 