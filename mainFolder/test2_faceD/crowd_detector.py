import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import pandas as pd
from pathlib import Path
import logging

class PeopleDetector:
    def __init__(self,
                 model_path: str,
                 camera_source: int,
                 camera_location: str,
                 confidence_threshold: float = 0.5,
                 csv_path: str = "data/people_count_log.csv"):
        """Initialize people detector."""
        try:
            # Load model with weights_only=True
            import torch
            torch.hub.set_dir(str(Path(model_path).parent))
            self.model = YOLO(model_path, weights_only=True)  # Add weights_only=True
            
            # Set model parameters
            self.model.conf = confidence_threshold
            self.model.iou = 0.45
            self.model.classes = [0]  # Only detect people (class 0 in COCO)
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
            
        self.camera_source = camera_source
        self.camera_location = camera_location
        self.confidence_threshold = confidence_threshold
        self.csv_path = Path(csv_path)
        
        # Initialize state
        self.cap = None
        self.last_count = 0
        self.last_detection_time = 0
        
        # Initialize CSV file
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file for logging."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=['Timestamp', 'People_Count', 'Camera_Location'])
            df.to_csv(self.csv_path, index=False)
    
    def start_capture(self):
        """Start video capture."""
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise Exception("Failed to open camera")
    
    def stop_capture(self):
        """Stop video capture."""
        if self.cap:
            self.cap.release()
    
    def detect_people(self):
        """
        Detect people in the current frame.
        
        Returns:
            frame: Processed frame with detections
        """
        if not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Run detection
        results = self.model(frame, conf=self.confidence_threshold)[0]
        
        # Count people (class 0 is 'person' in COCO)
        people_count = sum(1 for box in results.boxes if box.cls == 0)
        self.last_count = people_count
        
        # Log count periodically
        current_time = time.time()
        if current_time - self.last_detection_time >= 2:  # Log every 2 seconds
            self._log_count(people_count)
            self.last_detection_time = current_time
        
        # Draw results
        annotated_frame = results.plot()
        
        # Add count display
        cv2.putText(
            annotated_frame,
            f"People Count: {people_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return annotated_frame
    
    def _log_count(self, count: int):
        """Log people count to CSV."""
        try:
            record = {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'People_Count': count,
                'Camera_Location': self.camera_location
            }
            
            df = pd.DataFrame([record])
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
            
        except Exception as e:
            logging.error(f"Error logging count: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture() 