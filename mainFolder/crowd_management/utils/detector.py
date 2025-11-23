import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PeopleDetector:
    def __init__(self, 
                 model_path: str,
                 camera_source: str = "rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101",
                 camera_location: str = "Main Entrance",
                 confidence_threshold: float = 0.5,
                 csv_path: str = "data/people_count_log.csv"):
        """
        Initialize people detector.
        
        Args:
            model_path: Path to YOLOv8 model
            camera_source: Camera source (RTSP URL for IP camera or device index for webcam)
            camera_location: Location identifier for the camera
            confidence_threshold: Detection confidence threshold
            csv_path: Path to save detection logs
        """
        self.model = YOLO(model_path)
        self.camera_source = camera_source
        self.camera_location = camera_location
        self.confidence_threshold = confidence_threshold
        self.csv_path = Path(csv_path)
        
        # Initialize video capture
        self.cap = None
        self.is_running = False
        self.last_detection_time = 0
        
        # Ensure CSV file exists
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file if it doesn't exist."""
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=['Timestamp', 'People_Count', 'Camera_Location'])
            df.to_csv(self.csv_path, index=False)
    
    def start_capture(self):
        """Start video capture."""
        try:
            # If camera_source is a number string (e.g., "0"), convert to int
            source = int(self.camera_source) if str(self.camera_source).isdigit() else self.camera_source
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera source: {source}")
            
            # Set buffer size for RTSP stream
            if isinstance(source, str) and source.startswith('rtsp'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size for lower latency
            
            self.is_running = True
            logger.info(f"Started capture from source: {source}")
        except Exception as e:
            logger.error(f"Error starting capture: {str(e)}")
            raise
    
    def stop_capture(self):
        """Stop video capture."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("Stopped capture")
    
    def detect_people(self, detection_interval: float = 2.0) -> tuple:
        """
        Detect people in the current frame.
        
        Args:
            detection_interval: Minimum time between detections in seconds
            
        Returns:
            tuple: (processed frame, people count)
        """
        if not self.is_running or not self.cap:
            return None, 0
        
        try:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                return None, 0
            
            current_time = time.time()
            people_count = 0
            
            # Only run detection if enough time has passed
            if current_time - self.last_detection_time >= detection_interval:
                # Run YOLOv8 inference
                results = self.model(frame, conf=self.confidence_threshold, classes=0)  # class 0 is person
                
                # Process results
                if results and len(results) > 0:
                    result = results[0]
                    people_count = len(result.boxes)
                    
                    # Draw boxes
                    annotated_frame = results[0].plot()
                    
                    # Log count
                    self._log_count(people_count)
                    
                    self.last_detection_time = current_time
                    return annotated_frame, people_count
            
            return frame, people_count
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return None, 0
    
    def _log_count(self, count: int):
        """Log people count to CSV file."""
        try:
            now = datetime.now()
            record = {
                'Timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                'People_Count': count,
                'Camera_Location': self.camera_location
            }
            
            df = pd.DataFrame([record])
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
            
        except Exception as e:
            logger.error(f"Error logging count: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture() 