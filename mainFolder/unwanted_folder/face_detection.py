import cv2
import numpy as np
import json
import os
import logging
from pathlib import Path
from typing import List, Dict
from mtcnn import MTCNN
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, 
                 db_path: str = "data/face_db",
                 detection_threshold: float = 0.7):
        """
        Initialize face detector.
        
        Args:
            db_path: Path to store face data
            detection_threshold: Minimum confidence for face detection
        """
        self.db_path = Path(db_path)
        self.detection_threshold = detection_threshold
        self.known_faces: Dict[str, Dict] = {}
        
        # Initialize MTCNN face detector
        try:
            self.detector = MTCNN()
            logger.info("MTCNN face detector initialized")
        except Exception as e:
            logger.error(f"Error initializing MTCNN: {str(e)}")
            raise
        
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing face database
        self._load_database()
    
    def _load_database(self):
        """Load face database from JSON files."""
        try:
            for json_file in self.db_path.glob("*.json"):
                with open(json_file, 'r') as f:
                    face_data = json.load(f)
                    self.known_faces[face_data['id']] = face_data
            logger.info(f"Loaded {len(self.known_faces)} faces from database")
        except Exception as e:
            logger.error(f"Error loading face database: {str(e)}")
    
    def _save_face_data(self, face_data: Dict):
        """Save face data to JSON file."""
        try:
            json_path = self.db_path / f"{face_data['id']}.json"
            with open(json_path, 'w') as f:
                json.dump(face_data, f, indent=2)
            logger.info(f"Saved face data for {face_data['name']}")
        except Exception as e:
            logger.error(f"Error saving face data: {str(e)}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in the frame using MTCNN.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing face detection results
        """
        try:
            # Convert frame to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using MTCNN
            detections = self.detector.detect_faces(rgb_frame)
            
            faces = []
            for detection in detections:
                if detection['confidence'] >= self.detection_threshold:
                    x, y, w, h = detection['box']
                    landmarks = detection['keypoints']
                    
                    faces.append({
                        'bbox': (x, y, w, h),
                        'confidence': detection['confidence'],
                        'landmarks': {
                            'left_eye': landmarks['left_eye'],
                            'right_eye': landmarks['right_eye'],
                            'nose': landmarks['nose'],
                            'mouth_left': landmarks['mouth_left'],
                            'mouth_right': landmarks['mouth_right']
                        }
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def add_face(self, frame: np.ndarray, name: str, person_id: str) -> bool:
        """
        Add a new face to the database.
        
        Args:
            frame: Input image frame
            name: Name of the person
            person_id: Unique identifier for the person
            
        Returns:
            bool: True if face was added successfully
        """
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            if not faces:
                logger.warning("No faces detected in the image")
                return False
            
            # Use the face with highest confidence
            face = max(faces, key=lambda x: x['confidence'])
            
            if face['confidence'] < self.detection_threshold:
                logger.warning("Face detection confidence too low")
                return False
            
            # Create face data
            face_data = {
                'name': name,
                'id': person_id,
                'landmarks': face['landmarks'],
                'added_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to database
            self._save_face_data(face_data)
            self.known_faces[person_id] = face_data
            
            logger.info(f"Added face for {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face: {str(e)}")
            return False
    
    def draw_results(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw detection results on the frame.
        
        Args:
            frame: Input image frame
            faces: List of face detection results
            
        Returns:
            Frame with visualization
        """
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks
            landmarks = face['landmarks']
            for point in landmarks.values():
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        
        return frame

def main():
    # Initialize detector
    detector = FaceDetector(
        db_path="data/face_db",
        detection_threshold=0.7
    )
    
    # Initialize camera
    cap = cv2.VideoCapture("rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101")
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = output_dir / "face_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw results
            frame = detector.draw_results(frame, faces)
            
            # Write frame to video
            out.write(frame)
            
            # Log detection results
            if faces:
                logger.info(f"Detected {len(faces)} faces")
                for face in faces:
                    logger.info(f"Face confidence: {face['confidence']:.2f}")
            
            # Check for keyboard interrupt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        logger.info(f"Video saved to {output_path}")

if __name__ == "__main__":
    main() 