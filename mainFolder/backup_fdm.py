import cv2
import numpy as np
from mtcnn import MTCNN
import logging
from typing import Tuple, List, Dict
import tensorflow as tf
import json
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalFaceDetector:
    def __init__(self, 
                 detection_threshold: float = 0.9,
                 target_size: Tuple[int, int] = (640, 480),
                 db_path: str = "data/face_db"):
        """
        Initialize minimal face detector.
        
        Args:
            detection_threshold: Minimum confidence for face detection
            target_size: Target frame size for processing (width, height)
            db_path: Path to store face embeddings
        """
        self.detection_threshold = detection_threshold
        self.target_size = target_size
        self.db_path = Path(db_path)
        self.frame_count = 0
        self.embedding_interval = 5  # Capture embeddings every 5 frames
        
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Configure GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU acceleration enabled")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {str(e)}")
        else:
            logger.warning("No GPU found, using CPU")
        
        # Initialize MTCNN
        try:
            self.detector = MTCNN()
            logger.info("MTCNN detector initialized")
        except Exception as e:
            logger.error(f"Error initializing MTCNN: {str(e)}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for detection.
        
        Args:
            frame: Input BGR frame from OpenCV
            
        Returns:
            Preprocessed frame
        """
        # Resize frame
        frame = cv2.resize(frame, self.target_size)
        
        # Convert BGR to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame_rgb
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in preprocessed frame.
        
        Args:
            frame: Preprocessed RGB frame
            
        Returns:
            List of face detections with bounding boxes and landmarks
        """
        # Run detection
        detections = self.detector.detect_faces(frame)
        
        # Filter by confidence
        faces = []
        for detection in detections:
            if detection['confidence'] >= self.detection_threshold:
                faces.append({
                    'bbox': detection['box'],
                    'confidence': detection['confidence'],
                    'landmarks': detection['keypoints']
                })
        
        return faces
    
    def draw_detections(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Input frame
            faces: List of face detections
            
        Returns:
            Frame with visualizations
        """
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(frame, conf_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks
            landmarks = face['landmarks']
            for point in landmarks.values():
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        
        return frame
    
    def save_face_data(self, face_data: Dict):
        """Save face data to JSON file."""
        try:
            json_path = self.db_path / f"{face_data['id']}.json"
            with open(json_path, 'w') as f:
                json.dump(face_data, f, indent=2)
            logger.info(f"Saved face data for {face_data['name']}")
        except Exception as e:
            logger.error(f"Error saving face data: {str(e)}")
    
    def add_face(self, frame: np.ndarray, name: str, person_id: str) -> bool:
        """
        Add a new face to the database.
        
        Args:
            frame: Input frame
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
            self.save_face_data(face_data)
            
            logger.info(f"Added face for {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face: {str(e)}")
            return False

def main():
    # Initialize detector
    detector = MinimalFaceDetector(
        detection_threshold=0.9,
        target_size=(640, 480),
        db_path="data/face_db"
    )
    
    # Initialize camera
    cap = cv2.VideoCapture("rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101")
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Press 'a' to add a new face, 'q' to quit")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Preprocess frame
            processed_frame = detector.preprocess_frame(frame)
            
            # Detect faces
            faces = detector.detect_faces(processed_frame)
            
            # Draw detections
            frame = detector.draw_detections(frame, faces)
            
            # Show frame
            cv2.imshow("Face Detection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                if faces:
                    # Get name and ID from user
                    name = input("Enter name: ")
                    person_id = input("Enter ID: ")
                    
                    if name and person_id:
                        detector.add_face(frame, name, person_id)
            
            # Log results
            if faces:
                logger.info(f"Detected {len(faces)} faces:")
                for i, face in enumerate(faces, 1):
                    logger.info(f"Face {i}: confidence={face['confidence']:.2f}")
                    logger.info(f"  Bounding box: {face['bbox']}")
                    logger.info(f"  Landmarks: {face['landmarks']}")
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 