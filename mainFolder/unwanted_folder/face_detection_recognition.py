import cv2
import numpy as np
import torch
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetectorRecognizer:
    def __init__(self, 
                 db_path: str = "data/face_db",
                 model_name: str = "buffalo_l",
                 detection_threshold: float = 0.5,
                 recognition_threshold: float = 0.5):
        """
        Initialize face detector and recognizer with GPU acceleration.
        
        Args:
            db_path: Path to store face embeddings database
            model_name: Name of the InsightFace model to use
            detection_threshold: Minimum confidence for face detection
            recognition_threshold: Minimum similarity threshold for face recognition
        """
        self.db_path = Path(db_path)
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.known_faces: Dict[str, Dict] = {}
        
        # Check CUDA availability
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            logger.info("CUDA is available, using GPU acceleration")
            torch.cuda.set_device(0)
        else:
            logger.warning("CUDA is not available, falling back to CPU")
        
        # Initialize YOLO face detector
        try:
            self.detector = YOLO('yolov8n.pt')
            logger.info("YOLO face detector initialized")
        except Exception as e:
            logger.error(f"Error initializing YOLO: {str(e)}")
            raise
        
        # Initialize InsightFace
        try:
            self.face_analyzer = FaceAnalysis(
                name=model_name,
                root='models',
                providers=['CUDAExecutionProvider' if self.use_cuda else 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0 if self.use_cuda else -1)
            logger.info(f"InsightFace initialized with {model_name}")
        except Exception as e:
            logger.error(f"Error initializing InsightFace: {str(e)}")
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
        Detect faces in the frame using YOLO.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing face detection results
        """
        try:
            # Run YOLO detection
            results = self.detector(frame, classes=0)  # class 0 is person
            
            faces = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.conf[0] >= self.detection_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Extract face region
                        face_img = frame[y1:y2, x1:x2]
                        
                        # Get face landmarks and alignment
                        face_analysis = self.face_analyzer.get(face_img)
                        if face_analysis:
                            face = face_analysis[0]
                            faces.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                                'embedding': face.embedding.tolist() if hasattr(face, 'embedding') else None
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
                'embedding': face['embedding'],
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
    
    def recognize_face(self, face_embedding: List[float]) -> Tuple[Optional[str], float]:
        """
        Recognize a face using its embedding.
        
        Args:
            face_embedding: Face embedding vector
            
        Returns:
            Tuple of (person_id, similarity_score)
        """
        try:
            best_match = None
            highest_similarity = -1
            
            for person_id, face_data in self.known_faces.items():
                known_embedding = np.array(face_data['embedding'])
                current_embedding = np.array(face_embedding)
                
                # Compute cosine similarity
                similarity = np.dot(known_embedding, current_embedding) / (
                    np.linalg.norm(known_embedding) * np.linalg.norm(current_embedding)
                )
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = person_id
            
            if highest_similarity >= self.recognition_threshold:
                return best_match, highest_similarity
            
            return None, highest_similarity
            
        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return None, 0.0
    
    def draw_results(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw detection and recognition results on the frame.
        
        Args:
            frame: Input image frame
            faces: List of face detection results
            
        Returns:
            Frame with visualization
        """
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            confidence = face['confidence']
            
            # Recognize face if embedding is available
            if face['embedding']:
                person_id, similarity = self.recognize_face(face['embedding'])
                if person_id:
                    name = self.known_faces[person_id]['name']
                    label = f"{name} ({similarity:.2f})"
                    color = (0, 255, 0)  # Green for recognized
                else:
                    label = f"Unknown ({similarity:.2f})"
                    color = (0, 0, 255)  # Red for unknown
            else:
                label = f"Face ({confidence:.2f})"
                color = (255, 0, 0)  # Blue for detected but not recognized
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks if available
            if face['landmarks']:
                for point in face['landmarks']:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, color, -1)
        
        return frame

def main():
    # Initialize detector and recognizer
    detector = FaceDetectorRecognizer(
        db_path="data/face_db",
        model_name="buffalo_l",
        detection_threshold=0.5,
        recognition_threshold=0.5
    )
    
    # Initialize camera
    cap = cv2.VideoCapture("rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101")
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Press 'a' to add a new face, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw results
        frame = detector.draw_results(frame, faces)
        
        # Show frame
        cv2.imshow("Face Detection and Recognition", frame)
        
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
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 