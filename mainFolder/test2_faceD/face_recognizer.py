import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, 
                 embeddings_dir: str = "embeddings",
                 recognition_threshold: float = 0.6,
                 min_detection_confidence: float = 0.6):
        """
        Initialize face recognizer.
        
        Args:
            embeddings_dir: Directory containing face embeddings
            recognition_threshold: Minimum similarity threshold for recognition
            min_detection_confidence: Minimum confidence for face detection
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.recognition_threshold = recognition_threshold
        self.min_detection_confidence = min_detection_confidence
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )
        
        # Load known faces
        self.known_faces: Dict[str, List[np.ndarray]] = {}
        self.cluster_names: Dict[str, str] = {}  # Store cluster_id to name mapping
        self.representative_images: Dict[str, np.ndarray] = {}  # Store representative images
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load known face embeddings and representative images from files."""
        try:
            for cluster_dir in self.embeddings_dir.glob("cluster_*"):
                if not cluster_dir.is_dir():
                    continue
                
                # Load metadata to get cluster name
                metadata_file = cluster_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        cluster_id = str(metadata['cluster_id'])
                        name = metadata['name']
                        self.cluster_names[cluster_id] = name
                
                # Load representative embeddings
                embeddings_file = cluster_dir / "representative_embeddings.npy"
                if embeddings_file.exists():
                    embeddings = np.load(embeddings_file)
                    self.known_faces[cluster_id] = embeddings.tolist()
                    logger.info(f"Loaded {len(embeddings)} embeddings for {name}")
                
                # Load representative image if available
                rep_image_file = cluster_dir / "representative.jpg"
                if rep_image_file.exists():
                    try:
                        rep_image = cv2.imread(str(rep_image_file))
                        if rep_image is not None:
                            self.representative_images[cluster_id] = rep_image
                            logger.info(f"Loaded representative image for {name}")
                    except Exception as e:
                        logger.error(f"Error loading representative image for {name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading known faces: {str(e)}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a frame and recognize faces.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed frame, list of recognition results)
        """
        try:
            if frame is None or frame.size == 0:
                logger.error("Received empty frame")
                return frame, []

            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(rgb_small_frame)
            
            # Create a visualization frame (we'll add the representative images here)
            vis_frame = small_frame.copy()
            
            recognition_results = []
            if results.detections:
                for detection in results.detections:
                    try:
                        # Get bounding box
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = small_frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                    int(bboxC.width * iw), int(bboxC.height * ih)
                        
                        # Validate bounding box coordinates
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, iw - x)
                        h = min(h, ih - y)
                        
                        # Check if face region is valid
                        if w <= 0 or h <= 0:
                            continue
                        
                        # Extract face region
                        face = small_frame[y:y+h, x:x+w]
                        if face is None or face.size == 0:
                            continue
                            
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        
                        # Resize face for embedding
                        face_resized = cv2.resize(face_rgb, (150, 150))
                        
                        # Get face embedding
                        encoding = face_recognition.face_encodings(face_resized)
                        if encoding:
                            # Compare with known faces
                            best_match = None
                            highest_similarity = -1
                            
                            for cluster_id, embeddings in self.known_faces.items():
                                for known_embedding in embeddings:
                                    similarity = 1 - face_recognition.face_distance([known_embedding], encoding[0])[0]
                                    
                                    if similarity > highest_similarity:
                                        highest_similarity = similarity
                                        best_match = cluster_id
                            
                            # Draw results
                            if best_match and highest_similarity >= self.recognition_threshold:
                                person_name = self.cluster_names.get(best_match, best_match)
                                color = (0, 255, 0)  # Green for recognized
                                label = f"{person_name} ({highest_similarity:.2f})"
                                recognition_results.append({
                                    "cluster_id": best_match,
                                    "person_name": person_name,
                                    "confidence": highest_similarity,
                                    "bbox": (x, y, w, h)
                                })
                                
                                # Display representative image if available
                                if best_match in self.representative_images:
                                    rep_img = self.representative_images[best_match]
                                    rep_height = 100  # Fixed height for rep image
                                    rep_width = int(rep_img.shape[1] * (rep_height / rep_img.shape[0]))
                                    rep_img_resized = cv2.resize(rep_img, (rep_width, rep_height))
                                    
                                    # Calculate position for rep image (above the face bbox)
                                    rep_x = max(0, x)
                                    rep_y = max(0, y - rep_height - 30)
                                    
                                    # Make sure rep image fits in the frame
                                    if rep_y + rep_height <= ih and rep_x + rep_width <= iw:
                                        # Create background for the image
                                        cv2.rectangle(vis_frame, 
                                                     (rep_x-2, rep_y-2), 
                                                     (rep_x+rep_width+2, rep_y+rep_height+2), 
                                                     (255, 255, 255), -1)
                                        
                                        # Place the rep image
                                        vis_frame[rep_y:rep_y+rep_height, rep_x:rep_x+rep_width] = rep_img_resized
                            else:
                                color = (0, 0, 255)  # Red for unknown
                                label = "Unknown"
                            
                            # Draw bounding box and label
                            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(vis_frame, label, (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    except Exception as e:
                        logger.error(f"Error processing detection: {str(e)}")
                        continue
            
            return vis_frame, recognition_results
            
        except Exception as e:
            logger.error(f"Error in process_frame: {str(e)}")
            return frame, []

def main():
    # Initialize recognizer
    recognizer = FaceRecognizer(
        embeddings_dir="embeddings",
        recognition_threshold=0.6,
        min_detection_confidence=0.6
    )
    
    # Initialize camera
    cap = cv2.VideoCapture("rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101")
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame
            processed_frame, results = recognizer.process_frame(frame)
            
            # Show frame
            cv2.imshow('Face Recognition', processed_frame)
            
            # Handle key presses
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()