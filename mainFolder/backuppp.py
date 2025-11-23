#openAI code for FD and FR 1st

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import os
import time
import json
from pathlib import Path
import logging
from typing import List, Dict, Optional
from sklearn.cluster import DBSCAN, KMeans

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceCollector:
    def __init__(self, 
                 embeddings_dir: str = "embeddings",
                 collection_interval: float = 1.0,
                 max_embeddings: int = 150,
                 min_detection_confidence: float = 0.6):
        """
        Initialize face collector.
        
        Args:
            embeddings_dir: Directory to store embeddings
            collection_interval: Time between embedding captures (seconds)
            max_embeddings: Maximum number of embeddings to collect per person
            min_detection_confidence: Minimum confidence for face detection
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.collection_interval = collection_interval
        self.max_embeddings = max_embeddings
        self.min_detection_confidence = min_detection_confidence
        
        # Create directories
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )
        
        # Collection state
        self.last_capture_time = 0
        self.current_person_id = None
        self.collected_embeddings = []
    
    def start_collection(self, person_id: str):
        """Start collecting embeddings for a person."""
        self.current_person_id = person_id
        self.collected_embeddings = []
        self.last_capture_time = 0
        logger.info(f"Started collecting embeddings for person {person_id}")
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a frame and collect face embedding if conditions are met.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with visualizations
        """
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_small_frame)
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = small_frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Draw bounding box
                cv2.rectangle(small_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Check if it's time to capture embedding
                current_time = time.time()
                if (self.current_person_id and 
                    current_time - self.last_capture_time >= self.collection_interval and
                    len(self.collected_embeddings) < self.max_embeddings):
                    
                    # Extract face region
                    face = small_frame[y:y+h, x:x+w]
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    
                    # Resize face for embedding
                    face_resized = cv2.resize(face_rgb, (150, 150))
                    
                    # Get face embedding
                    encoding = face_recognition.face_encodings(face_resized)
                    if encoding:
                        self.collected_embeddings.append(encoding[0])
                        self.last_capture_time = current_time
                        
                        # Save progress
                        self._save_progress()
                        
                        logger.info(f"Collected embedding {len(self.collected_embeddings)}/{self.max_embeddings}")
        
        return small_frame
    
    def _save_progress(self):
        """Save collected embeddings to file."""
        if not self.current_person_id or not self.collected_embeddings:
            return
        
        # Create person directory
        person_dir = self.embeddings_dir / self.current_person_id
        person_dir.mkdir(exist_ok=True)
        
        # Save embeddings
        embeddings_file = person_dir / "raw_embeddings.npy"
        np.save(embeddings_file, np.array(self.collected_embeddings))
        
        # Save metadata
        metadata = {
            "person_id": self.current_person_id,
            "num_embeddings": len(self.collected_embeddings),
            "last_update": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(person_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_embeddings(self, 
                          eps: float = 0.5,
                          min_samples: int = 5,
                          n_clusters: int = 5) -> Optional[List[np.ndarray]]:
        """
        Process collected embeddings using DBSCAN and KMeans.
        
        Args:
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            n_clusters: Number of KMeans clusters
            
        Returns:
            List of representative embeddings
        """
        if not self.collected_embeddings:
            logger.warning("No embeddings to process")
            return None
        
        # Convert to numpy array
        embeddings = np.array(self.collected_embeddings)
        
        # Use DBSCAN to remove outliers
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings)
        
        # Get the largest cluster
        largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
        cluster_embeddings = embeddings[labels == largest_cluster]
        
        if len(cluster_embeddings) < n_clusters:
            logger.warning(f"Not enough samples in largest cluster ({len(cluster_embeddings)} < {n_clusters})")
            return None
        
        # Use KMeans to find representative vectors
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(cluster_embeddings)
        
        # Get cluster centers
        representative_embeddings = kmeans.cluster_centers_
        
        # Save processed embeddings
        if self.current_person_id:
            person_dir = self.embeddings_dir / self.current_person_id
            np.save(person_dir / "representative_embeddings.npy", representative_embeddings)
            
            # Save metadata
            metadata = {
                "person_id": self.current_person_id,
                "num_representatives": len(representative_embeddings),
                "processing_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "dbscan_params": {"eps": eps, "min_samples": min_samples},
                "kmeans_params": {"n_clusters": n_clusters}
            }
            with open(person_dir / "processed_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return representative_embeddings.tolist()

def main():
    # Initialize collector
    collector = FaceCollector(
        embeddings_dir="embeddings",
        collection_interval=5.0,
        max_embeddings=150,
        min_detection_confidence=0.6
    )
    
    # Initialize camera
    cap = cv2.VideoCapture("rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101")
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    # Get person ID
    person_id = input("Enter person ID: ")
    collector.start_collection(person_id)
    
    logger.info("Press 'q' to quit, 'p' to process embeddings")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame
            processed_frame = collector.process_frame(frame)
            
            # Show frame
            cv2.imshow('Face Collection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                if len(collector.collected_embeddings) >= collector.max_embeddings:
                    logger.info("Processing embeddings...")
                    representative_embeddings = collector.process_embeddings()
                    if representative_embeddings:
                        logger.info(f"Found {len(representative_embeddings)} representative embeddings")
                else:
                    logger.warning(f"Need more embeddings: {len(collector.collected_embeddings)}/{collector.max_embeddings}")
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 





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
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load known face embeddings from files."""
        try:
            for person_dir in self.embeddings_dir.glob("*"):
                if not person_dir.is_dir():
                    continue
                
                # Load representative embeddings
                embeddings_file = person_dir / "representative_embeddings.npy"
                if embeddings_file.exists():
                    embeddings = np.load(embeddings_file)
                    self.known_faces[person_dir.name] = embeddings.tolist()
                    logger.info(f"Loaded {len(embeddings)} embeddings for {person_dir.name}")
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
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_small_frame)
        
        recognition_results = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = small_frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Extract face region
                face = small_frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                
                # Resize face for embedding
                face_resized = cv2.resize(face_rgb, (150, 150))
                
                # Get face embedding
                encoding = face_recognition.face_encodings(face_resized)
                if encoding:
                    # Compare with known faces
                    best_match = None
                    highest_similarity = -1
                    
                    for person_id, embeddings in self.known_faces.items():
                        for known_embedding in embeddings:
                            similarity = face_recognition.compare_faces(
                                [known_embedding], 
                                encoding[0],
                                tolerance=1.0 - self.recognition_threshold
                            )[0]
                            
                            if similarity and similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match = person_id
                    
                    # Draw results
                    if best_match and highest_similarity >= self.recognition_threshold:
                        color = (0, 255, 0)  # Green for recognized
                        label = f"{best_match} ({highest_similarity:.2f})"
                        recognition_results.append({
                            "person_id": best_match,
                            "confidence": highest_similarity,
                            "bbox": (x, y, w, h)
                        })
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        label = "Unknown"
                    
                    # Draw bounding box and label
                    cv2.rectangle(small_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(small_frame, label, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return small_frame, recognition_results

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
            
            # Log results
            if results:
                for result in results:
                    logger.info(f"Recognized {result['person_id']} with confidence {result['confidence']:.2f}")
            
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