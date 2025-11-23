#openAI code for FD and FR 2nd

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import os
import time
import json
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class AutoFaceCollector:
    def __init__(self, 
                 embeddings_dir: str = "embeddings",
                 collection_interval: float = 1.0,
                 initial_embeddings: int = 50,
                 update_interval: int = 20,
                 min_detection_confidence: float = 0.6):
        """
        Initialize automatic face collector with dynamic learning.
        
        Args:
            embeddings_dir: Directory to store embeddings
            collection_interval: Time between embedding captures (seconds)
            initial_embeddings: Initial number of embeddings to collect per person
            update_interval: Number of new embeddings to collect before updating representative embeddings
            min_detection_confidence: Minimum confidence for face detection
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.collection_interval = collection_interval
        self.initial_embeddings = initial_embeddings
        self.update_interval = update_interval
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
        self.collected_embeddings = []
        self.face_images = []  # Store face images for representative image selection
        self.face_locations = []  # Store face locations with embeddings
        self.initial_collection_complete = False
        self.clusters = {}  # Store clusters after processing
        self.cluster_names = {}  # Store cluster names
        self.cluster_embeddings = defaultdict(list)  # Store ongoing embeddings by cluster
        self.last_update_time = {}  # Track last update time per cluster
        self.new_embeddings_count = defaultdict(int)  # Count new embeddings per cluster
        
        # Load existing clusters if available
        self._load_existing_clusters()
    
    def _load_existing_clusters(self):
        """Load existing clusters if available."""
        try:
            for cluster_dir in self.embeddings_dir.glob("cluster_*"):
                if not cluster_dir.is_dir():
                    continue
                
                # Load metadata
                metadata_file = cluster_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        cluster_id = int(metadata['cluster_id'])
                        name = metadata['name']
                        self.cluster_names[cluster_id] = name
                
                # Load representative embeddings
                embeddings_file = cluster_dir / "representative_embeddings.npy"
                if embeddings_file.exists():
                    embeddings = np.load(embeddings_file)
                    self.clusters[cluster_id] = embeddings
                    self.last_update_time[cluster_id] = time.time()
                    
                    logger.info(f"Loaded existing cluster {cluster_id} ({name}) with {len(embeddings)} representatives")
        except Exception as e:
            logger.error(f"Error loading existing clusters: {str(e)}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process a frame and collect face embeddings if conditions are met.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed frame, whether initial collection is complete)
        """
        # Create a copy for visualization
        vis_frame = frame.copy()
        
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
                
                current_time = time.time()
                # Initial collection phase
                if not self.initial_collection_complete and current_time - self.last_capture_time >= self.collection_interval:
                    self._collect_new_face(small_frame, x, y, w, h)
                    
                    # Check if initial collection is complete
                    if len(self.collected_embeddings) >= self.initial_embeddings:
                        self.initial_collection_complete = True
                        self._process_embeddings()
                
                # Dynamic update phase - if we already have clusters
                elif self.initial_collection_complete and current_time - self.last_capture_time >= self.collection_interval:
                    self._update_face_data(small_frame, rgb_small_frame, x, y, w, h)
        
        # Return processed frame for visualization
        return small_frame, self.initial_collection_complete
    
    def _collect_new_face(self, frame, x, y, w, h):
        """Collect a new face for initial training."""
        # Extract face region
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Resize face for embedding
        face_resized = cv2.resize(face_rgb, (150, 150))
        
        # Get face embedding
        encoding = face_recognition.face_encodings(face_resized)
        if encoding:
            self.collected_embeddings.append(encoding[0])
            self.face_locations.append((x, y, w, h))
            self.face_images.append(face_resized)  # Store face image
            self.last_capture_time = time.time()
            
            logger.info(f"Collected embedding {len(self.collected_embeddings)}/{self.initial_embeddings}")
    
    def _update_face_data(self, frame, rgb_frame, x, y, w, h):
        """Update face data for existing clusters."""
        # Extract face region
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Resize face for embedding
        face_resized = cv2.resize(face_rgb, (150, 150))
        
        # Get face embedding
        encoding = face_recognition.face_encodings(face_resized)
        if not encoding:
            return
        
        face_encoding = encoding[0]
        
        # Find best matching cluster
        best_match = None
        highest_similarity = -1
        
        for cluster_id, representatives in self.clusters.items():
            for rep_embedding in representatives:
                # Compare face with representative embedding
                similarity = face_recognition.compare_faces(
                    [rep_embedding], 
                    face_encoding,
                    tolerance=0.6  # Adjust as needed
                )[0]
                
                if similarity and similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = cluster_id
        
        # If we found a match, add this face to the cluster's data
        if best_match is not None and highest_similarity >= 0.5:  # Threshold for adding to existing cluster
            self.cluster_embeddings[best_match].append(face_encoding)
            self.new_embeddings_count[best_match] += 1
            
            # Also save the face image if it's a good quality example
            # We'll use this later for updating the representative image
            cluster_img_dir = self.embeddings_dir / f"cluster_{best_match}" / "images"
            cluster_img_dir.mkdir(exist_ok=True)
            
            # Save this face as a candidate representative image if it's centered and clear
            face_quality = self._assess_face_quality(face_resized)
            if face_quality > 0.7:  # Quality threshold
                img_filename = cluster_img_dir / f"face_{int(time.time())}.jpg"
                cv2.imwrite(str(img_filename), cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
            
            # Check if we need to update representatives
            if self.new_embeddings_count[best_match] >= self.update_interval:
                self._update_representatives(best_match)
                self.new_embeddings_count[best_match] = 0
        
        self.last_capture_time = time.time()
    
    def _assess_face_quality(self, face_img: np.ndarray) -> float:
        """
        Assess the quality of a face image for use as a representative.
        Higher score is better.
        """
        # Simple heuristics for face quality:
        # 1. Variance of pixel values (higher variance = more detail)
        # 2. Average brightness in middle range (not too dark, not too bright)
        
        var = np.var(face_img)
        avg_brightness = np.mean(face_img)
        
        # Normalize variance to 0-1 range (empirical max value)
        var_score = min(var / 5000.0, 1.0)
        
        # Brightness score is highest in middle range (around 128)
        brightness_score = 1.0 - abs(avg_brightness - 128) / 128
        
        # Combine scores
        quality_score = 0.6 * var_score + 0.4 * brightness_score
        return quality_score
    
    def _update_representatives(self, cluster_id: int):
        """Update representative embeddings for a cluster."""
        if cluster_id not in self.clusters or len(self.cluster_embeddings[cluster_id]) < 5:
            return
        
        logger.info(f"Updating representatives for cluster {cluster_id} ({self.cluster_names.get(cluster_id, 'Unknown')})")
        
        # Combine existing and new embeddings
        all_embeddings = list(self.clusters[cluster_id]) + self.cluster_embeddings[cluster_id]
        embeddings_array = np.array(all_embeddings)
        
        # Use KMeans to find 5 representative vectors
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(embeddings_array)
        
        # Update cluster representatives
        self.clusters[cluster_id] = kmeans.cluster_centers_
        
        # Clear collected embeddings for this cluster
        self.cluster_embeddings[cluster_id] = []
        
        # Save updated cluster data
        self._save_cluster(cluster_id, self.cluster_names[cluster_id], kmeans.cluster_centers_)
        
        # Update representative image
        self._update_representative_image(cluster_id)
    
    def _update_representative_image(self, cluster_id: int):
        """Update representative image for the cluster."""
        cluster_img_dir = self.embeddings_dir / f"cluster_{best_match}" / "images"
        if not cluster_img_dir.exists():
            return
        
        # Find all candidate images
        image_files = list(cluster_img_dir.glob("*.jpg"))
        if not image_files:
            return
        
        # Sort by creation time (newest first)
        image_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Take the newest image as representative
        representative_img_path = self.embeddings_dir / f"cluster_{cluster_id}" / "representative.jpg"
        
        # Copy the newest image as representative
        import shutil
        shutil.copy(str(image_files[0]), str(representative_img_path))
        
        logger.info(f"Updated representative image for cluster {cluster_id}")
    
    def _process_embeddings(self):
        """Process collected embeddings using DBSCAN and KMeans."""
        if not self.collected_embeddings:
            return
        
        logger.info("Processing embeddings for initial clustering...")
        
        # Convert to numpy array
        embeddings = np.array(self.collected_embeddings)
        
        # Use DBSCAN to find clusters
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(embeddings)
        
        # Group embeddings by cluster
        clusters = defaultdict(list)
        cluster_faces = defaultdict(list)  # Store face images by cluster
        for i, label in enumerate(labels):
            if label >= 0:  # Skip noise points
                clusters[label].append(embeddings[i])
                cluster_faces[label].append(self.face_images[i])
        
        # Process each cluster
        for cluster_id, cluster_embeddings in clusters.items():
            if len(cluster_embeddings) < 5:
                continue
            
            # Use KMeans to find 5 representative vectors
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(cluster_embeddings)
            
            # Store cluster results
            self.clusters[cluster_id] = kmeans.cluster_centers_
            
            # Prompt for cluster name
            name = input(f"Enter name for cluster {cluster_id}: ")
            self.cluster_names[cluster_id] = name
            
            # Save cluster data
            self._save_cluster(cluster_id, name, kmeans.cluster_centers_)
            
            # Save representative image
            faces = cluster_faces[cluster_id]
            if faces:
                # Create images directory
                cluster_img_dir = self.embeddings_dir / f"cluster_{cluster_id}" / "images"
                cluster_img_dir.mkdir(exist_ok=True)
                
                # Save first 5 images as candidates
                for i, face in enumerate(faces[:5]):
                    img_filename = f"face_{i}.jpg"
                    cv2.imwrite(
                        str(cluster_img_dir / img_filename), 
                        cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    )
                
                # Also save one as the representative image
                representative_img_path = self.embeddings_dir / f"cluster_{cluster_id}" / "representative.jpg"
                cv2.imwrite(
                    str(representative_img_path), 
                    cv2.cvtColor(faces[0], cv2.COLOR_RGB2BGR)
                )
    
    def _save_cluster(self, cluster_id: int, name: str, embeddings: np.ndarray):
        """Save cluster data to files."""
        # Create cluster directory
        cluster_dir = self.embeddings_dir / f"cluster_{cluster_id}"
        cluster_dir.mkdir(exist_ok=True)
        
        # Save embeddings
        np.save(cluster_dir / "representative_embeddings.npy", embeddings)
        
        # Save metadata
        metadata = {
            "cluster_id": int(cluster_id),  # Convert to Python int
            "name": name,
            "num_representatives": int(len(embeddings)),  # Convert to Python int
            "processing_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(cluster_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved cluster {cluster_id} as {name}")

def main():
    # Initialize collector
    collector = AutoFaceCollector(
        embeddings_dir="embeddings",
        collection_interval=1.0,
        initial_embeddings=50,  # Reduced from 150 for faster initial setup
        update_interval=20,     # Update representatives after collecting 20 new faces
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
            processed_frame, _ = collector.process_frame(frame)
            
            # Show frame
            cv2.imshow('Face Collection', processed_frame)
            
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
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load known face embeddings from files."""
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
                    
                    for cluster_id, embeddings in self.known_faces.items():
                        for known_embedding in embeddings:
                            similarity = face_recognition.compare_faces(
                                [known_embedding], 
                                encoding[0],
                                tolerance=1.0 - self.recognition_threshold
                            )[0]
                            
                            if similarity and similarity > highest_similarity:
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



# behaviour

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import joblib
from face_recognizer import FaceRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseClassifier:
    def __init__(self, model_path: str = "models/pose_model.pkl"):
        """Initialize pose classifier."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load trained model if exists
        self.model_path = Path(model_path)
        self.model = None
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info("Loaded pose classification model")
            except Exception as e:
                logger.error(f"Error loading pose model: {str(e)}")
    
    def extract_pose_features(self, landmarks) -> np.ndarray:
        """Extract relevant features from pose landmarks."""
        features = []
        
        # Convert landmarks to numpy array
        points = np.zeros((33, 3))
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = [landmark.x, landmark.y, landmark.z]
        
        # Calculate angles between joints
        # Shoulders
        left_shoulder = points[11]
        right_shoulder = points[12]
        
        # Elbows
        left_elbow = points[13]
        right_elbow = points[14]
        
        # Wrists
        left_wrist = points[15]
        right_wrist = points[16]
        
        # Hips
        left_hip = points[23]
        right_hip = points[24]
        
        # Calculate angles
        def calculate_angle(p1, p2, p3):
            """Calculate angle between three points."""
            v1 = p1 - p2
            v2 = p3 - p2
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return np.degrees(angle)
        
        # Upper body angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Shoulder angles
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, [right_shoulder[0], right_shoulder[1] + 1, right_shoulder[2]])
        
        # Hip angles
        hip_angle = calculate_angle(left_hip, right_hip, [right_hip[0], right_hip[1] + 1, right_hip[2]])
        
        # Add features
        features.extend([
            left_elbow_angle,
            right_elbow_angle,
            shoulder_angle,
            hip_angle,
            # Add relative positions
            left_wrist[1] - left_shoulder[1],  # Left hand raise
            right_wrist[1] - right_shoulder[1],  # Right hand raise
            (left_hip[1] + right_hip[1])/2 - (left_shoulder[1] + right_shoulder[1])/2,  # Torso angle
        ])
        
        return np.array(features)
    
    def classify_pose(self, frame) -> tuple:
        """
        Classify pose in frame.
        
        Returns:
            tuple: (pose_name, landmarks)
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return "Unknown", None
        
        # Extract features
        features = self.extract_pose_features(results.pose_landmarks)
        
        # Rule-based classification if no model
        if self.model is None:
            return self._rule_based_classification(features), results.pose_landmarks
        
        # Model-based classification
        try:
            pose = self.model.predict([features])[0]
            return pose, results.pose_landmarks
        except Exception as e:
            logger.error(f"Error in pose classification: {str(e)}")
            return "Unknown", results.pose_landmarks
    
    def _rule_based_classification(self, features) -> str:
        """Simple rule-based pose classification."""
        left_elbow_angle = features[0]
        right_elbow_angle = features[1]
        shoulder_angle = features[2]
        hip_angle = features[3]
        left_hand_raise = features[4]
        right_hand_raise = features[5]
        torso_angle = features[6]
        
        # Sitting detection
        if torso_angle > 0.2:  # Upper body leaning back
            return "Sitting"
        
        # Clapping detection
        if (left_elbow_angle < 60 and right_elbow_angle < 60 and 
            abs(left_hand_raise - right_hand_raise) < 0.1):
            return "Clapping"
        
        # Raised hand detection
        if left_hand_raise < -0.2 or right_hand_raise < -0.2:
            return "Raise Hand"
        
        # Folded hands detection
        if (left_elbow_angle < 90 and right_elbow_angle < 90 and 
            abs(left_hand_raise - right_hand_raise) < 0.1):
            return "Fold Hand"
        
        # Default to standing
        return "Standing"

class BehaviorAnalyzer:
    def __init__(self, 
                 location: str,
                 log_file: str = "behavior_log.csv",
                 detection_interval: float = 1.5):
        """
        Initialize behavior analyzer.
        
        Args:
            location: Location identifier (e.g., "Library", "Classroom")
            log_file: Path to behavior log CSV file
            detection_interval: Minimum time between detections in seconds
        """
        self.location = location
        self.log_file = Path(log_file)
        self.detection_interval = detection_interval
        
        # Initialize components
        self.face_recognizer = FaceRecognizer(
            embeddings_dir="embeddings",
            recognition_threshold=0.6
        )
        self.pose_classifier = PoseClassifier()
        
        # Initialize state
        self.last_detection_time = 0
        self.last_message = ""
        
        # Ensure log file exists
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize behavior log CSV file."""
        if not self.log_file.exists():
            df = pd.DataFrame(columns=['Timestamp', 'Name', 'Pose', 'Location', 'Sentence'])
            df.to_csv(self.log_file, index=False)
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a frame for behavior analysis.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (processed_frame, message)
        """
        current_time = datetime.now()
        
        # Check detection interval
        if (current_time - datetime.fromtimestamp(self.last_detection_time)).total_seconds() < self.detection_interval:
            return frame, self.last_message
        
        try:
            # Face recognition
            processed_frame, face_results = self.face_recognizer.process_frame(frame)
            
            # Pose detection
            pose_name, pose_landmarks = self.pose_classifier.classify_pose(frame)
            
            # Draw pose landmarks if detected
            if pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    processed_frame,
                    pose_landmarks,
                    self.pose_classifier.mp_pose.POSE_CONNECTIONS
                )
            
            # Generate message for each detected person
            messages = []
            for result in face_results:
                name = result["person_name"]
                message = f"{name} is {pose_name.lower()} in the {self.location}"
                messages.append(message)
                
                # Log behavior
                self._log_behavior(name, pose_name, message)
            
            # Update state
            self.last_detection_time = current_time.timestamp()
            self.last_message = "; ".join(messages) if messages else "No one detected"
            
            return processed_frame, self.last_message
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, "Error processing frame"
    
    def _log_behavior(self, name: str, pose: str, sentence: str):
        """Log behavior to CSV file."""
        try:
            record = {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Name': name,
                'Pose': pose,
                'Location': self.location,
                'Sentence': sentence
            }
            
            df = pd.DataFrame([record])
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            
        except Exception as e:
            logger.error(f"Error logging behavior: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass 

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
import threading
from PIL import Image, ImageTk
from behavior_analyzer import BehaviorAnalyzer

class BehaviorMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Behavior Analysis System")
        self.root.state('zoomed')  # Maximize window
        
        # Initialize variables
        self.camera_source = "rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101"
        self.location = "Library"  # Default location
        self.is_monitoring = False
        self.analyzer = None
        self.cap = None
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel (video and controls)
        self.left_panel = ttk.Frame(self.main_container)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right panel (status and logs)
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup components
        self.setup_control_panel()
        self.setup_video_panel()
        self.setup_status_panel()
        self.setup_log_panel()
    
    def setup_control_panel(self):
        """Setup control panel with buttons and settings."""
        control_frame = ttk.LabelFrame(self.left_panel, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Location settings
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Location:").pack(side=tk.LEFT, padx=5)
        self.location_var = tk.StringVar(value=self.location)
        location_entry = ttk.Entry(settings_frame, textvariable=self.location_var, width=20)
        location_entry.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    def setup_video_panel(self):
        """Setup video display panel."""
        video_frame = ttk.LabelFrame(self.left_panel, text="Camera Feed")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
    
    def setup_status_panel(self):
        """Setup status display panel."""
        status_frame = ttk.LabelFrame(self.right_panel, text="Current Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="System Status: Ready", font=('Arial', 12))
        self.status_label.pack(pady=5)
        
        self.behavior_label = ttk.Label(status_frame, text="No behavior detected", font=('Arial', 12))
        self.behavior_label.pack(pady=5)
    
    def setup_log_panel(self):
        """Setup behavior log display panel."""
        log_frame = ttk.LabelFrame(self.right_panel, text="Behavior Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview
        self.log_tree = ttk.Treeview(log_frame, 
                                    columns=("Time", "Name", "Pose", "Location", "Description"),
                                    show="headings")
        
        # Define columns
        self.log_tree.heading("Time", text="Time")
        self.log_tree.heading("Name", text="Name")
        self.log_tree.heading("Pose", text="Pose")
        self.log_tree.heading("Location", text="Location")
        self.log_tree.heading("Description", text="Description")
        
        # Set column widths
        self.log_tree.column("Time", width=150)
        self.log_tree.column("Name", width=100)
        self.log_tree.column("Pose", width=100)
        self.log_tree.column("Location", width=100)
        self.log_tree.column("Description", width=300)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack components
        self.log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def update_log_display(self):
        """Update behavior log display."""
        try:
            if not Path("behavior_log.csv").exists():
                return
            
            # Read latest logs
            df = pd.read_csv("behavior_log.csv")
            
            # Clear current display
            for item in self.log_tree.get_children():
                self.log_tree.delete(item)
            
            # Add latest entries (last 100)
            for _, row in df.tail(100).iloc[::-1].iterrows():
                self.log_tree.insert("", 0, values=(
                    row['Timestamp'],
                    row['Name'],
                    row['Pose'],
                    row['Location'],
                    row['Sentence']
                ))
                
        except Exception as e:
            print(f"Error updating log display: {str(e)}")
    
    def start_monitoring(self):
        """Start behavior monitoring."""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            # Initialize analyzer
            self.analyzer = BehaviorAnalyzer(
                location=self.location_var.get(),
                log_file="behavior_log.csv",
                detection_interval=1.5
            )
            
            # Update state
            self.is_monitoring = True
            self.status_label.config(text="System Status: Monitoring")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Start update loop
            self.update_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop behavior monitoring."""
        self.is_monitoring = False
        if self.cap:
            self.cap.release()
        
        # Update UI
        self.status_label.config(text="System Status: Stopped")
        self.behavior_label.config(text="No behavior detected")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.video_label.config(image='')
    
    def update_frame(self):
        """Update video frame and behavior analysis."""
        if self.is_monitoring and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Process frame
                    processed_frame, message = self.analyzer.process_frame(frame)
                    
                    # Update behavior message
                    self.behavior_label.config(text=message)
                    
                    # Update log display periodically
                    if datetime.now().second % 2 == 0:
                        self.update_log_display()
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (800, 600))
                    img = Image.fromarray(frame_resized)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                
            except Exception as e:
                print(f"Error updating frame: {str(e)}")
            
            if self.is_monitoring:
                self.root.after(30, self.update_frame)  # Update at ~30 FPS
    
    def on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.stop_monitoring()
            self.root.quit()

def main():
    root = tk.Tk()
    app = BehaviorMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 








import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
from pathlib import Path
import sys
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import our systems
from attendance_system import AttendanceUI, AttendanceManager
from crowd_management_app import CrowdManagementApp
from behavior_monitor import BehaviorMonitorApp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OmniSightApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OmniSight - Smart Surveillance System")
        self.root.state('zoomed')  # Maximize window
        
        # Initialize paths
        self.setup_paths()
        
        # Setup main UI
        self.setup_ui()
        
        # Initialize subsystems
        self.attendance_system = None
        self.crowd_system = None
        self.behavior_system = None
    
    def setup_paths(self):
        """Setup necessary directories."""
        # Create main directories
        self.base_dir = Path("omnisight_data")
        self.logs_dir = self.base_dir / "logs"
        self.reports_dir = self.base_dir / "reports"
        self.attendance_dir = self.base_dir / "attendance_logs"
        self.models_dir = Path("models")
        self.data_dir = Path("data")
        self.utils_dir = Path("utils")
        
        # Create directories
        for dir_path in [self.base_dir, self.logs_dir, self.reports_dir, 
                        self.attendance_dir, self.models_dir, self.data_dir, 
                        self.utils_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / f"omnisight_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    def setup_ui(self):
        """Setup the main user interface."""
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create title frame
        title_frame = ttk.Frame(self.main_container)
        title_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Title label with large, bold font
        title_label = ttk.Label(
            title_frame, 
            text="OmniSight - Smart Surveillance System",
            font=('Arial', 24, 'bold')
        )
        title_label.pack()
        
        # Subtitle
        subtitle_label = ttk.Label(
            title_frame,
            text="Integrated Surveillance Solution for Attendance, Crowd Management, and Behavior Analysis",
            font=('Arial', 12)
        )
        subtitle_label.pack(pady=5)
        
        # Create systems frame
        systems_frame = ttk.Frame(self.main_container)
        systems_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create system buttons with descriptions
        self.create_system_button(
            systems_frame,
            "Attendance System",
            "Track attendance using facial recognition",
            self.launch_attendance_system,
            0
        )
        
        self.create_system_button(
            systems_frame,
            "Crowd Management",
            "Monitor and analyze crowd density",
            self.launch_crowd_management,
            1
        )
        
        self.create_system_button(
            systems_frame,
            "Behavior Analysis",
            "Analyze and log behavioral patterns",
            self.launch_behavior_analysis,
            2
        )
        
        # Create status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.main_container,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(fill=tk.X, padx=5, pady=5)
    
    def create_system_button(self, parent, title, description, command, column):
        """Create a styled button for each system."""
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, padx=10, pady=10, sticky="nsew")
        
        # Configure style for button
        style = ttk.Style()
        style.configure("System.TButton", font=('Arial', 12, 'bold'))
        
        # Create button
        btn = ttk.Button(
            frame,
            text=title,
            style="System.TButton",
            command=command
        )
        btn.pack(fill=tk.X, padx=20, pady=10)
        
        # Add description
        desc_label = ttk.Label(
            frame,
            text=description,
            wraplength=200,
            justify=tk.CENTER
        )
        desc_label.pack(pady=5)
        
        # Configure grid weights
        parent.grid_columnconfigure(column, weight=1)
    
    def launch_attendance_system(self):
        """Launch the attendance system in a new window."""
        try:
            # Create window
            window = tk.Toplevel(self.root)
            window.title("Attendance System")
            window.state('zoomed')
            
            # Initialize attendance manager
            attendance_manager = AttendanceManager(
                class_name="Class1",
                attendance_dir=str(self.base_dir / "attendance_logs"),
                slot_start=(9, 0),
                slot_end=(17, 0),
                late_threshold=(9, 15)
            )
            
            # Initialize UI with both window and manager
            self.attendance_system = AttendanceUI(window, attendance_manager)
            
            # Update status
            self.status_var.set("Attendance System launched")
            logger.info("Attendance System launched")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Attendance System: {str(e)}")
            logger.error(f"Error launching Attendance System: {str(e)}")
    
    def launch_crowd_management(self):
        """Launch the crowd management system."""
        try:
            # Create window
            crowd_window = tk.Toplevel(self.root)
            
            # Import and launch Crowdy
            from crowdy import CrowdyApp
            self.crowd_system = CrowdyApp(crowd_window)
            
            # Update status
            self.status_var.set("Crowd Management System launched")
            logger.info("Crowd Management System launched")
            
        except Exception as e:
            logger.error(f"Error launching Crowd Management System: {str(e)}")
            messagebox.showerror("Error", f"Failed to launch Crowd Management System: {str(e)}")
    
    def launch_behavior_analysis(self):
        """Launch the behavior analysis system in a new window."""
        try:
            window = tk.Toplevel(self.root)
            window.title("Behavior Analysis System")
            window.state('zoomed')
            
            # Initialize behavior analysis system
            self.behavior_system = BehaviorMonitorApp(window)
            
            # Update status
            self.status_var.set("Behavior Analysis System launched")
            logger.info("Behavior Analysis System launched")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Behavior Analysis System: {str(e)}")
            logger.error(f"Error launching Behavior Analysis System: {str(e)}")
    
    def on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit OmniSight?"):
            # Stop all active systems
            if self.attendance_system:
                self.attendance_system.stop_attendance()
            if self.crowd_system:
                self.crowd_system.stop_monitoring()
            if self.behavior_system:
                self.behavior_system.stop_monitoring()
            
            logger.info("OmniSight application closed")
            self.root.quit()

def main():
    # Create main window
    root = tk.Tk()
    app = OmniSightApp(root)
    
    # Set closing protocol
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start application
    logger.info("OmniSight application started")
    root.mainloop()

if __name__ == "__main__":
    main() 