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
                 min_detection_confidence: float = 0.6,
                 update_interval: int = 50):  # Update embeddings every 50 new samples
        """
        Initialize automatic face collector.
        
        Args:
            embeddings_dir: Directory to store embeddings
            collection_interval: Time between embedding captures (seconds)
            min_detection_confidence: Minimum confidence for face detection
            update_interval: Number of new samples before updating embeddings
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.collection_interval = collection_interval
        self.min_detection_confidence = min_detection_confidence
        self.update_interval = update_interval
        
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
        self.collected_embeddings = defaultdict(list)  # cluster_id -> embeddings
        self.face_images = defaultdict(list)  # cluster_id -> face images
        self.clusters = {}  # Store clusters after processing
        self.cluster_names = {}  # Store cluster names
        
        # Load existing clusters
        self._load_existing_clusters()
    
    def _load_existing_clusters(self):
        """Load existing clusters from disk."""
        try:
            for cluster_dir in self.embeddings_dir.glob("cluster_*"):
                if not cluster_dir.is_dir():
                    continue
                
                # Load metadata
                metadata_file = cluster_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        cluster_id = str(metadata['cluster_id'])
                        self.cluster_names[cluster_id] = metadata['name']
                
                # Load embeddings
                embeddings_file = cluster_dir / "raw_embeddings.npy"
                if embeddings_file.exists():
                    embeddings = np.load(embeddings_file)
                    self.collected_embeddings[cluster_id] = embeddings.tolist()
                    
                    # Load face images if available
                    face_images_file = cluster_dir / "face_images.npy"
                    if face_images_file.exists():
                        self.face_images[cluster_id] = np.load(face_images_file, allow_pickle=True)
                
                logger.info(f"Loaded cluster {cluster_id} with {len(embeddings)} embeddings")
        
        except Exception as e:
            logger.error(f"Error loading existing clusters: {str(e)}")
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a frame and collect face embeddings.
        
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
                if current_time - self.last_capture_time >= self.collection_interval:
                    # Extract face region
                    face = small_frame[y:y+h, x:x+w]
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    
                    # Resize face for embedding
                    face_resized = cv2.resize(face_rgb, (150, 150))
                    
                    # Get face embedding
                    encoding = face_recognition.face_encodings(face_resized)
                    if encoding:
                        # Find closest cluster or create new one
                        cluster_id = self._assign_to_cluster(encoding[0])
                        
                        # Store embedding and face image
                        self.collected_embeddings[cluster_id].append(encoding[0])
                        self.face_images[cluster_id].append(face_resized)
                        
                        self.last_capture_time = current_time
                        
                        # Update cluster if enough new samples
                        if len(self.collected_embeddings[cluster_id]) % self.update_interval == 0:
                            self._update_cluster(cluster_id)
                        
                        logger.info(f"Collected embedding for cluster {cluster_id}")
        
        return small_frame
    
    def _assign_to_cluster(self, embedding: np.ndarray) -> str:
        """Assign embedding to closest cluster or create new one."""
        if not self.collected_embeddings:
            # Create first cluster with next available ID
            next_id = self._get_next_cluster_id()
            cluster_id = str(next_id)
            self.cluster_names[cluster_id] = f"Person_{cluster_id}"
            return cluster_id
        
        # Find closest cluster
        best_match = None
        highest_similarity = -1
        
        for cluster_id, embeddings in self.collected_embeddings.items():
            # Skip if this is a named cluster (protected)
            is_default_name = self.cluster_names[cluster_id].startswith("Person_")
            
            for known_embedding in embeddings[-5:]:  # Compare with last 5 embeddings
                similarity = 1 - face_recognition.face_distance([known_embedding], embedding)[0]
                if similarity > highest_similarity:
                    # Only update match if:
                    # 1. Current cluster has default name, or
                    # 2. Similarity is very high (0.8) for named clusters
                    if is_default_name or similarity > 0.8:
                        highest_similarity = similarity
                        best_match = cluster_id
        
        if highest_similarity >= 0.6:  # Similarity threshold
            return best_match
        
        # Create new cluster with next available ID
        next_id = self._get_next_cluster_id()
        cluster_id = str(next_id)
        self.cluster_names[cluster_id] = f"Person_{cluster_id}"
        return cluster_id
    
    def _get_next_cluster_id(self) -> int:
        """Get next available cluster ID."""
        existing_ids = set()
        # Check embeddings directory for existing clusters
        for cluster_dir in self.embeddings_dir.glob("cluster_*"):
            try:
                cluster_id = int(cluster_dir.name.split('_')[1])
                existing_ids.add(cluster_id)
            except (ValueError, IndexError):
                continue
        
        # Check current embeddings
        for cluster_id in self.collected_embeddings.keys():
            try:
                existing_ids.add(int(cluster_id))
            except ValueError:
                continue
        
        # Find first available ID
        next_id = 0
        while next_id in existing_ids:
            next_id += 1
        return next_id
    
    def _update_cluster(self, cluster_id: str):
        """Update cluster with new embeddings."""
        try:
            embeddings = np.array(self.collected_embeddings[cluster_id])
            face_images = np.array(self.face_images[cluster_id])
            
            # Use DBSCAN to find core samples
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(embeddings)
            
            # Get largest cluster
            if len(set(labels)) > 1:  # If multiple clusters found
                largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
                mask = labels == largest_cluster
                embeddings = embeddings[mask]
                face_images = face_images[mask]
            
            # Use KMeans to find representative embeddings
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(embeddings)
            
            # Find closest samples to centroids
            representatives = []
            representative_images = []
            for centroid in kmeans.cluster_centers_:
                distances = np.linalg.norm(embeddings - centroid, axis=1)
                closest_idx = np.argmin(distances)
                representatives.append(embeddings[closest_idx])
                representative_images.append(face_images[closest_idx])
            
            # Save cluster data
            cluster_dir = self.embeddings_dir / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            
            # Save raw data
            np.save(cluster_dir / "raw_embeddings.npy", embeddings)
            np.save(cluster_dir / "face_images.npy", face_images)
            
            # Save representative embeddings
            np.save(cluster_dir / "representative_embeddings.npy", np.array(representatives))
            
            # Save representative image (best quality one)
            best_image = representative_images[0]  # Use first representative
            cv2.imwrite(str(cluster_dir / "representative.jpg"), 
                       cv2.cvtColor(best_image, cv2.COLOR_RGB2BGR))
            
            # Save metadata
            metadata = {
                "cluster_id": cluster_id,
                "name": self.cluster_names[cluster_id],
                "num_embeddings": len(embeddings),
                "num_representatives": len(representatives),
                "last_update": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(cluster_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Updated cluster {cluster_id} with {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Error updating cluster {cluster_id}: {str(e)}")

def main():
    # Initialize collector
    collector = AutoFaceCollector(
        embeddings_dir="embeddings",
        collection_interval=1.0,
        min_detection_confidence=0.6,
        update_interval=50
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
            processed_frame = collector.process_frame(frame)
            
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