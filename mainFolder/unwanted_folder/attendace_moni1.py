import cv2
import numpy as np
import pandas as pd
import os
import json
import datetime
import time
import logging
import threading
import face_recognition
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from pathlib import Path
import base64
from queue import Queue, Empty
import csv
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttendanceConfig:
    """Configuration class for attendance system settings"""
    def __init__(self):
        self.class_name = "Class101"
        self.attendance_slots = {
            "Morning": {"start": "09:00", "end": "09:10"},
            "Afternoon": {"start": "14:00", "end": "14:10"}
        }
        self.cooldown_period = 60  # seconds before re-marking the same person
        self.data_directory = "attendance_data"
        self.voice_feedback = True
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Load student IDs from file if exists or create a new mapping
        self.id_mapping_file = os.path.join(self.data_directory, "id_mapping.json")
        self.load_id_mapping()
    
    def load_id_mapping(self):
        """Load existing ID mapping or create a new one"""
        if os.path.exists(self.id_mapping_file):
            with open(self.id_mapping_file, 'r') as f:
                self.id_mapping = json.load(f)
        else:
            self.id_mapping = {}
    
    def save_id_mapping(self):
        """Save ID mapping to file"""
        with open(self.id_mapping_file, 'w') as f:
            json.dump(self.id_mapping, f, indent=2)
    
    def get_or_create_id(self, name):
        """Get existing ID for a name or create a new one"""
        if name not in self.id_mapping:
            # Create new ID: use prefix "STU" + next available number
            next_id = len(self.id_mapping) + 1
            self.id_mapping[name] = f"STU{next_id:03d}"
            self.save_id_mapping()
        
        return self.id_mapping[name]
    
    def is_within_attendance_slot(self):
        """Check if current time is within any attendance slot"""
        now = datetime.datetime.now().time()
        current_time_str = now.strftime("%H:%M")
        
        for slot_name, slot_times in self.attendance_slots.items():
            start_time = datetime.datetime.strptime(slot_times["start"], "%H:%M").time()
            end_time = datetime.datetime.strptime(slot_times["end"], "%H:%M").time()
            
            if start_time <= now <= end_time:
                return True, slot_name
        
        return False, None

class FaceRecognizer:
    """Face recognition implementation using OpenCV and face_recognition"""
    def __init__(self, embeddings_dir="embeddings", recognition_threshold=0.6, min_detection_confidence=0.5):
        self.embeddings_dir = embeddings_dir
        self.recognition_threshold = recognition_threshold
        self.min_detection_confidence = min_detection_confidence
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load known face embeddings
        self.known_face_embeddings = []
        self.known_face_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known face embeddings from disk"""
        if not os.path.exists(self.embeddings_dir):
            return
        
        # Look for numpy files containing embeddings
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith('.npz'):
                filepath = os.path.join(self.embeddings_dir, filename)
                data = np.load(filepath)
                
                # Extract the name from the filename
                name = os.path.splitext(filename)[0]
                
                # Add to known faces
                self.known_face_embeddings.append(data['embedding'])
                self.known_face_names.append(name)
        
        logger.info(f"Loaded {len(self.known_face_names)} known faces")
    
    def save_face_embedding(self, name, embedding):
        """Save a new face embedding to disk"""
        # Ensure name is safe for filename
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ").rstrip()
        
        # Save as numpy file
        filepath = os.path.join(self.embeddings_dir, f"{safe_name}.npz")
        np.savez(filepath, embedding=embedding)
        
        # Reload known faces
        self.known_face_embeddings = []
        self.known_face_names = []
        self.load_known_faces()
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def compute_face_embedding(self, face_image):
        """Compute face embedding for recognition"""
        try:
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Compute face encoding
            face_encodings = face_recognition.face_encodings(rgb_image)
            
            if face_encodings:
                return face_encodings[0]
            return None
        except Exception as e:
            logger.error(f"Error computing face embedding: {e}")
            return None
    
    def recognize_face(self, face_embedding):
        """Recognize a face based on its embedding"""
        if not self.known_face_embeddings:
            return None
        
        try:
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_face_embeddings, face_embedding)
            
            # Find the closest match
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            # Check if the match is close enough
            if min_distance <= self.recognition_threshold:
                return self.known_face_names[min_distance_idx]
            
            return None
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None
    
    def process_frame(self, frame, skip_recognition=False):
        """Process a video frame for face detection and recognition"""
        # Make a copy of the frame and resize for faster processing
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        processed_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame_resized)
        
        # Recognition results
        recognition_results = []
        
        # Process each face
        for (x, y, w, h) in faces:
            # Scale back to original size
            x, y, w, h = x*2, y*2, w*2, h*2
            
            # Draw rectangle around face
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if not skip_recognition:
                # Extract face region
                face_image = frame[y:y+h, x:x+w]
                
                # Compute face embedding
                face_embedding = self.compute_face_embedding(face_image)
                
                if face_embedding is not None:
                    # Try to recognize the face
                    person_name = self.recognize_face(face_embedding)
                    
                    if person_name:
                        # Draw label with name
                        cv2.putText(processed_frame, person_name, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Add to results
                        recognition_results.append({
                            "person_name": person_name,
                            "confidence": 1.0,  # Placeholder for confidence
                            "location": (x, y, w, h)
                        })
                    else:
                        # Unknown face
                        cv2.putText(processed_frame, "Unknown", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                # Skip recognition, just show "Face Detected"
                cv2.putText(processed_frame, "Face Detected", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return processed_frame, recognition_results
    
    def add_new_face(self, frame, name):
        """Add a new face to the known faces"""
        # Detect faces
        faces = self.detect_faces(frame)
        
        if not faces:
            return False, "No face detected"
        
        if len(faces) > 1:
            return False, "Multiple faces detected. Please ensure only one face is in the frame."
        
        # Extract the face
        (x, y, w, h) = faces[0]
        face_image = frame[y:y+h, x:x+w]
        
        # Compute face embedding
        face_embedding = self.compute_face_embedding(face_image)
        
        if face_embedding is None:
            return False, "Failed to compute face embedding"
        
        # Save the face embedding
        self.save_face_embedding(name, face_embedding)
        
        return True, f"Face for {name} added successfully"

class AttendanceSystem:
    """Main attendance system using face recognition"""
    def __init__(self, config=None, camera_source=0):
        """Initialize the attendance system"""
        self.config = config if config else AttendanceConfig()
        self.camera_source = camera_source
        self.face_recognizer = FaceRecognizer(
            embeddings_dir="embeddings",
            recognition_threshold=0.6,
            min_detection_confidence=0.6
        )
        
        # Initialize the attendance tracker
        self.last_attendance_time = {}  # To track when a person was last marked present
        self.current_day_attendance = set()  # Track attendance for current day
        
        # Date tracking
        self.current_date = datetime.date.today()
        
        # Frame processing variables
        self.frame_queue = Queue(maxsize=10)
        self.result_frame = None
        self.latest_attendance_info = None
        self.processing = False
        self.recent_attendance = []
        
        # Process every N frames for recognition
        self.recognition_interval = 10
        self.frame_count = 0
        
        # Create CSV file for today if it doesn't exist
        self._initialize_today_csv()
    
    def _initialize_today_csv(self):
        """Initialize CSV file for today's attendance"""
        today_str = self.current_date.strftime("%Y-%m-%d")
        filename = f"{self.config.class_name}_{today_str}.csv"
        filepath = os.path.join(self.config.data_directory, filename)
        
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['ID', 'Name', 'Date', 'Time', 'Status'])
    
    def _get_today_csv_path(self):
        """Get the path to today's CSV file"""
        today_str = self.current_date.strftime("%Y-%m-%d")
        filename = f"{self.config.class_name}_{today_str}.csv"
        return os.path.join(self.config.data_directory, filename)
    
    def _log_attendance(self, person_id, person_name, status="Present"):
        """Log attendance to CSV file"""
        now = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        with open(self._get_today_csv_path(), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([person_id, person_name, today_str, time_str, status])
        
        # Add to recent attendance list (limit to 20 entries)
        attendance_record = {
            "id": person_id,
            "name": person_name,
            "time": time_str,
            "status": status,
            "date": today_str
        }
        self.recent_attendance.insert(0, attendance_record)
        self.recent_attendance = self.recent_attendance[:20]
        
        logger.info(f"Logged attendance for {person_name} ({person_id}) at {time_str}")
        
        return attendance_record
    
    def start_processing(self):
        """Start the frame processing thread"""
        if not self.processing:
            self.processing = True
            self.process_thread = threading.Thread(target=self._process_frames)
            self.process_thread.daemon = True
            self.process_thread.start()
    
    def stop_processing(self):
        """Stop the frame processing thread"""
        self.processing = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)
    
    def _process_frames(self):
        """Process frames from the queue in a separate thread"""
        while self.processing:
            try:
                # Get a frame from the queue, wait up to 1 second
                frame = self.frame_queue.get(timeout=1.0)
                
                # Check if it's a new day
                today = datetime.date.today()
                if today != self.current_date:
                    self.current_date = today
                    self.current_day_attendance = set()
                    self._initialize_today_csv()
                
                # Increment frame counter
                self.frame_count += 1
                
                # Process for face detection every frame, but recognition only every N frames
                skip_recognition = (self.frame_count % self.recognition_interval != 0)
                
                # Process frame with face recognizer
                processed_frame, recognition_results = self.face_recognizer.process_frame(frame, skip_recognition)
                
                # Update result frame
                self.result_frame = processed_frame
                
                # If not skipping recognition, check for attendance
                if not skip_recognition and recognition_results:
                    # Check if inside attendance slot
                    is_attendance_time, slot_name = self.config.is_within_attendance_slot()
                    
                    # Process recognition results
                    for result in recognition_results:
                        person_name = result["person_name"]
                        person_id = self.config.get_or_create_id(person_name)
                        current_time = time.time()
                        
                        # Check if we're in an attendance slot
                        if is_attendance_time:
                            # Check if person has already been marked today or is within cooldown period
                            key = f"{person_id}_{slot_name}"
                            if key not in self.current_day_attendance and (
                                key not in self.last_attendance_time or 
                                current_time - self.last_attendance_time.get(key, 0) > self.config.cooldown_period
                            ):
                                # Mark attendance
                                attendance_record = self._log_attendance(person_id, person_name)
                                self.current_day_attendance.add(key)
                                self.last_attendance_time[key] = current_time
                                
                                # Set latest attendance info for UI
                                self.latest_attendance_info = attendance_record
                        
                        # Even outside attendance time, update last seen time
                        self.last_attendance_time[person_id] = current_time
                
                # Mark the task as done
                self.frame_queue.task_done()
            
            except Empty:
                # No frames in queue, just continue
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
    
    def add_frame(self, frame):
        """Add a frame to the processing queue"""
        if self.frame_queue.full():
            # If queue is full, remove the oldest frame
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except Empty:
                pass
        
        # Add new frame to queue
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except:
            return False
    
    def get_recent_attendance(self):
        """Get the recent attendance records"""
        return self.recent_attendance
    
    def clear_recent_attendance(self):
        """Clear the recent attendance records"""
        self.recent_attendance = []
    
    def generate_attendance_report(self, start_date=None, end_date=None):
        """Generate attendance report for a date range"""
        if start_date is None:
            # Default to start of current month
            today = datetime.date.today()
            start_date = today.replace(day=1)
        
        if end_date is None:
            # Default to today
            end_date = datetime.date.today()
        
        # Load all CSV files in the date range
        all_attendance = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            filename = f"{self.config.class_name}_{date_str}.csv"
            filepath = os.path.join(self.config.data_directory, filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                all_attendance.append(df)
            
            current_date += datetime.timedelta(days=1)
        
        if not all_attendance:
            return pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time', 'Status'])
        
        # Combine all data
        attendance_df = pd.concat(all_attendance, ignore_index=True)
        return attendance_df
    
    def get_attendance_stats(self, report_df):
        """Calculate attendance statistics from report dataframe"""
        if report_df.empty:
            return {
                "total_days": 0,
                "attendance_by_student": {},
                "daily_attendance": {},
                "students": []
            }
        
        # Convert Date to datetime
        report_df['Date'] = pd.to_datetime(report_df['Date'])
        
        # Get unique dates and student IDs
        dates = report_df['Date'].dt.date.unique()
        students = report_df['ID'].unique()
        
        # Calculate attendance for each student
        attendance_by_student = {}
        
        for student_id in students:
            student_df = report_df[report_df['ID'] == student_id]
            student_name = student_df['Name'].iloc[0]
            
            # Count days present
            days_present = len(student_df['Date'].dt.date.unique())
            attendance_rate = days_present / len(dates) * 100
            
            # Store stats
            attendance_by_student[student_id] = {
                "name": student_name,
                "days_present": days_present,
                "total_days": len(dates),
                "attendance_rate": attendance_rate
            }
        
        # Calculate daily attendance count
        daily_attendance = {}
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            daily_count = len(report_df[report_df['Date'].dt.date == date]['ID'].unique())
            daily_attendance[date_str] = daily_count
        
        return {
            "total_days": len(dates),
            "attendance_by_student": attendance_by_student,
            "daily_attendance": daily_attendance,
            "students": students.tolist()
        }

    def get_config(self):
        """Get the current configuration as a dict"""
        return {
            "class_name": self.config.class_name,
            "attendance_slots": self.config.attendance_slots,
            "cooldown_period": self.config.cooldown_period,
        }
    
    def update_config(self, config_data):
        """Update the configuration"""
        if 'class_name' in config_data:
            self.config.class_name = config_data['class_name']
        
        if 'attendance_slots' in config_data:
            self.config.attendance_slots = config_data['attendance_slots']
        
        if 'cooldown_period' in config_data:
            self.config.cooldown_period = int(config_data['cooldown_period'])
        
        # Re-initialize today's CSV with new class name if needed
        self._initialize_today_csv()
        
        # Save ID mapping
        self.config.save_id_mapping()
        
        return self.get_config()

# Flask application
app = Flask(__name__, static_folder='static')

# Initialize attendance system
attendance_system = None
camera = None

# Camera feed processing
def generate_frames():
    global camera
    
    if camera is None:
        camera = cv2.VideoCapture(attendance_system.camera_source)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Add frame to processing queue
            attendance_system.add_frame(frame)
            
            # Get the processed frame if available
            if attendance_system.result_frame is not None:
                display_frame = attendance_system.result_frame
            else:
                display_frame = frame
            
            # Resize frame for more efficient streaming
            display_frame = cv2.resize(display_frame, (0, 0), fx=0.8, fy=0.8)
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in multipart response format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/attendance/recent', methods=['GET'])
def get_recent_attendance():
    """Get recent attendance records"""
    return jsonify({"attendance": attendance_system.get_recent_attendance()})

@app.route('/api/attendance/clear', methods=['POST'])
def clear_recent_attendance():
    """Clear recent attendance records"""
    attendance_system.clear_recent_attendance()
    return jsonify({"status": "success"})

@app.route('/api/report', methods=['POST'])
def generate_report():
    """Generate attendance report"""
    data = request.json
    try:
        start_date = datetime.datetime.strptime(data.get('start_date'), "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(data.get('end_date'), "%Y-%m-%d").date()
    except:
        # Default to current month
        today = datetime.date.today()
        start_date = today.replace(day=1)
        end_date = today
    
    # Generate report
    df = attendance_system.generate_attendance_report(start_date, end_date)
    stats = attendance_system.get_attendance_stats(df)
    
    # Convert to JSON-serializable format
    detailed_records = []
    for _, row in df.iterrows():
        detailed_records.append({
            "id": row['ID'],
            "name": row['Name'],
            "date": row['Date'],
            "time": row['Time'],
            "status": row['Status']
        })
    
    # Convert daily attendance keys to strings
    daily_attendance = {str(k): v for k, v in stats['daily_attendance'].items()}
    
    return jsonify({
        "summary": {
            "total_days": stats['total_days'],
            "total_students": len(stats['students']),
            "attendance_by_student": stats['attendance_by_student'],
            "daily_attendance": daily_attendance
        },
        "detailed": detailed_records
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    return jsonify(attendance_system.get_config())

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update system configuration"""
    config_data = request.json
    updated_config = attendance_system.update_config(config_data)
    return jsonify(updated_config)

def initialize_system(camera_source=0):
    """Initialize the attendance system"""
    global attendance_system
    
    # Create configuration
    config = AttendanceConfig()
    
    # Create attendance system
    attendance_system = AttendanceSystem(config, camera_source)
    
    # Start processing
    attendance_system.start_processing()
    
    return attendance_system

def create_templates_folder():
    """Create templates folder and index.html"""
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Attendance System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container-fluid">
        <nav class="navbar navbar-expand-lg navbar-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="#">
                    <i class="fas fa-user-check"></i> Smart Attendance
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link active" data-bs-toggle="tab" href="#live">Live</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" data-bs-toggle="tab" href="#reports">Reports</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" data-bs-toggle="tab" href="#settings">Settings</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>

        <div class="tab-content mt-3">
            <!-- Live Tab -->
            <div class="tab-pane fade show active" id="live">
                <div class="row">
                    <div class="col-lg-8">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Camera Feed</h5>
                            </div>
                            <div class="card-body text-center">
                                <div class="video-container">
                                    <img src="{{ url_for('video_feed') }}" class="img-fluid" alt="Camera Feed">
                                </div>
                                <div class="status-bar mt-2" id="status-bar">
                                    <span class="badge" id="status-badge">Waiting for attendance slot</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-4">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Recent Attendance</h5>
                                <button class="btn btn-sm btn-outline-danger float-end" id="clearRecentBtn">
                                    <i class="fas fa-trash"></i> Clear
                                </button>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped" id="recentAttendanceTable">
                                        <thead>
                                            <tr>
                                                <th>ID</th>
                                                <th>Name</th>
                                                <th>Time</th>
                                                <th>Status</th>
                                            </tr>
                                        </thead>
                                        <tbody id="recentAttendanceBody">
                                            <!-- Attendance entries will be added here dynamically -->
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Reports Tab -->
            <div class="tab-pane fade" id="reports">
                <div class="row mb-3">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Generate Report</h5>
                            </div>
                            <div class="card-body">
                                <form id="reportForm" class="row g-3">
                                    <div class="col-md-4">
                                        <label for="startDate" class="form-label">Start Date</label>
                                        <input type="date" class="form-control" id="startDate" required>
                                    </div>
                                    <div class="col-md-4">
                                        <label for="endDate" class="form-label">End Date</label>
                                        <input type="date" class="form-control" id="endDate" required>
                                    </div>
                                    <div class="col-md-4
                <div class="col-md-4">
                                        <label for="reportType" class="form-label">Report Type</label>
                                        <select class="form-control" id="reportType">
                                            <option value="summary">Summary</option>
                                            <option value="detailed">Detailed</option>
                                        </select>
                                    </div>
                                    <div class="col-12">
                                        <button type="submit" class="btn btn-primary">
                                            <i class="fas fa-file-alt"></i> Generate Report
                                        </button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Attendance Summary</h5>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <div class="d-flex justify-content-between">
                                        <span>Total Days: <strong id="totalDays">0</strong></span>
                                        <span>Total Students: <strong id="totalStudents">0</strong></span>
                                    </div>
                                </div>
                                <canvas id="attendanceChart" height="250"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Daily Attendance</h5>
                            </div>
                            <div class="card-body">
                                <canvas id="dailyChart" height="250"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-3">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Detailed Attendance Records</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped" id="detailedReportTable">
                                        <thead>
                                            <tr>
                                                <th>ID</th>
                                                <th>Name</th>
                                                <th>Date</th>
                                                <th>Time</th>
                                                <th>Status</th>
                                            </tr>
                                        </thead>
                                        <tbody id="detailedReportBody">
                                            <!-- Report entries will be added here dynamically -->
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Settings Tab -->
            <div class="tab-pane fade" id="settings">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">System Settings</h5>
                            </div>
                            <div class="card-body">
                                <form id="settingsForm">
                                    <div class="mb-3">
                                        <label for="className" class="form-label">Class/Course Name</label>
                                        <input type="text" class="form-control" id="className" required>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Attendance Slots</label>
                                        <div id="attendanceSlots">
                                            <!-- Attendance slots will be added here dynamically -->
                                        </div>
                                        <button type="button" class="btn btn-sm btn-outline-primary mt-2" id="addSlotBtn">
                                            <i class="fas fa-plus"></i> Add Slot
                                        </button>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label for="cooldownPeriod" class="form-label">Cooldown Period (seconds)</label>
                                        <input type="number" class="form-control" id="cooldownPeriod" min="0" required>
                                    </div>
                                    
                                    <button type="submit" class="btn btn-primary">
                                        <i class="fas fa-save"></i> Save Settings
                                    </button>
                                </form>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Add New Face</h5>
                            </div>
                            <div class="card-body">
                                <form id="addFaceForm">
                                    <div class="mb-3">
                                        <label for="personName" class="form-label">Person Name</label>
                                        <input type="text" class="form-control" id="personName" required>
                                    </div>
                                    <div class="mb-3">
                                        <button type="button" class="btn btn-primary" id="captureFaceBtn">
                                            <i class="fas fa-camera"></i> Capture Face
                                        </button>
                                    </div>
                                    <div class="mb-3 d-none" id="capturedImageContainer">
                                        <label class="form-label">Captured Image</label>
                                        <div class="text-center">
                                            <img id="capturedImage" class="img-fluid mb-2" alt="Captured Face">
                                        </div>
                                        <button type="submit" class="btn btn-success">
                                            <i class="fas fa-save"></i> Save Face
                                        </button>
                                        <button type="button" class="btn btn-secondary" id="recaptureBtn">
                                            <i class="fas fa-redo"></i> Recapture
                                        </button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="/static/js/main.js"></script>
</body>
</html>
        ''')

def create_static_folder():
    """Create static folder with CSS and JS files"""
    # Create folder structure
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Create CSS file
    with open('static/css/style.css', 'w') as f:
        f.write('''
body {
    background-color: #f5f8fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.navbar {
    background-color: #3f51b5;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.navbar-brand i {
    margin-right: 8px;
}

.card {
    border: none;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.card-header {
    background-color: #fff;
    border-bottom: 1px solid #eaeaea;
    padding: 15px 20px;
    border-radius: 10px 10px 0 0 !important;
}

.card-body {
    padding: 20px;
}

.video-container {
    background-color: #000;
    border-radius: 8px;
    overflow: hidden;
    max-height: 480px;
}

.video-container img {
    max-height: 480px;
    width: 100%;
    object-fit: contain;
}

.status-bar {
    padding: 8px;
    background-color: #f8f9fa;
    border-radius: 5px;
}

.status-bar .badge {
    padding: 8px 12px;
    border-radius: 30px;
    background-color: #6c757d;
}

table {
    font-size: 0.9rem;
}

.table-striped tbody tr:nth-of-type(odd) {
    background-color: rgba(0, 0, 0, 0.02);
}

.attendance-slot {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
}

.btn-close-slot {
    color: #dc3545;
    cursor: pointer;
}

#capturedImage {
    max-width: 100%;
    max-height: 200px;
    border-radius: 8px;
    border: 1px solid #ddd;
}

/* Present status */
.status-present {
    background-color: #28a745;
    color: white;
}

/* Late status */
.status-late {
    background-color: #ffc107;
    color: #212529;
}

/* Absent status */
.status-absent {
    background-color: #dc3545;
    color: white;
}
        ''')
    
    # Create JS file
    with open('static/js/main.js', 'w') as f:
        f.write('''
document.addEventListener('DOMContentLoaded', function() {
    // Initialize date inputs with current date range (current month)
    const today = new Date();
    const firstDay = new Date(today.getFullYear(), today.getMonth(), 1);
    
    document.getElementById('startDate').valueAsDate = firstDay;
    document.getElementById('endDate').valueAsDate = today;
    
    // Load recent attendance immediately and every 5 seconds
    loadRecentAttendance();
    setInterval(loadRecentAttendance, 5000);
    
    // Load settings on page load
    loadSettings();
    
    // Set up event listeners
    document.getElementById('clearRecentBtn').addEventListener('click', clearRecentAttendance);
    document.getElementById('reportForm').addEventListener('submit', generateReport);
    document.getElementById('settingsForm').addEventListener('submit', saveSettings);
    document.getElementById('addSlotBtn').addEventListener('click', addAttendanceSlot);
    document.getElementById('captureFaceBtn').addEventListener('click', captureFace);
    document.getElementById('recaptureBtn').addEventListener('click', recaptureFace);
    document.getElementById('addFaceForm').addEventListener('submit', saveFace);
    
    // Update status badge every second
    updateStatusBadge();
    setInterval(updateStatusBadge, 1000);
});

// Load recent attendance from API
function loadRecentAttendance() {
    fetch('/api/attendance/recent')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('recentAttendanceBody');
            tbody.innerHTML = '';
            
            if (data.attendance && data.attendance.length > 0) {
                data.attendance.forEach(record => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${record.id}</td>
                        <td>${record.name}</td>
                        <td>${record.time}</td>
                        <td><span class="badge status-${record.status.toLowerCase()}">${record.status}</span></td>
                    `;
                    tbody.appendChild(row);
                });
            } else {
                const row = document.createElement('tr');
                row.innerHTML = '<td colspan="4" class="text-center">No recent attendance records</td>';
                tbody.appendChild(row);
            }
        })
        .catch(error => console.error('Error loading recent attendance:', error));
}

// Clear recent attendance
function clearRecentAttendance() {
    fetch('/api/attendance/clear', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        loadRecentAttendance();
    })
    .catch(error => console.error('Error clearing attendance:', error));
}

// Generate attendance report
function generateReport(e) {
    e.preventDefault();
    
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const reportType = document.getElementById('reportType').value;
    
    fetch('/api/report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            start_date: startDate,
            end_date: endDate,
            report_type: reportType
        })
    })
    .then(response => response.json())
    .then(data => {
        // Update summary information
        document.getElementById('totalDays').textContent = data.summary.total_days;
        document.getElementById('totalStudents').textContent = data.summary.total_students;
        
        // Generate attendance chart
        generateAttendanceChart(data.summary.attendance_by_student);
        
        // Generate daily attendance chart
        generateDailyChart(data.summary.daily_attendance);
        
        // Populate detailed table if applicable
        populateDetailedTable(data.detailed);
    })
    .catch(error => console.error('Error generating report:', error));
}

// Generate attendance chart
function generateAttendanceChart(attendanceData) {
    const ctx = document.getElementById('attendanceChart').getContext('2d');
    
    // Extract data for chart
    const labels = [];
    const attendanceRates = [];
    
    for (const studentId in attendanceData) {
        const student = attendanceData[studentId];
        labels.push(student.name);
        attendanceRates.push(student.attendance_rate);
    }
    
    // Destroy existing chart if any
    if (window.attendanceChart) {
        window.attendanceChart.destroy();
    }
    
    // Create new chart
    window.attendanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Attendance Rate (%)',
                data: attendanceRates,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Generate daily attendance chart
function generateDailyChart(dailyData) {
    const ctx = document.getElementById('dailyChart').getContext('2d');
    
    // Extract data for chart
    const dates = Object.keys(dailyData).sort();
    const counts = dates.map(date => dailyData[date]);
    
    // Destroy existing chart if any
    if (window.dailyChart) {
        window.dailyChart.destroy();
    }
    
    // Create new chart
    window.dailyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Student Count',
                data: counts,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2,
                tension: 0.1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            }
        }
    });
}

// Populate detailed attendance table
function populateDetailedTable(detailedData) {
    const tbody = document.getElementById('detailedReportBody');
    tbody.innerHTML = '';
    
    if (detailedData && detailedData.length > 0) {
        detailedData.forEach(record => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${record.id}</td>
                <td>${record.name}</td>
                <td>${record.date}</td>
                <td>${record.time}</td>
                <td><span class="badge status-${record.status.toLowerCase()}">${record.status}</span></td>
            `;
            tbody.appendChild(row);
        });
    } else {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="5" class="text-center">No attendance records found</td>';
        tbody.appendChild(row);
    }
}

// Load system settings
function loadSettings() {
    fetch('/api/config')
        .then(response => response.json())
        .then(data => {
            // Set class name
            document.getElementById('className').value = data.class_name;
            
            // Set cooldown period
            document.getElementById('cooldownPeriod').value = data.cooldown_period;
            
            // Add attendance slots
            const slotsContainer = document.getElementById('attendanceSlots');
            slotsContainer.innerHTML = '';
            
            for (const slotName in data.attendance_slots) {
                const slot = data.attendance_slots[slotName];
                addAttendanceSlotElement(slotName, slot.start, slot.end);
            }
        })
        .catch(error => console.error('Error loading settings:', error));
}

// Save system settings
function saveSettings(e) {
    e.preventDefault();
    
    // Get class name and cooldown period
    const className = document.getElementById('className').value;
    const cooldownPeriod = document.getElementById('cooldownPeriod').value;
    
    // Get attendance slots
    const slotElements = document.querySelectorAll('.attendance-slot');
    const attendanceSlots = {};
    
    slotElements.forEach(element => {
        const slotName = element.querySelector('.slot-name').value;
        const startTime = element.querySelector('.slot-start').value;
        const endTime = element.querySelector('.slot-end').value;
        
        attendanceSlots[slotName] = {
            start: startTime,
            end: endTime
        };
    });
    
    // Save settings
    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            class_name: className,
            attendance_slots: attendanceSlots,
            cooldown_period: cooldownPeriod
        })
    })
    .then(response => response.json())
    .then(data => {
        alert('Settings saved successfully!');
    })
    .catch(error => console.error('Error saving settings:', error));
}

// Add a new attendance slot field
function addAttendanceSlot() {
    const slotCount = document.querySelectorAll('.attendance-slot').length;
    const slotName = `Slot ${slotCount + 1}`;
    addAttendanceSlotElement(slotName, '09:00', '09:10');
}

// Add an attendance slot element to the UI
function addAttendanceSlotElement(name, startTime, endTime) {
    const slotsContainer = document.getElementById('attendanceSlots');
    
    const slotDiv = document.createElement('div');
    slotDiv.className = 'attendance-slot';
    slotDiv.innerHTML = `
        <div class="row g-2">
            <div class="col-md-4">
                <input type="text" class="form-control slot-name" value="${name}" placeholder="Slot Name" required>
            </div>
            <div class="col-md-3">
                <input type="time" class="form-control slot-start" value="${startTime}" required>
            </div>
            <div class="col-md-3">
                <input type="time" class="form-control slot-end" value="${endTime}" required>
            </div>
            <div class="col-md-2">
                <button type="button" class="btn btn-outline-danger btn-sm remove-slot" title="Remove Slot">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
    `;
    
    // Add remove button functionality
    slotDiv.querySelector('.remove-slot').addEventListener('click', function() {
        slotDiv.remove();
    });
    
    slotsContainer.appendChild(slotDiv);
}

// Capture face for registration
function captureFace() {
    const personName = document.getElementById('personName').value;
    
    if (!personName) {
        alert('Please enter a name first');
        return;
    }
    
    // Get current frame from video feed
    fetch('/api/capture_face', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show captured image
            document.getElementById('capturedImage').src = 'data:image/jpeg;base64,' + data.image;
            document.getElementById('capturedImageContainer').classList.remove('d-none');
        } else {
            alert('Failed to capture face: ' + data.message);
        }
    })
    .catch(error => console.error('Error capturing face:', error));
}

// Recapture face
function recaptureFace() {
    document.getElementById('capturedImageContainer').classList.add('d-none');
}

// Save captured face
function saveFace(e) {
    e.preventDefault();
    
    const personName = document.getElementById('personName').value;
    const imageData = document.getElementById('capturedImage').src.split(',')[1];
    
    fetch('/api/add_face', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name: personName,
            image: imageData
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Face saved successfully!');
            document.getElementById('capturedImageContainer').classList.add('d-none');
            document.getElementById('personName').value = '';
        } else {
            alert('Failed to save face: ' + data.message);
        }
    })
    .catch(error => console.error('Error saving face:', error));
}

// Update status badge
function updateStatusBadge() {
    const statusBadge = document.getElementById('status-badge');
    const now = new Date();
    const currentHour = now.getHours();
    const currentMinute = now.getMinutes();
    
    // Check if we're in attendance slot (this is just for UI display)
    // Actual tracking is handled by the backend
    fetch('/api/config')
        .then(response => response.json())
        .then(data => {
            let inAttendanceSlot = false;
            let slotName = '';
            
            for (const name in data.attendance_slots) {
                const slot = data.attendance_slots[name];
                const [startHour, startMinute] = slot.start.split(':').map(Number);
                const [endHour, endMinute] = slot.end.split(':').map(Number);
                
                if ((currentHour > startHour || (currentHour === startHour && currentMinute >= startMinute)) &&
                    (currentHour < endHour || (currentHour === endHour && currentMinute <= endMinute))) {
                    inAttendanceSlot = true;
                    slotName = name;
                    break;
                }
            }
            
            if (inAttendanceSlot) {
                statusBadge.textContent = `Taking Attendance (${slotName})`;
                statusBadge.className = 'badge bg-success';
            } else {
                statusBadge.textContent = 'Waiting for attendance slot';
                statusBadge.className = 'badge bg-secondary';
            }
        })
        .catch(error => console.error('Error checking status:', error));
}
        ''')

# Add missing endpoints
@app.route('/api/capture_face', methods=['POST'])
def capture_face():
    """Capture the current frame for face registration"""
    if not attendance_system or not camera:
        return jsonify({"success": False, "message": "Camera not initialized"})
    
    # Capture a frame
    success, frame = camera.read()
    if not success:
        return jsonify({"success": False, "message": "Failed to capture frame"})
    
    # Detect faces
    faces = attendance_system.face_recognizer.detect_faces(frame)
    
    if not faces:
        return jsonify({"success": False, "message": "No face detected"})
    
    if len(faces) > 1:
        return jsonify({"success": False, "message": "Multiple faces detected. Please ensure only one face is in the frame."})
    
    # Convert to base64 for sending to client
    _, buffer = cv2.imencode('.jpg', frame)
    image_data = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "success": True,
        "image": image_data
    })

@app.route('/api/add_face', methods=['POST'])
def add_face():
    """Add a new face to the system"""
    data = request.json
    
    if not data or 'name' not in data or 'image' not in data:
        return jsonify({"success": False, "message": "Missing required data"})
    
    name = data['name']
    image_data = data['image']
    
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Add face to recognizer
    success, message = attendance_system.face_recognizer.add_new_face(frame, name)
    
    return jsonify({
        "success": success,
        "message": message
    })

def main():
    """Main function to run the system"""
    # Create necessary folders
    create_templates_folder()
    create_static_folder()
    
    # Initialize the system
    initialize_system()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

if __name__ == '__main__':
    main()