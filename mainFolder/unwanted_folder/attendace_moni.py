import cv2
import numpy as np
import pandas as pd
import os
import json
import datetime
import time
import calendar
import pygame
import threading
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
import csv

# Import face_recognizer module (assuming it's in the same directory)
from face_recognizer import FaceRecognizer

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

class AttendanceSystem:
    """Main attendance system using face recognition"""
    def __init__(self, config=None, camera_source="rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101"):
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
        
        # Initialize audio feedback
        if self.config.voice_feedback:
            pygame.mixer.init()
            self.audio_thread = None
        
        # Date tracking
        self.current_date = datetime.date.today()
        
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
        
        logger.info(f"Logged attendance for {person_name} ({person_id}) at {time_str}")
    
    def _play_audio_greeting(self, person_name):
        """Play audio greeting for the recognized person"""
        if self.config.voice_feedback:
            import pyttsx3
            
            def speak():
                engine = pyttsx3.init()
                engine.say(f"Hello, {person_name}. You are marked present.")
                engine.runAndWait()
            
            # Run in a separate thread to avoid blocking
            if self.audio_thread is None or not self.audio_thread.is_alive():
                self.audio_thread = threading.Thread(target=speak)
                self.audio_thread.daemon = True
                self.audio_thread.start()
    
    def process_attendance(self, frame):
        """Process a video frame for attendance"""
        # Check if it's a new day
        today = datetime.date.today()
        if today != self.current_date:
            self.current_date = today
            self.current_day_attendance = set()
            self._initialize_today_csv()
        
        # Process frame with face recognizer
        processed_frame, recognition_results = self.face_recognizer.process_frame(frame)
        
        # Check if inside attendance slot
        is_attendance_time, slot_name = self.config.is_within_attendance_slot()
        
        # Process recognition results
        attendance_info = None
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
                    self._log_attendance(person_id, person_name)
                    self.current_day_attendance.add(key)
                    self.last_attendance_time[key] = current_time
                    
                    # Play audio greeting
                    self._play_audio_greeting(person_name)
                    
                    # Return attendance info for UI
                    attendance_info = {
                        "name": person_name,
                        "id": person_id,
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "status": "Present"
                    }
            
            # Even outside attendance time, update last seen time
            self.last_attendance_time[person_id] = current_time
        
        return processed_frame, attendance_info
    
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
                "punctuality_by_student": {},
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
        punctuality_by_student = {}
        
        for student_id in students:
            student_df = report_df[report_df['ID'] == student_id]
            student_name = student_df['Name'].iloc[0]
            
            # Count days present
            days_present = len(student_df['Date'].dt.date.unique())
            attendance_rate = days_present / len(dates) * 100
            
            # Calculate average time
            student_df['TimeObj'] = pd.to_datetime(student_df['Time'])
            avg_time = student_df['TimeObj'].dt.time.mean()
            
            # Store stats
            attendance_by_student[student_id] = {
                "name": student_name,
                "days_present": days_present,
                "total_days": len(dates),
                "attendance_rate": attendance_rate
            }
            
            if avg_time:
                punctuality_by_student[student_id] = {
                    "name": student_name,
                    "avg_time": avg_time
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
            "punctuality_by_student": punctuality_by_student,
            "daily_attendance": daily_attendance,
            "students": students.tolist()
        }

class AttendanceUI:
    """UI for the attendance system"""
    def __init__(self, attendance_system):
        self.attendance_system = attendance_system
        self.cap = None
        self.is_running = False
        self.after_id = None
        
        # Initialize the UI
        self.root = tk.Tk()
        self.root.title("Smart Attendance System")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_live_tab()
        self.create_reports_tab()
        self.create_settings_tab()
        
        # Setup Styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))
        self.style.configure("Info.TLabel", font=("Helvetica", 12))
        
        # Start the camera
        self.start_camera()
    
    def create_live_tab(self):
        """Create the live attendance tab"""
        self.live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text="Live Attendance")
        
        # Split into left (video) and right (info) panels
        self.left_panel = ttk.Frame(self.live_tab)
        self.right_panel = ttk.Frame(self.live_tab)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.right_panel = ttk.Frame(self.live_tab, width=300)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5, expand=False)
        self.right_panel.pack_propagate(False)
        
        # Video display
        self.video_label = ttk.Label(self.left_panel)
        self.video_label.pack(pady=10)
        
        # Status label
        self.status_frame = ttk.Frame(self.left_panel)
        self.status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Status: Waiting for attendance slot", 
                                      font=("Helvetica", 10))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Right panel - Recent attendance
        ttk.Label(self.right_panel, text="Recent Attendance", 
                 style="Header.TLabel").pack(pady=10)
        
        # Attendance log
        self.attendance_frame = ttk.Frame(self.right_panel)
        self.attendance_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.attendance_tree = ttk.Treeview(self.attendance_frame, columns=("id", "name", "time", "status"),
                                          show="headings", height=15)
        self.attendance_tree.heading("id", text="ID")
        self.attendance_tree.heading("name", text="Name")
        self.attendance_tree.heading("time", text="Time")
        self.attendance_tree.heading("status", text="Status")
        
        self.attendance_tree.column("id", width=70)
        self.attendance_tree.column("name", width=120)
        self.attendance_tree.column("time", width=70)
        self.attendance_tree.column("status", width=70)
        
        scrollbar = ttk.Scrollbar(self.attendance_frame, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscroll=scrollbar.set)
        
        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        self.control_frame = ttk.Frame(self.right_panel)
        self.control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_camera)
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_camera)
        
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.pack(side=tk.LEFT, padx=5)
    
    def create_reports_tab(self):
        """Create the reports tab"""
        self.reports_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.reports_tab, text="Reports")
        
        # Date range selection
        date_frame = ttk.Frame(self.reports_tab)
        date_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(date_frame, text="From:").pack(side=tk.LEFT, padx=5)
        self.from_date = ttk.Entry(date_frame, width=12)
        self.from_date.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(date_frame, text="To:").pack(side=tk.LEFT, padx=5)
        self.to_date = ttk.Entry(date_frame, width=12)
        self.to_date.pack(side=tk.LEFT, padx=5)
        
        # Set default dates (current month)
        today = datetime.date.today()
        first_day = today.replace(day=1)
        self.from_date.insert(0, first_day.strftime("%Y-%m-%d"))
        self.to_date.insert(0, today.strftime("%Y-%m-%d"))
        
        generate_button = ttk.Button(date_frame, text="Generate Report", command=self.generate_report)
        generate_button.pack(side=tk.LEFT, padx=20)
        
        # Tabs for different report views
        self.report_notebook = ttk.Notebook(self.reports_tab)
        self.report_notebook.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Summary tab
        self.summary_tab = ttk.Frame(self.report_notebook)
        self.report_notebook.add(self.summary_tab, text="Summary")
        
        # Detailed tab
        self.detailed_tab = ttk.Frame(self.report_notebook)
        self.report_notebook.add(self.detailed_tab, text="Detailed")
        
        # Charts tab
        self.charts_tab = ttk.Frame(self.report_notebook)
        self.report_notebook.add(self.charts_tab, text="Charts")
        
        # Add a table view for detailed data
        self.detailed_tree = ttk.Treeview(self.detailed_tab, 
                                        columns=("id", "name", "date", "time", "status"),
                                        show="headings")
        self.detailed_tree.heading("id", text="ID")
        self.detailed_tree.heading("name", text="Name")
        self.detailed_tree.heading("date", text="Date")
        self.detailed_tree.heading("time", text="Time")
        self.detailed_tree.heading("status", text="Status")
        
        self.detailed_tree.column("id", width=70)
        self.detailed_tree.column("name", width=120)
        self.detailed_tree.column("date", width=100)
        self.detailed_tree.column("time", width=80)
        self.detailed_tree.column("status", width=80)
        
        scrollbar = ttk.Scrollbar(self.detailed_tab, orient=tk.VERTICAL, command=self.detailed_tree.yview)
        self.detailed_tree.configure(yscroll=scrollbar.set)
        
        self.detailed_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_settings_tab(self):
        """Create the settings tab"""
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Class settings
        class_frame = ttk.LabelFrame(self.settings_tab, text="Class Settings")
        class_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(class_frame, text="Class Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.class_name_var = tk.StringVar(value=self.attendance_system.config.class_name)
        ttk.Entry(class_frame, textvariable=self.class_name_var, width=25).grid(row=0, column=1, padx=5, pady=5)
        
        # Attendance slots
        slots_frame = ttk.LabelFrame(self.settings_tab, text="Attendance Slots")
        slots_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Morning slot
        ttk.Label(slots_frame, text="Morning:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.morning_start_var = tk.StringVar(value=self.attendance_system.config.attendance_slots["Morning"]["start"])
        self.morning_end_var = tk.StringVar(value=self.attendance_system.config.attendance_slots["Morning"]["end"])
        
        ttk.Entry(slots_frame, textvariable=self.morning_start_var, width=8).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(slots_frame, text="to").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(slots_frame, textvariable=self.morning_end_var, width=8).grid(row=0, column=3, padx=5, pady=5)
        
        # Afternoon slot
        ttk.Label(slots_frame, text="Afternoon:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.afternoon_start_var = tk.StringVar(value=self.attendance_system.config.attendance_slots["Afternoon"]["start"])
        self.afternoon_end_var = tk.StringVar(value=self.attendance_system.config.attendance_slots["Afternoon"]["end"])
        
        ttk.Entry(slots_frame, textvariable=self.afternoon_start_var, width=8).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(slots_frame, text="to").grid(row=1, column=2, padx=5, pady=5)
        ttk.Entry(slots_frame, textvariable=self.afternoon_end_var, width=8).grid(row=1, column=3, padx=5, pady=5)
        
        # Other settings
        other_frame = ttk.LabelFrame(self.settings_tab, text="Other Settings")
        other_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(other_frame, text="Cooldown Period (seconds):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.cooldown_var = tk.IntVar(value=self.attendance_system.config.cooldown_period)
        ttk.Entry(other_frame, textvariable=self.cooldown_var, width=8).grid(row=0, column=1, padx=5, pady=5)
        
        self.voice_feedback_var = tk.BooleanVar(value=self.attendance_system.config.voice_feedback)
        ttk.Checkbutton(other_frame, text="Voice Feedback", variable=self.voice_feedback_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Save button
        ttk.Button(self.settings_tab, text="Save Settings", command=self.save_settings).pack(pady=20)
    
    def save_settings(self):
        """Save settings to config"""
        # Update class name
        self.attendance_system.config.class_name = self.class_name_var.get()
        
        # Update attendance slots
        self.attendance_system.config.attendance_slots = {
            "Morning": {
                "start": self.morning_start_var.get(),
                "end": self.morning_end_var.get()
            },
            "Afternoon": {
                "start": self.afternoon_start_var.get(),
                "end": self.afternoon_end_var.get()
            }
        }
        
        # Update other settings
        self.attendance_system.config.cooldown_period = self.cooldown_var.get()
        self.attendance_system.config.voice_feedback = self.voice_feedback_var.get()
        
        # Reinitialize today's CSV
        self.attendance_system._initialize_today_csv()
        
        messagebox.showinfo("Settings", "Settings saved successfully!")
    
    def start_camera(self):
        """Start the camera feed"""
        if self.is_running:
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.attendance_system.camera_source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera!")
            return
        
        self.is_running = True
        self.update_frame()
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.is_running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
    
    def update_frame(self):
        """Update the video frame"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Error: Failed to capture frame")
            self.after_id = self.root.after(100, self.update_frame)
            return
        
        # Process frame for attendance
        processed_frame, attendance_info = self.attendance_system.process_attendance(frame)
        
        # Update UI with attendance info
        if attendance_info:
            self.add_attendance_record(attendance_info)
        
        # Update status label
        is_attendance_time, slot_name = self.attendance_system.config.is_within_attendance_slot()
        if is_attendance_time:
            self.status_label.config(text=f"Status: Taking attendance for {slot_name} slot")
        else:
            self.status_label.config(text="Status: Outside attendance slot")
        
        # Convert processed frame to display in UI
        cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Schedule next frame update
        self.after_id = self.root.after(10, self.update_frame)
    
    def add_attendance_record(self, info):
        """Add a new attendance record to the live view"""
        self.attendance_tree.insert("", 0, values=(info["id"], info["name"], info["time"], info["status"]))
        
        # Limit the number of visible records
        records = self.attendance_tree.get_children()
        if len(records) > 20:
            self.attendance_tree.delete(records[-1])
    
    def generate_report(self):
        """Generate attendance report based on date range"""
        try:
            from_date = datetime.datetime.strptime(self.from_date.get(), "%Y-%m-%d").date()
            to_date = datetime.datetime.strptime(self.to_date.get(), "%Y-%m-%d").date()
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
            return
        
        # Get attendance data
        attendance_df = self.attendance_system.generate_attendance_report(from_date, to_date)
        if attendance_df.empty:
            messagebox.showinfo("Report", "No attendance data found for the selected period")
            return
        
        # Get statistics
        stats = self.attendance_system.get_attendance_stats(attendance_df)
        
        # Update summary tab
        self.update_summary_tab(stats)
        
        # Update detailed tab
        self.update_detailed_tab(attendance_df)
        
        # Update charts tab
        self.update_charts_tab(stats)
    
    def update_summary_tab(self, stats):
        """Update the summary tab with attendance statistics"""
        # Clear existing widgets
        for widget in self.summary_tab.winfo_children():
            widget.destroy()
        
        # Add summary information
        summary_frame = ttk.Frame(self.summary_tab)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(summary_frame, text=f"Total Days: {stats['total_days']}", 
                  style="Header.TLabel").pack(anchor=tk.W, pady=5)
        
        # Create treeview for student attendance rates
        ttk.Label(summary_frame, text="Attendance Rates by Student", 
                  style="Header.TLabel").pack(anchor=tk.W, pady=5)
        
        tree_frame = ttk.Frame(summary_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        attendance_tree = ttk.Treeview(tree_frame, 
                                    columns=("id", "name", "present", "total", "rate"),
                                    show="headings")
        attendance_tree.heading("id", text="ID")
        attendance_tree.heading("name", text="Name")
        attendance_tree.heading("present", text="Days Present")
        attendance_tree.heading("total", text="Total Days")
        attendance_tree.heading("rate", text="Rate (%)")
        
        attendance_tree.column("id", width=70)
        attendance_tree.column("name", width=120)
        attendance_tree.column("present", width=100)
        attendance_tree.column("total", width=100)
        attendance_tree.column("rate", width=80)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=attendance_tree.yview)
        attendance_tree.configure(yscroll=scrollbar.set)
        # Continuing from where the previous code left off

        attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate the tree
        for student_id, info in stats['attendance_by_student'].items():
            attendance_tree.insert("", tk.END, values=(
                student_id, 
                info['name'], 
                info['days_present'], 
                info['total_days'], 
                f"{info['attendance_rate']:.1f}"
            ))
    
    def update_detailed_tab(self, attendance_df):
        """Update the detailed tab with attendance records"""
        # Clear existing entries
        for item in self.detailed_tree.get_children():
            self.detailed_tree.delete(item)
        
        # Add new entries
        for _, row in attendance_df.iterrows():
            self.detailed_tree.insert("", tk.END, values=(
                row['ID'],
                row['Name'],
                row['Date'],
                row['Time'],
                row['Status']
            ))
    
    def update_charts_tab(self, stats):
        """Update the charts tab with visualization of attendance data"""
        # Clear existing widgets
        for widget in self.charts_tab.winfo_children():
            widget.destroy()
        
        # Create frame for charts
        charts_frame = ttk.Frame(self.charts_tab)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(10, 8))
        
        # Attendance rate by student
        ax1 = fig.add_subplot(211)
        
        # Extract data for plotting
        students = []
        rates = []
        for student_id, info in stats['attendance_by_student'].items():
            students.append(info['name'])
            rates.append(info['attendance_rate'])
        
        # Sort by attendance rate
        sorted_indices = np.argsort(rates)
        sorted_students = [students[i] for i in sorted_indices]
        sorted_rates = [rates[i] for i in sorted_indices]
        
        # Plot attendance rates
        bars = ax1.barh(sorted_students, sorted_rates, color='skyblue')
        ax1.set_xlabel('Attendance Rate (%)')
        ax1.set_title('Attendance Rate by Student')
        ax1.set_xlim(0, 100)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f"{sorted_rates[i]:.1f}%", 
                    va='center')
        
        # Daily attendance plot
        ax2 = fig.add_subplot(212)
        
        dates = []
        counts = []
        for date_str, count in stats['daily_attendance'].items():
            dates.append(date_str)
            counts.append(count)
        
        # Sort by date
        sorted_indices = np.argsort(dates)
        sorted_dates = [dates[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        # Plot daily attendance
        ax2.plot(sorted_dates, sorted_counts, marker='o', linestyle='-', color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Students Present')
        ax2.set_title('Daily Attendance Count')
        ax2.tick_params(axis='x', rotation=45)
        
        fig.tight_layout()
        
        # Add the figure to the UI
        canvas = FigureCanvasTkAgg(fig, master=charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_close(self):
        """Handle window close event"""
        self.stop_camera()
        self.root.destroy()
    
    def run(self):
        """Run the UI main loop"""
        self.root.mainloop()


class FaceRecognizer:
    """Face recognition implementation using OpenCV and face_recognition"""
    def __init__(self, embeddings_dir="embeddings", recognition_threshold=0.6, min_detection_confidence=0.5):
        self.embeddings_dir = embeddings_dir
        self.recognition_threshold = recognition_threshold
        self.min_detection_confidence = min_detection_confidence
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize face detector and recognition models
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
            import face_recognition
            
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Compute face encoding
            face_encodings = face_recognition.face_encodings(rgb_image)
            
            if face_encodings:
                return face_encodings[0]
            return None
        except ImportError:
            # Fallback if face_recognition is not available
            # In a real system, you might want to use a different model or show an error
            logger.error("face_recognition library not found. Face embedding can't be computed.")
            return None
    
    def recognize_face(self, face_embedding):
        """Recognize a face based on its embedding"""
        if not self.known_face_embeddings:
            return None
        
        try:
            import face_recognition
            
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_face_embeddings, face_embedding)
            
            # Find the closest match
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            # Check if the match is close enough
            if min_distance <= self.recognition_threshold:
                return self.known_face_names[min_distance_idx]
            
            return None
        except ImportError:
            logger.error("face_recognition library not found. Face recognition can't be performed.")
            return None
    
    def process_frame(self, frame):
        """Process a video frame for face detection and recognition"""
        # Make a copy of the frame
        processed_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Recognition results
        recognition_results = []
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_image = frame[y:y+h, x:x+w]
            
            # Compute face embedding
            face_embedding = self.compute_face_embedding(face_image)
            
            if face_embedding is not None:
                # Try to recognize the face
                person_name = self.recognize_face(face_embedding)
                
                if person_name:
                    # Draw rectangle and label
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(processed_frame, "Unknown", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
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


def main():
    """Main function to start the application"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='attendance_system.log'
    )
    
    # Create attendance system
    config = AttendanceConfig()
    system = AttendanceSystem(config)
    
    # Create and run the UI
    ui = AttendanceUI(system)
    ui.run()


if __name__ == "__main__":
    main()