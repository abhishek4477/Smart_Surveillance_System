import cv2
import numpy as np
import pandas as pd
import datetime
import os
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import time
import pyttsx3
import matplotlib.pyplot as plt
import seaborn as sns
from face_recognizer import FaceRecognizer
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttendanceManager:
    def __init__(self, 
                 class_name: str = "DefaultClass",
                 attendance_dir: str = "attendance_logs",
                 slot_start: tuple = (9, 0),  # (hour, minute)
                 slot_end: tuple = (9, 10),
                 late_threshold: tuple = (9, 5)):
        """
        Initialize attendance manager.
        
        Args:
            class_name: Name of the class for attendance
            attendance_dir: Directory to store attendance logs
            slot_start: Start time for attendance (hour, minute)
            slot_end: End time for attendance (hour, minute)
            late_threshold: Time after which entry is marked as late
        """
        self.class_name = class_name
        self.attendance_dir = Path(attendance_dir)
        self.attendance_dir.mkdir(parents=True, exist_ok=True)
        
        # Time slots
        self.slot_start = datetime.time(slot_start[0], slot_start[1])
        self.slot_end = datetime.time(slot_end[0], slot_end[1])
        self.late_threshold = datetime.time(late_threshold[0], late_threshold[1])
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Today's attendance cache
        self.today_attendance = set()
        self._load_today_attendance()
        
        # Initialize face recognizer
        self.recognizer = FaceRecognizer(
            embeddings_dir="embeddings",
            recognition_threshold=0.6,
            min_detection_confidence=0.6
        )
    
    def _load_today_attendance(self):
        """Load today's attendance from file to prevent duplicates."""
        today = datetime.date.today()
        file_path = self._get_attendance_file(today)
        if file_path.exists():
            df = pd.read_csv(file_path)
            self.today_attendance = set(df['ID'].tolist())
    
    def _get_attendance_file(self, date: datetime.date) -> Path:
        """Get the attendance file path for a specific date."""
        return self.attendance_dir / f"{self.class_name}_{date.strftime('%Y-%m-%d')}.csv"
    
    def _is_within_slot(self) -> bool:
        """Check if current time is within attendance slot."""
        current_time = datetime.datetime.now().time()
        return self.slot_start <= current_time <= self.slot_end
    
    def _get_status(self) -> str:
        """Get attendance status based on current time."""
        current_time = datetime.datetime.now().time()
        if current_time <= self.late_threshold:
            return "Present"
        elif current_time <= self.slot_end:
            return "Late"
        return "Absent"
    
    def mark_attendance(self, recognition_result: Dict) -> bool:
        """
        Mark attendance for a recognized person.
        
        Args:
            recognition_result: Dictionary containing recognition details
            
        Returns:
            bool: True if attendance was marked
        """
        try:
            cluster_id = recognition_result["cluster_id"]
            name = recognition_result["person_name"]
            
            # Check if already marked for today
            if cluster_id in self.today_attendance:
                return False
            
            # Check if within attendance slot
            if not self._is_within_slot():
                logger.info(f"Outside attendance slot ({self.slot_start}-{self.slot_end})")
                return False
            
            # Get current timestamp
            now = datetime.datetime.now()
            status = self._get_status()
            
            # Create attendance record
            record = {
                'ID': cluster_id,
                'Name': name,
                'Date': now.strftime('%Y-%m-%d'),
                'Time': now.strftime('%H:%M:%S'),
                'Status': status
            }
            
            # Save to CSV
            file_path = self._get_attendance_file(now.date())
            df = pd.DataFrame([record])
            if file_path.exists():
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, index=False)
            
            # Update cache
            self.today_attendance.add(cluster_id)
            
            # Voice confirmation
            self.engine.say(f"Hello {name}. You're marked {status}")
            self.engine.runAndWait()
            
            logger.info(f"Marked {status} for {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error marking attendance: {str(e)}")
            return False
    
    def get_attendance_stats(self, start_date: datetime.date = None, 
                           end_date: datetime.date = None) -> pd.DataFrame:
        """Get attendance statistics for a date range."""
        if start_date is None:
            start_date = datetime.date.today() - datetime.timedelta(days=30)
        if end_date is None:
            end_date = datetime.date.today()
        
        # Collect all attendance records
        dfs = []
        current_date = start_date
        while current_date <= end_date:
            file_path = self._get_attendance_file(current_date)
            if file_path.exists():
                df = pd.read_csv(file_path)
                dfs.append(df)
            current_date += datetime.timedelta(days=1)
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    def generate_attendance_report(self, output_file: str = None):
        """Generate attendance report with visualizations."""
        stats = self.get_attendance_stats()
        if stats.empty:
            logger.warning("No attendance data available")
            return
        
        # Create report
        fig = plt.figure(figsize=(15, 10))
        
        # Attendance status distribution
        plt.subplot(2, 2, 1)
        status_counts = stats['Status'].value_counts()
        plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
        plt.title('Attendance Status Distribution')
        
        # Daily attendance trend
        plt.subplot(2, 2, 2)
        daily_counts = stats.groupby('Date').size()
        plt.plot(daily_counts.index, daily_counts.values)
        plt.xticks(rotation=45)
        plt.title('Daily Attendance Trend')
        
        # Student attendance frequency
        plt.subplot(2, 2, 3)
        student_freq = stats.groupby('Name').size().sort_values(ascending=False)
        sns.barplot(x=student_freq.index, y=student_freq.values)
        plt.xticks(rotation=45)
        plt.title('Student Attendance Frequency')
        
        # Time distribution
        plt.subplot(2, 2, 4)
        stats['Hour'] = pd.to_datetime(stats['Time']).dt.hour
        time_dist = stats['Hour'].value_counts().sort_index()
        plt.bar(time_dist.index, time_dist.values)
        plt.title('Attendance Time Distribution')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        
        return fig

class AttendanceUI:
    def __init__(self, window: tk.Toplevel, attendance_manager: AttendanceManager):
        """Initialize attendance system UI."""
        self.attendance_manager = attendance_manager
        self.is_running = False
        
        # Set main window
        self.root = window
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.monitoring_tab = ttk.Frame(self.notebook)
        self.attendance_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.monitoring_tab, text="Monitoring")
        self.notebook.add(self.attendance_tab, text="Attendance Data")
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Setup each tab
        self._setup_monitoring_tab()
        self._setup_attendance_tab()
        self._setup_settings_tab()
        
        # Initialize video capture
        self.cap = None
        
        # Update time
        self.update_time()
        self.update_attendance_table()
    
    def _setup_monitoring_tab(self):
        """Setup the monitoring tab with video feed and controls."""
        # Create main frame
        main_frame = ttk.Frame(self.monitoring_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create video label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(pady=5)
        
        # Create status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create status message with larger font and bold
        self.status_message = ttk.Label(status_frame, 
                                      text="System Status: Ready", 
                                      font=('Arial', 14, 'bold'))
        self.status_message.pack(side=tk.LEFT, pady=5)
        
        # Create time label
        self.time_label = ttk.Label(status_frame, text="", font=('Arial', 12))
        self.time_label.pack(side=tk.RIGHT, pady=5)
        
        # Create control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create buttons
        self.start_button = ttk.Button(control_frame, text="Start Monitoring", 
                                     command=self.start_attendance)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Monitoring", 
                                    command=self.stop_attendance, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = ttk.Button(control_frame, text="Exit Application", 
                                    command=self.exit_application)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
    
    def _setup_attendance_tab(self):
        """Setup the attendance data tab with table and controls."""
        # Create main frame
        main_frame = ttk.Frame(self.attendance_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add report button
        self.report_button = ttk.Button(control_frame, text="Generate Report", 
                                      command=self.show_report)
        self.report_button.pack(side=tk.LEFT, padx=5)
        
        # Create table frame
        table_frame = ttk.LabelFrame(main_frame, text="Today's Attendance")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview
        self.attendance_tree = ttk.Treeview(table_frame, 
                                          columns=("ID", "Name", "Time", "Status"),
                                          show="headings")
        
        # Define columns
        self.attendance_tree.heading("ID", text="ID")
        self.attendance_tree.heading("Name", text="Name")
        self.attendance_tree.heading("Time", text="Time")
        self.attendance_tree.heading("Status", text="Status")
        
        # Set column widths
        self.attendance_tree.column("ID", width=100)
        self.attendance_tree.column("Name", width=200)
        self.attendance_tree.column("Time", width=150)
        self.attendance_tree.column("Status", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, 
                                command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack table and scrollbar
        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_settings_tab(self):
        """Setup the settings tab with parameter controls."""
        # Create main frame
        main_frame = ttk.Frame(self.settings_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create time settings frame
        time_frame = ttk.LabelFrame(main_frame, text="Attendance Time Settings")
        time_frame.pack(fill=tk.X, pady=10)
        
        # Start time settings
        start_frame = ttk.Frame(time_frame)
        start_frame.pack(fill=tk.X, pady=5)
        ttk.Label(start_frame, text="Start Time:").pack(side=tk.LEFT, padx=5)
        
        self.start_hour = ttk.Spinbox(start_frame, from_=0, to=23, width=5)
        self.start_hour.pack(side=tk.LEFT, padx=5)
        self.start_hour.set(self.attendance_manager.slot_start.hour)
        
        ttk.Label(start_frame, text=":").pack(side=tk.LEFT)
        
        self.start_minute = ttk.Spinbox(start_frame, from_=0, to=59, width=5)
        self.start_minute.pack(side=tk.LEFT, padx=5)
        self.start_minute.set(self.attendance_manager.slot_start.minute)
        
        # End time settings
        end_frame = ttk.Frame(time_frame)
        end_frame.pack(fill=tk.X, pady=5)
        ttk.Label(end_frame, text="End Time:  ").pack(side=tk.LEFT, padx=5)
        
        self.end_hour = ttk.Spinbox(end_frame, from_=0, to=23, width=5)
        self.end_hour.pack(side=tk.LEFT, padx=5)
        self.end_hour.set(self.attendance_manager.slot_end.hour)
        
        ttk.Label(end_frame, text=":").pack(side=tk.LEFT)
        
        self.end_minute = ttk.Spinbox(end_frame, from_=0, to=59, width=5)
        self.end_minute.pack(side=tk.LEFT, padx=5)
        self.end_minute.set(self.attendance_manager.slot_end.minute)
        
        # Late threshold settings
        late_frame = ttk.Frame(time_frame)
        late_frame.pack(fill=tk.X, pady=5)
        ttk.Label(late_frame, text="Late After:").pack(side=tk.LEFT, padx=5)
        
        self.late_hour = ttk.Spinbox(late_frame, from_=0, to=23, width=5)
        self.late_hour.pack(side=tk.LEFT, padx=5)
        self.late_hour.set(self.attendance_manager.late_threshold.hour)
        
        ttk.Label(late_frame, text=":").pack(side=tk.LEFT)
        
        self.late_minute = ttk.Spinbox(late_frame, from_=0, to=59, width=5)
        self.late_minute.pack(side=tk.LEFT, padx=5)
        self.late_minute.set(self.attendance_manager.late_threshold.minute)
        
        # Save button
        save_frame = ttk.Frame(time_frame)
        save_frame.pack(fill=tk.X, pady=10)
        ttk.Button(save_frame, text="Save Settings", 
                  command=self.save_settings).pack(pady=5)
    
    def save_settings(self):
        """Save the time settings."""
        try:
            # Update attendance manager time settings
            self.attendance_manager.slot_start = datetime.time(
                int(self.start_hour.get()),
                int(self.start_minute.get())
            )
            self.attendance_manager.slot_end = datetime.time(
                int(self.end_hour.get()),
                int(self.end_minute.get())
            )
            self.attendance_manager.late_threshold = datetime.time(
                int(self.late_hour.get()),
                int(self.late_minute.get())
            )
            
            messagebox.showinfo("Success", "Settings saved successfully!")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid time values: {str(e)}")
    
    def exit_application(self):
        """Safely exit the application."""
        if self.is_running:
            self.stop_attendance()
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()
    
    def start_attendance(self):
        """Start attendance system."""
        if not self.is_running:
            self.cap = cv2.VideoCapture("rtsp://admin:Smart2025@192.168.1.100:554/Streaming/Channels/101")
            if not self.cap.isOpened():
                self.status_message.config(text="System Status: Failed to open camera")
                return
            
            self.is_running = True
            self.status_message.config(text="System Status: Monitoring Active")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.update_frame()
    
    def stop_attendance(self):
        """Stop attendance system."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.status_message.config(text="System Status: Monitoring Stopped")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.video_label.configure(image='')
    
    def update_time(self):
        """Update time display."""
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
    
    def update_attendance_table(self):
        """Update attendance table with today's data."""
        # Clear existing items
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # Get today's attendance file
        today = datetime.date.today()
        file_path = self.attendance_manager._get_attendance_file(today)
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                self.attendance_tree.insert("", tk.END, values=(
                    row['ID'],
                    row['Name'],
                    row['Time'],
                    row['Status']
                ))
        
        # Schedule next update
        self.root.after(5000, self.update_attendance_table)  # Update every 5 seconds
    
    def update_frame(self):
        """Update video frame and process attendance."""
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Process frame
                processed_frame, results = self.attendance_manager.recognizer.process_frame(frame)
                
                # Mark attendance for recognized faces
                for result in results:
                    if self.attendance_manager.mark_attendance(result):
                        name = result["person_name"]
                        self.status_message.config(text=f"System Status: Marked attendance for {name}")
                
                # Convert frame for display
                cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            if self.is_running:  # Check again to prevent extra calls after stopping
                self.root.after(10, self.update_frame)
    
    def show_report(self):
        """Show attendance report in a new window."""
        report_window = tk.Toplevel(self.root)
        report_window.title("Attendance Report")
        report_window.geometry("800x600")
        
        # Generate report
        fig = self.attendance_manager.generate_attendance_report()
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=report_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run(self):
        """Run the UI."""
        self.root.mainloop()

def main():
    # Initialize attendance manager
    attendance_manager = AttendanceManager(
        class_name="Class1",
        slot_start=(9, 0),
        slot_end=(9, 10),
        late_threshold=(9, 5)
    )
    
    # Create and run UI
    ui = AttendanceUI(tk.Toplevel(), attendance_manager)
    ui.run()

if __name__ == "__main__":
    main() 