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
import customtkinter as ctk

# Import our systems
from attendance_system import AttendanceUI, AttendanceManager
from test2_faaceD.crowd_management import CrowdManagementSystem
from behavior_monitor import BehaviorMonitorApp
from modern_ui import ModernUI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OmniSightApp:
    def __init__(self, root):
        self.root = root
        
        # Initialize paths
        self.setup_paths()
        
        # Initialize UI
        self.ui = ModernUI()
        self.main_container = self.ui.create_main_window(self.root)
        
        # Initialize subsystems
        self.attendance_system = None
        self.crowd_system = None
        self.behavior_system = None
        
        # Bind launch buttons
        self.bind_launch_buttons()
    
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
    
    def bind_launch_buttons(self):
        """Bind launch buttons to their respective functions."""
        # Find and bind buttons in system cards
        for widget in self.main_container.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                for card in widget.winfo_children():
                    if isinstance(card, ctk.CTkFrame):
                        for child in card.winfo_children():
                            if isinstance(child, ctk.CTkButton):
                                # Get the card's title
                                for label in card.winfo_children():
                                    if isinstance(label, ctk.CTkLabel):
                                        font = label.cget("font")
                                        if hasattr(font, "_size") and font._size == 20:
                                            title = label.cget("text")
                                            if "Attendance" in title:
                                                child.configure(command=self.launch_attendance_system)
                                            elif "Crowd" in title:
                                                child.configure(command=self.launch_crowd_management)
                                            elif "Behavior" in title:
                                                child.configure(command=self.launch_behavior_analysis)
    
    def launch_attendance_system(self):
        """Launch the attendance system."""
        try:
            # Create window
            window = tk.Toplevel(self.root)
            window.title("Attendance System")
            window.geometry("1200x800")
            
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
            self.ui.update_status("Attendance System launched")
            logger.info("Attendance System launched")
            
        except Exception as e:
            logger.error(f"Error launching Attendance System: {str(e)}")
            messagebox.showerror("Error", f"Failed to launch Attendance System: {str(e)}")
    
    def launch_crowd_management(self):
        """Launch the crowd management system."""
        try:
            # Create window
            crowd_window = ctk.CTkToplevel(self.root)
            crowd_window.title("Crowd Management System")
            crowd_window.geometry("1400x800")
            
            # Create main container
            main_container = ctk.CTkFrame(crowd_window)
            main_container.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create left panel for video
            video_frame = ctk.CTkFrame(main_container)
            video_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
            
            # Create right panel for stats
            stats_frame = ctk.CTkFrame(main_container)
            stats_frame.pack(side="right", fill="both", padx=(5, 0), width=400)
            
            # Add stats labels
            stats_title = ctk.CTkLabel(stats_frame, text="Crowd Statistics", font=("Helvetica", 20, "bold"))
            stats_title.pack(pady=(10, 20))
            
            # Current count
            count_frame = ctk.CTkFrame(stats_frame)
            count_frame.pack(fill="x", padx=10, pady=5)
            count_label = ctk.CTkLabel(count_frame, text="Current Count:", font=("Helvetica", 14))
            count_label.pack(side="left", padx=5)
            self.current_count = ctk.CTkLabel(count_frame, text="0", font=("Helvetica", 14, "bold"))
            self.current_count.pack(side="right", padx=5)
            
            # Alert Level
            alert_frame = ctk.CTkFrame(stats_frame)
            alert_frame.pack(fill="x", padx=10, pady=5)
            alert_label = ctk.CTkLabel(alert_frame, text="Alert Level:", font=("Helvetica", 14))
            alert_label.pack(side="left", padx=5)
            self.alert_level = ctk.CTkLabel(alert_frame, text="LOW", font=("Helvetica", 14, "bold"))
            self.alert_level.pack(side="right", padx=5)
            
            # Peak Count Today
            peak_frame = ctk.CTkFrame(stats_frame)
            peak_frame.pack(fill="x", padx=10, pady=5)
            peak_label = ctk.CTkLabel(peak_frame, text="Peak Count:", font=("Helvetica", 14))
            peak_label.pack(side="left", padx=5)
            self.peak_count = ctk.CTkLabel(peak_frame, text="0", font=("Helvetica", 14, "bold"))
            self.peak_count.pack(side="right", padx=5)
            
            # Average Count
            avg_frame = ctk.CTkFrame(stats_frame)
            avg_frame.pack(fill="x", padx=10, pady=5)
            avg_label = ctk.CTkLabel(avg_frame, text="Average Count:", font=("Helvetica", 14))
            avg_label.pack(side="left", padx=5)
            self.avg_count = ctk.CTkLabel(avg_frame, text="0", font=("Helvetica", 14, "bold"))
            self.avg_count.pack(side="right", padx=5)
            
            # Graph
            graph_frame = ctk.CTkFrame(stats_frame)
            graph_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create matplotlib figure
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # IP Camera settings
            settings_frame = ctk.CTkFrame(stats_frame)
            settings_frame.pack(fill="x", padx=10, pady=10)
            
            ip_label = ctk.CTkLabel(settings_frame, text="IP Camera URL:", font=("Helvetica", 12))
            ip_label.pack(side="left", padx=5)
            
            self.ip_entry = ctk.CTkEntry(settings_frame, placeholder_text="http://camera_ip:port/video")
            self.ip_entry.pack(side="left", padx=5, expand=True, fill="x")
            self.ip_entry.insert(0, "http://192.168.1.100:8080/video")  # Default IP
            
            connect_btn = ctk.CTkButton(settings_frame, text="Connect", command=self.connect_camera)
            connect_btn.pack(side="right", padx=5)
            
            # Create video canvas
            canvas_width = 800
            canvas_height = 600
            self.video_canvas = tk.Canvas(video_frame, width=canvas_width, height=canvas_height, bg="black")
            self.video_canvas.pack(pady=10)
            
            # Initialize the new crowd management system
            self.crowd_system = CrowdManagementSystem()
            self.cap = None
            self.peak_count_value = 0
            self.count_history = []
            
            # Start with default camera
            self.connect_camera()
            
            # Update status
            self.ui.update_status("New Crowd Management System launched")
            logger.info("New Crowd Management System launched")
            
        except Exception as e:
            logger.error(f"Error launching Crowd Management System: {str(e)}")
            messagebox.showerror("Error", f"Failed to launch Crowd Management System: {str(e)}")
    
    def connect_camera(self):
        """Connect to IP camera or default camera."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            camera_url = self.ip_entry.get()
            if camera_url and camera_url.startswith("http"):
                self.cap = cv2.VideoCapture(camera_url)
            else:
                self.cap = cv2.VideoCapture(0)  # Default to webcam if URL is invalid
            
            if not self.cap.isOpened():
                raise Exception("Could not connect to camera")
            
            self.update_frame()
        except Exception as e:
            logger.error(f"Camera connection error: {str(e)}")
            messagebox.showerror("Error", f"Failed to connect to camera: {str(e)}")
    
    def update_frame(self):
        """Update video frame and statistics."""
        try:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Process frame with crowd management system
                    processed_frame = self.crowd_system.process_frame(frame)
                    
                    # Get current count and update statistics
                    current_count = len(self.crowd_system.boxes) if hasattr(self.crowd_system, 'boxes') else 0
                    self.current_count.configure(text=str(current_count))
                    
                    # Update peak count
                    self.peak_count_value = max(self.peak_count_value, current_count)
                    self.peak_count.configure(text=str(self.peak_count_value))
                    
                    # Update count history and average
                    self.count_history.append(current_count)
                    if len(self.count_history) > 100:  # Keep last 100 readings
                        self.count_history.pop(0)
                    avg = sum(self.count_history) / len(self.count_history)
                    self.avg_count.configure(text=f"{avg:.1f}")
                    
                    # Update alert level
                    alert = "LOW" if current_count < 5 else "MODERATE" if current_count < 15 else "HIGH"
                    self.alert_level.configure(text=alert)
                    
                    # Update graph
                    if len(self.count_history) % 10 == 0:  # Update graph every 10 frames
                        self.ax.clear()
                        self.ax.plot(self.count_history[-50:])  # Show last 50 readings
                        self.ax.set_title("Crowd Count Trend")
                        self.ax.set_ylim(bottom=0)
                        self.canvas.draw()
                    
                    # Convert and display frame
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_resized = frame_pil.resize((800, 600))
                    photo = ImageTk.PhotoImage(frame_resized)
                    self.video_canvas.create_image(0, 0, image=photo, anchor="nw")
                    self.video_canvas.photo = photo
            
            # Schedule next update
            self.root.after(10, self.update_frame)
            
        except Exception as e:
            logger.error(f"Frame update error: {str(e)}")
    
    def launch_behavior_analysis(self):
        """Launch the behavior analysis system."""
        try:
            window = ctk.CTkToplevel(self.root)
            
            # Initialize behavior analysis system
            self.behavior_system = BehaviorMonitorApp(window)
            
            # Update status
            self.ui.update_status("Behavior Analysis System launched")
            logger.info("Behavior Analysis System launched")
            
        except Exception as e:
            logger.error(f"Error launching Behavior Analysis System: {str(e)}")
            messagebox.showerror("Error", f"Failed to launch Behavior Analysis System: {str(e)}")
    
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
    # Set appearance mode
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create main window
    root = ctk.CTk()
    app = OmniSightApp(root)
    
    # Set closing protocol
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start application
    logger.info("OmniSight application started")
    root.mainloop()

if __name__ == "__main__":
    main() 