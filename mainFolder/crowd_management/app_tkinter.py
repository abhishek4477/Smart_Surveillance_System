import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
from dotenv import load_dotenv
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
import numpy as np
from utils.detector import PeopleDetector
from utils.analyzer import CrowdAnalyzer
from playsound import playsound
import threading
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Initialize paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
UTILS_DIR = ROOT_DIR / "utils"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Load environment variables
load_dotenv()

class CrowdManagementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Management System")
        self.root.state('zoomed')  # Maximize window
        
        # Load configuration
        self.load_config()
        
        # Initialize components
        self.setup_ui()
        self.setup_variables()
        self.setup_analyzer()
        
        # Start update loop
        self.update_ui()
    
    def load_config(self):
        """Load configuration from environment variables."""
        self.camera_source = os.getenv('CAMERA_SOURCE', '0')
        self.camera_location = os.getenv('CAMERA_LOCATION', 'Main Entrance')
        self.detection_interval = float(os.getenv('DETECTION_INTERVAL', '2'))
        self.people_threshold = int(os.getenv('PEOPLE_COUNT_THRESHOLD', '15'))
        self.csv_path = os.getenv('CSV_PATH', str(DATA_DIR / 'people_count_log.csv'))
        self.model_path = os.getenv('MODEL_PATH', str(MODELS_DIR / 'yolov8n.pt'))
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
        
        # Ensure CSV file exists
        csv_path = Path(self.csv_path)
        if not csv_path.exists():
            df = pd.DataFrame(columns=['Timestamp', 'People_Count', 'Camera_Location'])
            df.to_csv(csv_path, index=False)
    
    def setup_variables(self):
        """Initialize variables and states."""
        self.detector = None
        self.is_monitoring = False
        self.last_alert = 0
        self.current_frame = None
    
    def setup_analyzer(self):
        """Initialize the crowd analyzer."""
        self.analyzer = CrowdAnalyzer(self.csv_path)
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel (video and controls)
        self.left_panel = ttk.Frame(self.main_container)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right panel (stats and graphs)
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup components
        self.setup_control_panel()
        self.setup_video_panel()
        self.setup_stats_panel()
        self.setup_graphs_panel()
    
    def setup_control_panel(self):
        """Setup control panel with buttons and settings."""
        control_frame = ttk.LabelFrame(self.left_panel, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Status indicator
        self.status_label = ttk.Label(control_frame, text="Status: Inactive", font=('Arial', 12))
        self.status_label.pack(pady=5)
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings
        settings_frame = ttk.LabelFrame(control_frame, text="Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Overcrowding Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.StringVar(value=str(self.people_threshold))
        threshold_entry = ttk.Entry(settings_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.pack(side=tk.LEFT, padx=5)
    
    def setup_video_panel(self):
        """Setup video display panel."""
        video_frame = ttk.LabelFrame(self.left_panel, text="Live Feed")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
    
    def setup_stats_panel(self):
        """Setup statistics panel."""
        stats_frame = ttk.LabelFrame(self.right_panel, text="Current Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create stats labels
        self.current_count_label = ttk.Label(stats_frame, text="Current Count: 0", font=('Arial', 12))
        self.current_count_label.pack(pady=2)
        
        self.daily_max_label = ttk.Label(stats_frame, text="Daily Maximum: 0", font=('Arial', 12))
        self.daily_max_label.pack(pady=2)
        
        self.daily_avg_label = ttk.Label(stats_frame, text="Daily Average: 0", font=('Arial', 12))
        self.daily_avg_label.pack(pady=2)
        
        self.last_update_label = ttk.Label(stats_frame, text="Last Update: -", font=('Arial', 10))
        self.last_update_label.pack(pady=2)
    
    def setup_graphs_panel(self):
        """Setup graphs panel."""
        graphs_frame = ttk.LabelFrame(self.right_panel, text="Analytics")
        graphs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Time series plot
        self.fig_time_series = Figure(figsize=(6, 4))
        self.ax_time_series = self.fig_time_series.add_subplot(111)
        self.canvas_time_series = FigureCanvasTkAgg(self.fig_time_series, master=graphs_frame)
        self.canvas_time_series.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Hourly distribution plot
        self.fig_hourly = Figure(figsize=(6, 4))
        self.ax_hourly = self.fig_hourly.add_subplot(111)
        self.canvas_hourly = FigureCanvasTkAgg(self.fig_hourly, master=graphs_frame)
        self.canvas_hourly.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)
    
    def update_ui(self):
        """Update UI elements."""
        try:
            if self.is_monitoring:
                # Update video feed
                if self.detector:
                    frame, count = self.detector.detect_people(self.detection_interval)
                    if frame is not None:
                        self.show_frame(frame)
                
                # Update analytics
                self.update_analytics()
            
            # Schedule next update
            self.root.after(33, self.update_ui)  # ~30 FPS
            
        except Exception as e:
            messagebox.showerror("Error", f"Error updating UI: {str(e)}")
    
    def update_analytics(self):
        """Update analytics displays."""
        try:
            df = self.analyzer.load_data(hours_back=24)
            if not df.empty:
                # Check overcrowding
                if self.analyzer.check_overcrowding(df, int(self.threshold_var.get())):
                    self.show_alert()
                
                # Update stats
                stats = self.analyzer.get_current_stats(df)
                self.current_count_label.config(text=f"Current Count: {stats['current_count']}")
                self.daily_max_label.config(text=f"Daily Maximum: {stats['daily_max']}")
                self.daily_avg_label.config(text=f"Daily Average: {stats['daily_avg']:.1f}")
                self.last_update_label.config(text=f"Last Update: {stats['last_update']}")
                
                # Update plots
                self.update_plots(df)
        except Exception as e:
            print(f"Error updating analytics: {str(e)}")
    
    def update_plots(self, df):
        """Update visualization plots."""
        try:
            # Time series plot
            self.ax_time_series.clear()
            self.ax_time_series.plot(df['Timestamp'], df['People_Count'])
            self.ax_time_series.set_title('People Count Over Time')
            self.ax_time_series.tick_params(axis='x', rotation=45)
            self.fig_time_series.tight_layout()
            self.canvas_time_series.draw()
            
            # Hourly distribution plot
            self.ax_hourly.clear()
            df['Hour'] = df['Timestamp'].dt.hour
            hourly_avg = df.groupby('Hour')['People_Count'].mean()
            self.ax_hourly.bar(hourly_avg.index, hourly_avg.values)
            self.ax_hourly.set_title('Average People Count by Hour')
            self.ax_hourly.set_xlabel('Hour of Day')
            self.ax_hourly.set_ylabel('Average Count')
            self.fig_hourly.tight_layout()
            self.canvas_hourly.draw()
            
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
    
    def show_frame(self, frame):
        """Display a frame in the video label."""
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.config(image=photo)
            self.video_label.image = photo
        except Exception as e:
            print(f"Error showing frame: {str(e)}")
    
    def show_alert(self):
        """Show overcrowding alert."""
        current_time = time.time()
        if current_time - self.last_alert >= 10:  # Limit alert frequency
            self.last_alert = current_time
            alert_path = UTILS_DIR / "alert.mp3"
            if alert_path.exists():
                threading.Thread(target=playsound, args=(str(alert_path),)).start()
            messagebox.showwarning("Alert", "Overcrowding Detected!")
    
    def start_monitoring(self):
        """Start crowd monitoring."""
        try:
            # Check if model exists, if not download it
            if not Path(self.model_path).exists():
                try:
                    from ultralytics import YOLO
                    
                    # Show download status
                    self.status_label.config(text="Status: Downloading YOLOv8 model...")
                    self.root.update()
                    
                    # Download and save the model
                    model = YOLO('yolov8n.pt')
                    model.save(self.model_path)
                    
                    self.status_label.config(text="Status: Model downloaded successfully")
                    self.root.update()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to download model: {str(e)}\n\nPlease ensure you have internet connection and ultralytics installed.")
                    return
            
            self.detector = PeopleDetector(
                model_path=self.model_path,
                camera_source=self.camera_source,
                camera_location=self.camera_location,
                confidence_threshold=self.confidence_threshold,
                csv_path=self.csv_path
            )
            self.detector.start_capture()
            self.is_monitoring = True
            
            # Update UI
            self.status_label.config(text="Status: Active")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
            self.status_label.config(text="Status: Error starting monitoring")
            self.is_monitoring = False
    
    def stop_monitoring(self):
        """Stop crowd monitoring."""
        if self.detector:
            self.detector.stop_capture()
        self.is_monitoring = False
        
        # Update UI
        self.status_label.config(text="Status: Inactive")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.video_label.config(image='')
    
    def on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.stop_monitoring()
            self.root.quit()

def main():
    root = tk.Tk()
    app = CrowdManagementApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 