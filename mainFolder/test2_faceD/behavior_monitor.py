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
        self.camera_source = "rtsp://admin:Smart2025@192.168.137.200:554/Streaming/Channels/101"
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
        
        self.confidence_label = ttk.Label(status_frame, text="Confidence: -", font=('Arial', 12))
        self.confidence_label.pack(pady=5)
    
    def setup_log_panel(self):
        """Setup behavior log display panel."""
        log_frame = ttk.LabelFrame(self.right_panel, text="Behavior Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview
        self.log_tree = ttk.Treeview(log_frame, 
                                    columns=("Time", "Name", "Pose", "Confidence", "Location", "Description"),
                                    show="headings")
        
        # Define columns
        self.log_tree.heading("Time", text="Time")
        self.log_tree.heading("Name", text="Name")
        self.log_tree.heading("Pose", text="Pose")
        self.log_tree.heading("Confidence", text="Conf.")
        self.log_tree.heading("Location", text="Location")
        self.log_tree.heading("Description", text="Description")
        
        # Set column widths
        self.log_tree.column("Time", width=150)
        self.log_tree.column("Name", width=100)
        self.log_tree.column("Pose", width=100)
        self.log_tree.column("Confidence", width=70)
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
                confidence = f"{row['Confidence']:.2f}" if 'Confidence' in row else "-"
                self.log_tree.insert("", 0, values=(
                    row['Timestamp'],
                    row['Name'],
                    row['Pose'],
                    confidence,
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
                    
                    # Update behavior message and confidence
                    if " with confidence " in message:
                        msg_parts = message.split(" with confidence ")
                        self.behavior_label.config(text=msg_parts[0])
                        self.confidence_label.config(text=f"Confidence: {msg_parts[1]}")
                    else:
                        self.behavior_label.config(text=message)
                        self.confidence_label.config(text="Confidence: -")
                    
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