import streamlit as st
import cv2
import os
from dotenv import load_dotenv
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
from utils.detector import PeopleDetector
from utils.analyzer import CrowdAnalyzer
from playsound import playsound
import threading

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

# Configuration
CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', '0')
CAMERA_LOCATION = os.getenv('CAMERA_LOCATION', 'Main Entrance')
DETECTION_INTERVAL = float(os.getenv('DETECTION_INTERVAL', '2'))
PEOPLE_COUNT_THRESHOLD = int(os.getenv('PEOPLE_COUNT_THRESHOLD', '15'))
CSV_PATH = os.getenv('CSV_PATH', str(DATA_DIR / 'people_count_log.csv'))
DASHBOARD_REFRESH = int(os.getenv('DASHBOARD_REFRESH_INTERVAL', '10'))
MODEL_PATH = os.getenv('MODEL_PATH', str(MODELS_DIR / 'yolov8n.pt'))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

# Ensure CSV file exists
csv_path = Path(CSV_PATH)
if not csv_path.exists():
    df = pd.DataFrame(columns=['Timestamp', 'People_Count', 'Camera_Location'])
    df.to_csv(csv_path, index=False)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'last_alert' not in st.session_state:
    st.session_state.last_alert = 0

def play_alert():
    """Play alert sound."""
    try:
        current_time = time.time()
        if current_time - st.session_state.last_alert >= 10:  # Limit alert frequency
            alert_path = UTILS_DIR / "alert.mp3"
            if alert_path.exists():
                alert_thread = threading.Thread(target=playsound, args=(str(alert_path),))
                alert_thread.start()
                st.session_state.last_alert = current_time
            else:
                st.warning("Alert sound file not found")
    except Exception as e:
        st.error(f"Failed to play alert: {str(e)}")

def start_monitoring():
    """Start crowd monitoring."""
    try:
        # Check if model exists
        if not Path(MODEL_PATH).exists():
            st.error(f"Model not found at {MODEL_PATH}. Please download the model first.")
            return
        
        st.session_state.detector = PeopleDetector(
            model_path=MODEL_PATH,
            camera_source=CAMERA_SOURCE,
            camera_location=CAMERA_LOCATION,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            csv_path=CSV_PATH
        )
        st.session_state.detector.start_capture()
        st.session_state.monitoring = True
    except Exception as e:
        st.error(f"Failed to start monitoring: {str(e)}")
        st.session_state.monitoring = False

def stop_monitoring():
    """Stop crowd monitoring."""
    if st.session_state.detector:
        st.session_state.detector.stop_capture()
    st.session_state.monitoring = False

def main():
    st.set_page_config(
        page_title="Crowd Management System",
        page_icon="👥",
        layout="wide"
    )
    
    # Title and description
    st.title("👥 Crowd Management System")
    st.markdown("""
    Monitor and analyze crowd levels in real-time. The system uses computer vision 
    to detect people and provides insights through interactive visualizations.
    """)
    
    # Initialize analyzer
    analyzer = CrowdAnalyzer(CSV_PATH)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # System status indicator
        status_color = "🟢" if st.session_state.monitoring else "🔴"
        st.subheader(f"{status_color} System Status")
        st.text("Active" if st.session_state.monitoring else "Inactive")
        
        # Camera controls
        st.subheader("Camera Controls")
        if not st.session_state.monitoring:
            if st.button("▶️ Start Monitoring"):
                start_monitoring()
        else:
            if st.button("⏹️ Stop Monitoring"):
                stop_monitoring()
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        threshold = st.number_input(
            "Overcrowding Threshold",
            min_value=1,
            value=PEOPLE_COUNT_THRESHOLD,
            help="Alert when people count exceeds this number"
        )
        
        # Download data
        st.subheader("📊 Data Export")
        if Path(CSV_PATH).exists():
            with open(CSV_PATH, 'rb') as f:
                st.download_button(
                    label="📥 Download CSV",
                    data=f,
                    file_name="crowd_data.csv",
                    mime="text/csv"
                )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video feed
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
        
        # Time series chart
        st.subheader("📈 People Count Over Time")
        time_series_placeholder = st.empty()
    
    with col2:
        # Current stats
        st.subheader("📊 Current Statistics")
        stats_placeholder = st.empty()
        
        # Hourly distribution
        st.subheader("⏰ Hourly Distribution")
        hourly_placeholder = st.empty()
    
    # Heatmap
    st.subheader("🗓️ Daily Heatmap")
    heatmap_placeholder = st.empty()
    
    # Main loop
    while True:
        try:
            # Load and analyze data
            df = analyzer.load_data(hours_back=24)
            
            if not df.empty:
                # Check for overcrowding
                if analyzer.check_overcrowding(df, threshold):
                    st.error("⚠️ WARNING: Overcrowding Detected!")
                    play_alert()
                
                # Update statistics
                stats = analyzer.get_current_stats(df)
                with stats_placeholder:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("👥 Current Count", stats['current_count'])
                    col2.metric("📈 Daily Maximum", stats['daily_max'])
                    col3.metric("📊 Daily Average", stats['daily_avg'])
                    st.text(f"Last Update: {stats['last_update']}")
                
                # Update visualizations
                time_series_placeholder.plotly_chart(
                    analyzer.create_time_series(df),
                    use_container_width=True
                )
                hourly_placeholder.plotly_chart(
                    analyzer.create_hourly_bar(df),
                    use_container_width=True
                )
                heatmap_placeholder.plotly_chart(
                    analyzer.create_heatmap(df),
                    use_container_width=True
                )
            
            # Update video feed
            if st.session_state.monitoring and st.session_state.detector:
                frame, count = st.session_state.detector.detect_people(DETECTION_INTERVAL)
                video_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_column_width=True
                )
            elif not st.session_state.monitoring:
                video_placeholder.info("📸 Camera feed is inactive. Click 'Start Monitoring' to begin.")
            
            # Wait before next update
            time.sleep(1.0 / 30.0)  # 30 FPS
            
        except Exception as e:
            st.error(f"Error updating dashboard: {str(e)}")
            time.sleep(5)  # Wait before retrying
        
        # Rerun the app for updates
        if not st.session_state.monitoring:
            break

if __name__ == "__main__":
    main()