# Crowd Management System

A real-time crowd monitoring and analysis system using computer vision and Streamlit.

## Features

- Real-time people detection using YOLOv8
- Automatic data logging to CSV
- Interactive dashboard with:
  - Live video feed
  - Real-time statistics
  - Time series visualization
  - Hourly distribution charts
  - Daily heatmaps
- Overcrowding alerts with sound notifications
- Data export functionality
- Configurable parameters via environment variables

## Setup

1. Clone the repository and navigate to the project directory:
```bash
cd crowd_management
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLOv8 model:
```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

5. Create a `.env` file from the example:
```bash
cp .env.example .env
```

6. Edit the `.env` file to configure your settings:
- `CAMERA_SOURCE`: Camera source (0 for default webcam, or RTSP URL)
- `CAMERA_LOCATION`: Name/location of the camera
- `DETECTION_INTERVAL`: Time between detections in seconds
- `PEOPLE_COUNT_THRESHOLD`: Maximum people before overcrowding alert
- Other parameters as needed

## Usage

1. Start the Streamlit dashboard:
```bash
streamlit run app.py
```

2. Use the dashboard controls:
- Click "Start Monitoring" to begin detection
- View real-time statistics and visualizations
- Download data using the export button
- Configure settings as needed

3. Monitor alerts:
- Red warning banner appears when overcrowding detected
- Audio alert plays (if enabled)
- Visual indicators on charts show threshold levels

## Project Structure

```
crowd_management/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example       # Example environment variables
├── README.md          # This file
├── data/              # Data storage
│   └── people_count_log.csv
├── models/            # Model storage
│   └── yolov8n.pt
└── utils/             # Utility modules
    ├── detector.py    # People detection module
    ├── analyzer.py    # Data analysis module
    └── alert.mp3      # Alert sound file
```

## Requirements

- Python 3.8+
- OpenCV
- YOLOv8
- Streamlit
- Plotly
- Pandas
- Other dependencies in requirements.txt

## Notes

- The system uses YOLOv8 for person detection
- Detection runs every 2 seconds by default to reduce computation
- Data is logged to CSV for historical analysis
- The dashboard auto-refreshes every 10 seconds
- Overcrowding alerts have a 10-second cooldown

## Troubleshooting

1. Camera issues:
   - Check camera source in .env file
   - Ensure camera permissions are granted
   - Try different camera index if using webcam

2. Performance issues:
   - Adjust DETECTION_INTERVAL for slower/faster updates
   - Use a smaller YOLOv8 model (nano vs small)
   - Reduce dashboard refresh rate

3. Alert sound not working:
   - Check audio device settings
   - Ensure alert.mp3 file exists
   - Install required audio codecs 