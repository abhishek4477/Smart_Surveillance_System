@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing core dependencies...
pip install numpy>=1.21.0
pip install opencv-python>=4.7.0
pip install python-dotenv>=0.19.0

echo Installing deep learning frameworks...
pip install torch>=2.0.0 torchvision>=0.15.0

echo Installing face recognition packages...
pip install face-recognition>=1.3.0
pip install deepface>=0.0.79

echo Installing database and UI packages...
pip install supabase>=1.0.3
pip install streamlit>=1.22.0
pip install pandas>=1.5.0
pip install matplotlib>=3.5.0
pip install plotly>=5.13.0

echo Installing computer vision packages...
pip install mediapipe>=0.9.0
pip install pillow>=9.0.0
pip install ultralytics>=8.0.0

echo Installing object tracking packages...
pip install supervision>=0.1.0

echo Installation complete!
pause 