@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Checking current installations...
pip list | findstr "torch ultralytics"

echo Uninstalling existing packages...
pip uninstall -y ultralytics
pip uninstall -y torch
pip uninstall -y torchvision

echo Installing GPU-enabled packages...
echo Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo Installing YOLOv8 with CUDA support...
pip install ultralytics[cuda]

echo Verifying installations...
python test_cuda.py

echo.
echo If you see any errors above, please check:
echo 1. CUDA Toolkit is installed (version 11.8 or compatible)
echo 2. NVIDIA drivers are up to date
echo 3. Your GPU is CUDA-capable
echo.
echo You can check your CUDA version with: nvcc --version
echo.

pause 