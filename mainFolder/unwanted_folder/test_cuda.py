import torch
import cv2
import sys

def check_cuda():
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    
    print("\n=== PyTorch CUDA Check ===")
    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"Error checking PyTorch CUDA: {str(e)}")
    
    print("\n=== OpenCV CUDA Check ===")
    try:
        print(f"OpenCV build information: {cv2.getBuildInformation()}")
        cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"OpenCV CUDA available: {cuda_enabled}")
        if cuda_enabled:
            print(f"OpenCV CUDA device count: {cv2.cuda.getCudaEnabledDeviceCount()}")
            print(f"OpenCV CUDA device name: {cv2.cuda.Device(0).name()}")
    except Exception as e:
        print(f"Error checking OpenCV CUDA: {str(e)}")
        print("Note: If you need CUDA support in OpenCV, you may need to build OpenCV from source with CUDA support.")

if __name__ == "__main__":
    check_cuda() 