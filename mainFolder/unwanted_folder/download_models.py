import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, filename: str):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        logger.error(f"Error downloading {filename}: {str(e)}")
        return False

def setup_models():
    """Download and set up required models."""
    # Create models directory
    models_dir = os.path.join('models', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Model URLs (updated with correct URLs)
    model_urls = {
        'retinaface_r50_v1': 'https://storage.googleapis.com/insightface-models/retinaface_r50_v1.zip',
        'buffalo_l': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
    }
    
    # Download and extract models
    for model_name, url in model_urls.items():
        model_dir = os.path.join(models_dir, model_name)
        if not os.path.exists(model_dir):
            logger.info(f"Downloading {model_name}...")
            zip_path = os.path.join(models_dir, f"{model_name}.zip")
            
            try:
                # Download the model
                if not download_file(url, zip_path):
                    logger.error(f"Failed to download {model_name}")
                    continue
                
                logger.info(f"Extracting {model_name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(models_dir)
                
                # Clean up zip file
                os.remove(zip_path)
                logger.info(f"{model_name} setup complete!")
                
            except Exception as e:
                logger.error(f"Error processing {model_name}: {str(e)}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        else:
            logger.info(f"{model_name} already exists!")

def verify_models():
    """Verify that all required models are present."""
    models_dir = os.path.join('models', 'models')
    required_models = ['retinaface_r50_v1', 'buffalo_l']
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
    
    if missing_models:
        logger.error(f"Missing models: {', '.join(missing_models)}")
        return False
    
    logger.info("All required models are present")
    return True

if __name__ == "__main__":
    setup_models()
    verify_models() 