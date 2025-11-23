import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import glob
import datetime
import shutil
import os

def load_latest_training_data(data_dir: str = "data/pose_training"):
    """Load the most recent training data file."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all pose data files
    data_files = list(data_dir.glob("pose_data_*.json"))
    if not data_files:
        raise FileNotFoundError("No training data files found")
    
    # Get the most recent file
    latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading training data from: {latest_file}")
    
    # Load the data
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Print metadata if available
    if 'metadata' in data:
        print("\nDataset Information:")
        print(f"Timestamp: {data['metadata']['timestamp']}")
        print(f"Total Samples: {data['metadata']['total_samples']}")
        print("\nSamples per pose:")
        for pose, count in data['metadata']['samples_per_pose'].items():
            print(f"  {pose}: {count}")
        print()
    
    X = np.array(data['features'])
    y = np.array(data['poses'])
    
    return X, y

def train_model(X, y):
    """Train pose classification model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    return model

def save_model(model, model_dir: Path):
    """Save model with timestamp and update latest version."""
    model_dir.mkdir(exist_ok=True)
    
    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"pose_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    
    # Save/update latest version by copying
    latest_path = model_dir / "pose_model_latest.pkl"
    try:
        shutil.copy2(model_path, latest_path)
    except Exception as e:
        print(f"Note: Could not update latest model link: {e}")
        print("You can still use the timestamped model file directly.")
    
    return model_path

def main():
    try:
        # Load data
        print("Loading training data...")
        X, y = load_latest_training_data()
        
        # Train model
        print("\nTraining model...")
        model = train_model(X, y)
        
        # Save model
        print("\nSaving model...")
        model_dir = Path("models")
        model_path = save_model(model, model_dir)
        
        print(f"Model saved to: {model_path}")
        print("Training completed successfully!")
        
        # Save model info
        info_path = model_dir / "model_info.txt"
        with open(info_path, "a") as f:
            f.write(f"\nModel trained on: {datetime.datetime.now()}")
            f.write(f"\nModel file: {model_path.name}")
            f.write("\n" + "="*50 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please run pose_data_collector.py first to collect training data.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")

if __name__ == "__main__":
    main() 