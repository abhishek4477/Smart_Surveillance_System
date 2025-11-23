import numpy as np
from scipy.io import wavfile
from pathlib import Path

# Create utils directory if it doesn't exist
utils_dir = Path('utils')
utils_dir.mkdir(exist_ok=True)

# Generate a simple beep sound
sample_rate = 44100
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration))
frequency = 440  # A4 note
signal = np.sin(2 * np.pi * frequency * t) * 0.5

# Save as WAV file
wavfile.write(str(utils_dir / 'alert.wav'), sample_rate, signal.astype(np.float32)) 