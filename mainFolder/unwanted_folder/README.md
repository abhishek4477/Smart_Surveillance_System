# Smart Surveillance System

An AI-powered surveillance system with face recognition, crowd management, and behavioral analytics capabilities.

## Features

1. **Attendance Monitoring**
   - Face detection and recognition
   - Automated attendance logging
   - Database integration

2. **Crowd Management**
   - Real-time people counting
   - Crowd density tracking
   - Statistical visualization

3. **Behavioral Analysis**
   - Pose detection
   - Activity recognition
   - Natural language querying

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Supabase credentials

## Project Structure

```
mainFolder/
├── requirements.txt
├── .env.example
├── src/
│   ├── face_recognition/
│   ├── crowd_management/
│   ├── behavior_analysis/
│   └── utils/
├── config/
├── data/
│   └── face_embeddings/
└── ui/
```

## Usage

1. Start the application:
   ```bash
   streamlit run src/app.py
   ```
2. Access the web interface at `http://localhost:8501`

## Configuration

- Adjust detection parameters in `config/settings.yaml`
- Manage database settings in `.env`
- Configure camera sources in `config/cameras.yaml` 