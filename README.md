# AI-Powered Classroom Engagement Monitor

## Overview

The **AI-Powered Classroom Engagement Monitor** is a tool designed to evaluate and enhance student engagement in online and hybrid learning environments. By leveraging facial recognition and emotion analysis technologies, this application provides real-time insights into student attention and engagement, helping educators make data-driven decisions to improve teaching effectiveness.

## Features

- **Real-time Emotion Detection**: Analyze and display students' emotional states using facial recognition.
- **Attention Measurement**: Evaluate student attention levels based on eye movements and blink rates.
- **Engagement Metrics**: Aggregate emotion and attention data to calculate overall engagement scores.
- **Session Summary**: Provide a detailed summary of engagement and attention metrics at the end of each session.

## Project Structure


## Installation

### Prerequisites

- Python 3.7 or higher
- A webcam connected to your computer

### Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/engagement-monitor.git
   cd engagement-monitor
   
2. **Install Dependencies:**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
  Install required packages:
   ```bash
   pip install -r requirements.txt
  ```
3. ***Download dlib's Shape Predictor Model:***:
   This [repository] [https://github.com/z-mahmud22/Dlib_Windows_Python3.x/blob/main/README.md] contains the compiled binary (.whl) files for the Dlib library to install on Python versions 3.7, 3.8, 3.9, 3.10, 
   3.11, and 3.12 on a Windows x64 OS.
  For Python 3.11
  ```bash
  python -m pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
  ```

***USAGE***
1. Start the Streamlit Application:
   ```bash
   streamlit run app.py
   ```
2. Interact with the Application:

  Open the Streamlit app in your browser (usually at http://localhost:8501).
  Click "Start Monitoring" to begin real-time monitoring.
  Click "Stop Monitoring" to end the session and view the summary.



