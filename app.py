import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import cv2
from webcam_processor import process_webcam_feed
from PIL import Image

st.title("AI-Powered Classroom Engagement Monitor")

st.write("""
This application monitors student engagement in real-time using facial recognition and emotion analysis.
""")

if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'summary' not in st.session_state:
    st.session_state.summary = None

start_monitoring = st.button('Start Monitoring', key='start_monitoring')

if start_monitoring:
    st.session_state.monitoring = True

if st.session_state.monitoring:
    stframe = st.empty()  # Placeholder for the webcam feed
    
    stop_monitoring = st.button('Stop Monitoring', key='stop_monitoring')
    
    if stop_monitoring:
        st.session_state.monitoring = False
        st.session_state.summary = process_webcam_feed(stframe, stop=True)
    else:
        process_webcam_feed(stframe)
    
    if st.session_state.summary:
        st.write("### Session Summary")
        st.json(st.session_state.summary)

st.write("Note: Ensure your webcam is connected and accessible.")
