import streamlit as st
from ultralytics import YOLO
import numpy as np
import tempfile
import cv2
import random

st.set_page_config(layout="wide")
st.title("🛡️ SmartSafe Video Dashboard")

# Sidebar
mode = st.sidebar.selectbox("Demo Mode", ["Auto", "Force Safe", "Force Risk"])

# Load model
model = YOLO("yolov8n.pt")

# Upload video
uploaded_file = st.file_uploader("📹 Upload Video", type=["mp4", "mov", "avi"])

# KPI
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("👷 Helmet Status", "Unknown")
kpi2.metric("⚠️ Risk Score", 0)
kpi3.metric("🚨 Alerts Today", 0)

frame_placeholder = st.empty()

if uploaded_file:
    # save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    alerts = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated = results.plot()

        # detect person
        person_detected = False
        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            if label == "person":
                person_detected = True

        # fake helmet logic
        if person_detected:
            if mode == "Force Safe":
                helmet_detected = True
            elif mode == "Force Risk":
                helmet_detected = False
            else:
                helmet_detected = random.choice([True, False])
        else:
            helmet_detected = False

        # risk
        if helmet_detected:
            helmet_status = "Safe"
            risk_score = random.randint(0, 30)
        else:
            helmet_status = "No Helmet"
            risk_score = random.randint(60, 90)
            alerts += 1

        # update KPI
        kpi1.metric("👷 Helmet Status", helmet_status)
        kpi2.metric("⚠️ Risk Score", risk_score)
        kpi3.metric("🚨 Alerts Today", alerts)

        # show frame
        frame_placeholder.image(annotated, channels="BGR")

        # delay (ปรับความเร็ว)
        cv2.waitKey(30)

    cap.release()
