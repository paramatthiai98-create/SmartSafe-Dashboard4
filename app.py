import streamlit as st
from ultralytics import YOLO
import numpy as np
import tempfile
import cv2
import random
import time

# ------------------ CONFIG ------------------
st.set_page_config(layout="wide")
st.title("🛡️ SmartSafe Video Dashboard")

# ------------------ SIDEBAR ------------------
mode = st.sidebar.selectbox(
    "🎛️ Demo Mode",
    ["Auto (สลับคน)", "Force Safe", "Force Risk"]
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------ KPI ------------------
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("👷 Helmet Status", "Waiting...")
kpi2.metric("⚠️ Risk Score", 0)
kpi3.metric("🚨 Alerts Today", 0)

# ------------------ UPLOAD VIDEO ------------------
uploaded_file = st.file_uploader("📹 Upload Video", type=["mp4", "mov", "avi"])

frame_placeholder = st.empty()

# ------------------ PROCESS VIDEO ------------------
if uploaded_file:
    st.success("✅ Video uploaded!")

    # save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    alerts = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- YOLO DETECTION ----------
        results = model(frame)[0]
        annotated = results.plot()

        # ---------- เก็บคนทั้งหมด ----------
        person_boxes = []
        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            if label == "person":
                person_boxes.append(box)

        # ---------- วนทีละคน ----------
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ---------- Fake Helmet Logic ----------
            if mode == "Force Safe":
                helmet = True
            elif mode == "Force Risk":
                helmet = False
            else:
                # 🎯 คนที่ 1 Safe, คนที่ 2 Risk, สลับไปเรื่อยๆ
                helmet = (i % 2 == 0)

            # ---------- วาดกรอบ ----------
            if helmet:
                color = (0, 255, 0)
                text = "Safe"
            else:
                color = (0, 0, 255)
                text = "No Helmet"
                alerts += 1
                st.toast("🚨 No Helmet Detected!")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # ---------- KPI ----------
        total_people = len(person_boxes)

        if total_people > 0:
            risk_score = int((alerts / (total_people + 1)) * 100)
        else:
            risk_score = 0

        if alerts > 0:
            helmet_status = "🔴 Risk Detected"
        else:
            helmet_status = "🟢 Safe"

        kpi1.metric("👷 Helmet Status", helmet_status)
        kpi2.metric("⚠️ Risk Score", risk_score)
        kpi3.metric("🚨 Alerts Today", alerts)

        # ---------- SHOW FRAME ----------
        frame_placeholder.image(annotated, channels="BGR")

        # ---------- DELAY ----------
        time.sleep(0.03)

    cap.release()
    st.success("🎬 Video finished!")
