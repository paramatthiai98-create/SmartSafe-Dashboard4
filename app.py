import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import random

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="SmartSafe Dashboard", layout="wide")
st.title("🛡️ SmartSafe Co-Pilot Dashboard")

# ------------------------
# Sidebar (Interaction)
# ------------------------
st.sidebar.header("⚙️ Control Panel")

time_range = st.sidebar.selectbox("Select Time Range", ["Last 1 min", "Last 5 min", "Last 1 hour"])
worker = st.sidebar.selectbox("Select Worker", ["Worker A", "Worker B", "Worker C"])

# 🔥 Demo Mode (สำคัญ)
mode = st.sidebar.selectbox("Demo Mode", ["Auto", "Force Safe", "Force Risk"])

# ------------------------
# Load Model
# ------------------------
model = YOLO("yolov8n.pt")

# ------------------------
# Upload Image
# ------------------------
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "png", "jpeg"])

# ------------------------
# KPI
# ------------------------
kpi1, kpi2, kpi3 = st.columns(3)

helmet_status = "Unknown"
risk_score = 0
alerts = 0

kpi1.metric("👷 Helmet Status", helmet_status)
kpi2.metric("⚠️ Risk Score", risk_score)
kpi3.metric("🚨 Alerts Today", alerts)

# ------------------------
# Main Logic
# ------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    results = model(img)[0]
    annotated = results.plot()

    st.image(annotated, caption="Detection Result", channels="BGR")

    # ------------------------
    # 🔥 ตรวจจับ person (ของจริง)
    # ------------------------
    helmet_detected = False
    person_detected = False

    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        if label == "person":
            person_detected = True

    # ------------------------
    # 🔥 Fake Helmet Logic (สำคัญมาก)
    # ------------------------
    if person_detected:
        if mode == "Force Safe":
            helmet_detected = True
        elif mode == "Force Risk":
            helmet_detected = False
        else:
            helmet_detected = random.choice([True, False])

    # ------------------------
    # Risk Calculation
    # ------------------------
    if helmet_detected:
        helmet_status = "Safe"
        risk_score = random.randint(0, 30)
    else:
        helmet_status = "No Helmet"
        risk_score = random.randint(60, 90)
        alerts += 1
        st.toast("🚨 No helmet detected!", icon="⚠️")

    # ------------------------
    # Update KPI
    # ------------------------
    kpi1.metric("👷 Helmet Status", helmet_status)
    kpi2.metric("⚠️ Risk Score", risk_score)
    kpi3.metric("🚨 Alerts Today", alerts)

    # ------------------------
    # Alert System
    # ------------------------
    st.subheader("🚨 Alert Status")

    if risk_score >= 70:
        st.error("🔴 CRITICAL: Stop machine immediately!")
    elif risk_score >= 40:
        st.warning("🟠 WARNING: Check system")
    else:
        st.success("🟢 SAFE")

    # ------------------------
    # AI Insight
    # ------------------------
    st.subheader("🤖 AI Insight")

    insights = []

    if not person_detected:
        insights.append("No worker detected in the scene")
    if person_detected and not helmet_detected:
        insights.append("Worker is not wearing helmet → high risk of injury")

    if insights:
        for i in insights:
            st.write(f"- {i}")
    else:
        st.write("All conditions are normal")

    # ------------------------
    # Recommendation
    # ------------------------
    st.subheader("📌 Recommendation")

    if risk_score >= 70:
        st.write("→ Shut down machine and alert supervisor immediately")
    elif risk_score >= 40:
        st.write("→ Inspect worker safety and machine condition")
    else:
        st.write("→ System operating normally")

    # ------------------------
    # Trend Chart
    # ------------------------
    st.subheader("📈 Risk Trend")
    chart_data = [random.randint(0, 100) for _ in range(30)]
    st.line_chart(chart_data)

# ------------------------
# Footer
# ------------------------
st.caption(f"🟢 LIVE | Last updated: {time.strftime('%H:%M:%S')}")
