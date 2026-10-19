"""Streamlit demo: PPE safety detection on an uploaded image or video.
Run:  streamlit run app.py
"""
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.model import DEFAULT_WEIGHTS, load_model, detect, annotate

st.set_page_config(page_title="PPE Safety Monitor", page_icon="🦺", layout="wide")


@st.cache_resource                      # load the model once, not on every click
def get_model():
    return load_model(DEFAULT_WEIGHTS)


model = get_model()

# ---------------- sidebar ----------------
st.sidebar.title("Settings")
conf = st.sidebar.slider("Detection confidence", 0.05, 0.9, 0.25, 0.05,
                         help="Boxes below this confidence are hidden")
alert_conf = st.sidebar.slider("Warning confidence", 0.1, 0.9, 0.40, 0.05,
                               help="NO-Hardhat / NO-Safety Vest above this raise a warning")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** YOLO26n, fine-tuned on 10 construction-safety classes")

# ---------------- main ----------------
st.title("🦺 Construction Site PPE Safety Monitor")
st.write("Upload a construction-site **image or video**. The model detects workers, PPE "
         "and machinery, and warns when a hard hat or safety vest is missing.")

file = st.file_uploader("Upload an image or video",
                        type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])


def show_results(counts, alerts):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected objects")
        if counts:
            st.dataframe(pd.DataFrame(sorted(counts.items()), columns=["Class", "Count"]),
                         hide_index=True, use_container_width=True)
        else:
            st.write("Nothing detected.")
    with col2:
        st.subheader("Safety status")
        if alerts:
            st.error(f"⚠️ {len(alerts)} violation(s) detected")
            st.dataframe(pd.DataFrame(alerts, columns=["Violation", "Confidence"]),
                         hide_index=True, use_container_width=True)
        else:
            st.success("✅ No hard-hat or vest violations detected")


if file is None:
    st.info("👆 Upload a file to start.")

elif file.type.startswith("image"):
    data = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    t0 = time.perf_counter()
    result, counts, alerts = detect(model, frame, conf, alert_conf)
    ms = 1000 * (time.perf_counter() - t0)

    annotated = annotate(result, counts, alerts)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption(f"Inference time: {ms:.0f} ms")
    show_results(counts, alerts)

else:  # video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file.read())
        video_path = tmp.name

    every_n = st.slider("Process every Nth frame (higher = faster)", 1, 10, 3)
    if st.button("▶ Run detection"):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_box, progress = st.empty(), st.progress(0.0)

        peak_counts, timeline, frame_no = {}, [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % every_n == 0:
                result, counts, alerts = detect(model, frame, conf, alert_conf)
                frame_box.image(cv2.cvtColor(annotate(result, counts, alerts), cv2.COLOR_BGR2RGB),
                                use_container_width=True)
                for k, v in counts.items():                    # most seen at the same moment
                    peak_counts[k] = max(peak_counts.get(k, 0), v)
                names = [a[0] for a in alerts]
                timeline.append({"time_s": round(frame_no / fps_in, 1),
                                 "NO-Hardhat": names.count("NO-Hardhat"),
                                 "NO-Safety Vest": names.count("NO-Safety Vest")})
            frame_no += 1
            progress.progress(min(frame_no / total, 1.0))
        cap.release()

        # ---------- summary ----------
        tl = pd.DataFrame(timeline).set_index("time_s")
        flagged = (tl.sum(axis=1) > 0)
        duration = frame_no / fps_in

        st.subheader("Video safety summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Video length", f"{duration:.1f} s")
        c2.metric("Time with violations", f"{100 * flagged.mean():.0f} %")
        c3.metric("Max workers without hard hat", int(tl["NO-Hardhat"].max()))
        c4.metric("Max workers without vest", int(tl["NO-Safety Vest"].max()))

        if flagged.any():
            st.error(f"⚠️ Violations seen between {tl.index[flagged].min()} s "
                     f"and {tl.index[flagged].max()} s")
        else:
            st.success("✅ No hard-hat or vest violations in this video")

        st.markdown("**Violations over time** (workers flagged in each processed frame)")
        st.line_chart(tl)

        st.markdown("**Most objects seen at the same moment**")
        st.dataframe(pd.DataFrame(sorted(peak_counts.items()), columns=["Class", "Max in one frame"]),
                     hide_index=True, use_container_width=True)