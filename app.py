import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import main
import av
import streamlit as st

import time
import threading
import queue
from typing import List
import pandas as pd

result_queue: "queue.Queue[float]" = queue.Queue()

st.title("Heart Rate Detection Demo")
stream = "Live Video Stream"
upload_video = "Uploaded Video/File"
app_mode = st.selectbox("Choose the app mode", [stream, upload_video])
run = st.checkbox("Run")

# checks = st.columns(2)
# with checks[0]:
#     stream = st.checkbox("Live Stream")
# with checks[1]:
#     st.checkbox("Upload Video File")

# upload_video = st.checkbox("Upload Video File")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
heart_rates = []
last_heart_rate_update = time.time()

container = st.empty()
container2 = st.empty()

h_df = pd.DataFrame()
i = 0
while run:
    start_time = time.time()
    frame_list = []

    while time.time() - start_time <= 5:
        _, frame = camera.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(image)
        frame_list.append(image)

    heart_rate_text = main.heart_rate(frame_list)
    try:
        heart_rate_text = round(heart_rate_text, 2)
    except:
        pass
    text_position = (image.shape[1] - 150, 30)
    cv2.putText(
        image,
        str("HR: ") + str(heart_rate_text),
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )

    if heart_rate_text is not None and int(heart_rate_text) > 45:
        if heart_rate_text not in list(result_queue.queue):
            result_queue.put(heart_rate_text)

    nh_df = pd.DataFrame([{"heart_rate": result_queue.get(), "time": i}])
    h_df = pd.concat([h_df, nh_df], ignore_index=True)

    # if time.time() - timer() > 3.5:
    container.write("Heart Rate: " + str(heart_rate_text))
    container2.line_chart(data=h_df, x="time", y="heart_rate", use_container_width=True)

    # Reset the frame list and start time
    frame_list = []
    start_time = time.time()
    i = i + 5
