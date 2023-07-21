import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
import av
from typing import List
import streamlit as st
import cv2
import main
import asyncio
import pandas as pd

from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

result_deque: deque = deque([])

webrtc_ctx = webrtc_streamer(
    key="vitals-from-video",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True},
    async_processing=True,
)

container1 = st.empty()
container2 = st.empty()

result_queue = queue.Queue()
image_place = st.empty()

fct = 0
i = 0

heart_rate_text = 0
h_df = pd.DataFrame([{"heart_rate": heart_rate_text, "time": i}])

while True:
    start_time = time.time()
    frames = []
    if webrtc_ctx.video_receiver:
        while time.time() - start_time <= 3:
            try:
                frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
                image = frame.to_ndarray(format="rgb24")
                frames.append(image)
                image_place.image(image)
                fct = fct + 1
            except:
                pass
        else:
            try:
                heart_rate_text = main.heart_rate(frames)
                if heart_rate_text is not None and int(heart_rate_text) > 45:
                    result_queue.put(heart_rate_text)
            except:
                pass

            start_time = time.time()
            frames = []

        nh_df = pd.DataFrame([{"heart_rate": result_queue.get(), "time": i}])
        h_df = pd.concat([h_df, nh_df], ignore_index=True)

        i = i + 3

        container1.write("Heart Rate: " + str(heart_rate_text))
        container2.line_chart(
            data=h_df, x="time", y="heart_rate", use_container_width=True
        )
