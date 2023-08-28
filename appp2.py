import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import tempfile
import os
import cv2
from streamlit_webrtc import (RTCConfiguration,
                              VideoTransformerBase,
                              WebRtcMode,
                              webrtc_streamer,
                              )
from collections import deque
import uuid
from liteindex import DefinedIndex

import argparse
import asyncio
import json
import logging
import os
import ssl
import time
import mediapipe as mp
from aiohttp import web
from av import VideoFrame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# import torch
import pandas as pd
import requests
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import base64
from queue import Queue
import pyVHR as vhr
import numpy as np
from pyVHR.utils.errors import BVP_windowing
import math
import datetime
import scipy.signal as signal
from scipy.signal import find_peaks, butter, lfilter, filtfilt
from scipy.fft import fft, ifft

from scipy.fftpack import fft, fftfreq
from sklearn.decomposition import FastICA
from numba import prange
from scipy.signal import butter, sosfilt, sosfreqz
from biosppy.signals import ppg


vitals_demo_db = DefinedIndex(
    "vitals_demo_db",
    schema={
        "vital_type": "heart_rate",
        "uid": "",
        "filepath": "",
        "stats": {},
        "processed": 0,
        "original_file_name": "",
        "mode": "",
    },
    db_path="./dbs/vitals_demo.db",
    auto_key=False,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

wsize = 1
fps = 30

heart_rate_queue = asyncio.Queue()
times_queue = asyncio.Queue()

mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def LGI(signal):
    X = signal
    # Singular Value Decomposition (SVD) of the signal
    U, _, _ = np.linalg.svd(X)
    # Extract the first principal component (eigenvector) from U
    S = U[:, :, 0]
    # Expand the dimensions of S to match the shape of X
    S = np.expand_dims(S, 2)
    # Calculate the self-similarity tensor (SST) using the extracted principal component
    sst = np.matmul(S, np.swapaxes(S, 1, 2))
    # Create a matrix P with the same shape as X, representing the projection matrix
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    # Calculate the projection matrix P minus the self-similarity tensor to obtain correction matrix
    P = p - sst
    # Apply the correction matrix to the original signal X to remove artifacts
    Y = np.matmul(P, X)
    # Extract the second column of Y (interpolated signal after LGI)
    bvp = Y[:, 1, :]
    return bvp  # Return the interpolated signal


def get_bvp_from_rgb(signal):
    count = 0
    print("signal", signal.shape)
    copy_sig = np.array(signal, dtype=np.float32)
    bvp = np.zeros((0, 1), dtype=np.float32)
    print(
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    if copy_sig.shape[0] == 0:
        bvp = np.zeros((0, copy_sig.shape[2]), dtype=copy_sig.dtype)
    else:
        bvp = LGI(np.array(copy_sig))
    print("bvp", bvp)
    print("bvp shape", bvp.shape)

    return bvp


minHz = 0.7
maxHz = 3.0


def BPfilter(sig):
    print(sig.shape)
    copy_sig = np.array(sig, dtype=np.float32)
    copy_sig = np.swapaxes(copy_sig, 0, 1)
    copy_sig = np.swapaxes(copy_sig, 1, 2)
    x = np.array(np.swapaxes(copy_sig, 1, 2))
    # print(x)
    b, a = butter(N=6, Wn=[minHz, maxHz], fs=30, btype="bandpass")
    y = filtfilt(b, a, x, axis=1)
    # print(y)
    y = np.swapaxes(y, 1, 2)
    print("Pre Filtering Done.")
    return y


def post_process(bvp):
    bvp = np.expand_dims(bvp, axis=1)
    print(bvp.shape)
    x = np.array(np.swapaxes(bvp, 1, 2))
    b, a = butter(N=6, Wn=[minHz, maxHz], fs=30, btype="bandpass")
    bvp = filtfilt(b, a, x, axis=1)
    bvp = np.swapaxes(bvp, 1, 2)
    bvp = np.squeeze(bvp, axis=1)
    return bvp


class VideoTransformTrack(VideoTransformerBase):
    def __init__(self, track, stats, mode):
        super().__init__()
        if mode != "use-webcam":
            player = MediaPlayer(stats["filepath"])
            self.track = player.video
        else:
            self.track = track

        self.prev_mean_green = 0.0
        self.prev_std_green = 1.0
        self.prev_mean_rgb = 0.0
        self.prev_std_rgb = 1.0

        self.stats = stats
        self.start_time = time.time()
        self.loop = asyncio.get_event_loop()
        self.frame_count = 0
        self.frames_for_heart_rate = []
        self.recorded_frames = []
        self.intensities_list = []
        self.mean_green_all = []
        self.mean_green_roi = []
        self.times_green = 0
        self.mode = mode

        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        self.out = cv2.VideoWriter(
            self.stats["filepath"], fourcc, fps, (640, 480), isColor=True
        )
        self.count = 0

    async def recv(self):
        frame = await self.track.recv()
        self.stats["total_frames"] += 1

        while not self.track._queue.empty():
            frame = await self.track.recv()
            self.stats["total_frames"] += 1

        print(self.stats["total_frames"])

        img = frame.to_ndarray(format="bgr24")
        img_resize = cv2.resize(img, (640, 480))
        img_resize.flags.writeable = False

        new_image, _ = self.FacemeshAndROIselection(image=img_resize)

        new_frame = VideoFrame.from_ndarray(new_image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        if len(self.stats["GREEN_MEAN"]) > 150:
            for i, mean in enumerate(self.stats["GREEN_MEAN"][-150:]):
                if i % 30 == 0:
                    self.stats["GREEN"].append(mean)

        if self.mode == "use-webcam":
            self.out.write(img_resize)

        self.recorded_frames.append(img_resize)
        self.frames_for_heart_rate.append(img_resize)

        if self.stats["analyzed_frames"] - self.stats["green_measured_frames"] >= 150:
            green_input_array = np.array(self.stats["GREEN_MEAN"][-150:])
            # Green + Peak Detection
            asyncio.ensure_future(self.signal_processing(green_input_array))
            self.stats["green_measured_frames"] += 150

        if self.stats["analyzed_frames"] - self.stats["measured_frames"] >= 150:
            try:
                hr_frames = self.frames_for_heart_rate
                self.frames_for_heart_rate = hr_frames[-10:]
                self.stats["measured_frames"] += 150

                # LGI
                asyncio.ensure_future(
                    self.calculate_heart_rate_lgi(
                        signal=np.array(self.stats["raw_signal"][-150:])
                    )
                )

                # MTTS_CAN
                asyncio.ensure_future(
                    self.calculate_heart_rate_mtts(frames=hr_frames))
            except Exception as e:
                print(e)
                pass

        return new_frame

    async def calculate_heart_rate_lgi(self, signal):
        try:
            # BPfilter of raw signal
            filtered_signal = BPfilter(signal)
            bvp_lgi = get_bvp_from_rgb(filtered_signal)
            bvp_processed = post_process(bvp_lgi)
            self.stats["bvp_lgi"] += bvp_processed.tolist()
            bvp_win, _ = BVP_windowing(bvp_processed, wsize, fps, stride=1)
            bpmES = vhr.BPM.BVP_to_BPM_PSD_clustering_cuda(bvp_win, fps)
            print("bpmES", bpmES)
            try:
                heart_rates_new = []
                for bpm in bpmES:
                    heart_rates_new.append(float(bpm))
            except:
                pass
            try:
                bvp = np.expand_dims(bvp_lgi, axis=1)
                bvp_1d = bvp.flatten()
                ppg_new = ppg.ppg(signal=bvp_1d, sampling_rate=30.0)
                print(ppg_new)
                heart_rates_new1 = []
                for i in ppg_new["heart_rate"].tolist():
                    heart_rates_new1.append(float(i))
            except Exception as e:
                print(e)
                pass

            self.stats["heart_rate_lgi"] += heart_rates_new
            self.stats["heart_rate_lgi_2"] += heart_rates_new1
            print("LGI", self.stats["heart_rate_lgi"])
            print("LGI1", self.stats["heart_rate_lgi_2"])

        except Exception as e:
            print(e)
            pass

    async def calculate_heart_rate_mtts(self, frames):
        try:
            rf = np.array(frames)
            bvp_pred = vhr.deepRPPG.MTTS_CAN_deep(rf, fps, verb=1)
            bvp_win, timesES = BVP_windowing(bvp_pred, wsize, fps, stride=1)
            bpmES = vhr.BPM.BVP_to_BPM_PSD_clustering_cuda(bvp_win, fps)
            print(bpmES, timesES)
            try:
                heart_rates_new = []
                for bpm in bpmES:
                    heart_rates_new.append(float(bpm))
            except:
                pass
            try:
                times_list_new = []
                try:
                    prev_time_end = self.stats["times_list"][-1]
                except:
                    prev_time_end = 0
                for time in timesES:
                    times_list_new.append(prev_time_end + math.ceil(time))
            except Exception as e:
                print(e)
                pass

            self.stats["heart_rate_list"] += heart_rates_new
            self.stats["times_list"] += times_list_new

        except Exception as e:
            print(e)
            pass

    def FacemeshAndROIselection(self, image):
        height, width = 480, 640
        org_img = image
        crop_img = None
        mean_r = np.float32(0.0)
        mean_g = np.float32(0.0)
        mean_b = np.float32(0.0)
        mean_rgb = np.zeros((1, 3), dtype=np.float32)

        try:
            img_with_facemesh = mp_face_mesh.process(image)

            if img_with_facemesh.multi_face_landmarks:
                for face_landmarks in img_with_facemesh.multi_face_landmarks:
                    landmark_points = []
                    for i in range(0, 468):
                        x = int(face_landmarks.landmark[i].x * width)
                        y = int(face_landmarks.landmark[i].y * height)
                        p = [x, y]
                        landmark_points.append([x, y])
                    # Set ROI points
                    forehead = np.array(
                        (
                            landmark_points[10],
                            landmark_points[109],
                            landmark_points[108],
                            landmark_points[107],
                            landmark_points[55],
                            landmark_points[8],
                            landmark_points[285],
                            landmark_points[336],
                            landmark_points[337],
                            landmark_points[338],
                        )
                    )
                    left_cheek = np.array(
                        (
                            landmark_points[266],
                            landmark_points[426],
                            landmark_points[436],
                            landmark_points[416],
                            landmark_points[376],
                            landmark_points[352],
                            landmark_points[347],
                            landmark_points[330],
                        )
                    )
                    right_cheek = np.array(
                        (
                            landmark_points[36],
                            landmark_points[206],
                            landmark_points[216],
                            landmark_points[192],
                            landmark_points[147],
                            landmark_points[123],
                            landmark_points[117],
                            landmark_points[118],
                            landmark_points[101],
                        )
                    )

                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(
                        mask, [forehead, left_cheek, right_cheek], (255))
                    crop_img = cv2.bitwise_and(image, image, mask=mask)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )
                    r, g, b, a = cv2.mean(org_img, mask=mask)
                    self.stats["GREEN_MEAN"].append(g)
                    # [[20.790142 25.340956 36.73323 ]]
                    mean_r += r
                    mean_g += g
                    mean_b += b
                    mean_rgb[0, 0] = mean_r
                    mean_rgb[0, 1] = mean_g
                    mean_rgb[0, 2] = mean_b
                    self.stats["raw_signal"].append(mean_rgb)
                    # print(mean_rgb)

            self.stats["analyzed_frames"] += 1
        except Exception as e:
            print(e)
            crop_img = image
            pass

        return image, crop_img

    async def signal_processing(self, green_array):
        try:
            bvp_win, _ = BVP_windowing(green_array, wsize, fps, stride=1)
            bpmES = vhr.BPM.BVP_to_BPM_PSD_clustering_cuda(bvp_win, fps)

            # hr_peak = peaks(signal=green_array)
            # i = 0
            # # while i < 5:
            heart_rates_new = []
            for bpm in bpmES:
                heart_rates_new.append(float(bpm))
            self.stats["hr_green_peak"] += heart_rates_new
        except Exception as e:
            print(e)
            pass

    async def rppg_methods(self, crop_img):
        try:
            # Only Green Channel
            r, g, b = cv2.split(crop_img)
            mean_green = np.mean(g)
            self.stats["GREEN"].append(mean_green)

            # Average of RGB
            mean_rgb = np.mean(crop_img)
            self.stats["RGB"].append(mean_rgb)

            # Perform temporal normalization for mean_green and mean_rgb
            mean_green_normalized = self.temporal_normalize(
                mean_green, self.prev_mean_green, self.prev_std_green
            )
            self.stats["GREEN_normalized"].append(mean_green_normalized)
            mean_rgb_normalized = self.temporal_normalize(
                mean_rgb, self.prev_mean_rgb, self.prev_std_rgb
            )
            self.stats["RGB_normalized"].append(mean_rgb_normalized)

            # Update previous mean and standard deviation values for the next iteration
            self.prev_mean_green = mean_green
            self.prev_std_green = (
                np.std(self.stats["GREEN"]) if len(
                    self.stats["GREEN"]) > 1 else 1.0
            )
            self.prev_mean_rgb = mean_rgb
            self.prev_std_rgb = (
                np.std(self.stats["RGB"]) if len(
                    self.stats["RGB"]) > 1 else 1.0
            )
        except Exception as e:
            print(e)
            pass

    def temporal_normalize(self, value, prev_mean, prev_std):
        return (value - prev_mean) / prev_std


file_path = None


def create_player():
    if file_path is not None:
        return MediaPlayer(str(file_path))
    else:
        return MediaPlayer(
            "1:none",
            format="avfoundation",
            options={"framerate": "30", "video_size": "1280x720"},
        )


def main():
    st.title("Healiom Vitals Demo")

    # align all below components in a single line
    st.write("Select the vitals to measure:")
    checks = st.columns(4)
    with checks[0]:
        st.checkbox("Heart Rate", value=True)
    with checks[1]:
        st.checkbox("HRV", value=False, disabled=True)
    with checks[2]:
        st.checkbox("Respiratory Rate", value=False, disabled=True)
    with checks[3]:
        st.checkbox("Blood Pressure", value=False, disabled=True)

    st.write("Modes:")
    mode_options = ["Using webcam video",
                    "Upload video file (.mp4)", "Validation Mode"]
    mode = st.radio("Select mode:", mode_options)

    st.write("Methods:")
    checks1 = st.columns(3)
    with checks1[0]:
        st.checkbox("rppg-GREEN", value=True, disabled=True)
    with checks1[1]:
        st.checkbox("rppg-LGI", value=True, disabled=True)
    with checks1[2]:
        st.checkbox("MTTS-CAN", value=True, disabled=True)

    start_button = st.button("Start")
    stop_button = st.button("Stop")
    upload_button = st.button("Upload")

    # fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    # self.out = cv2.VideoWriter(
    #     self.stats["filepath"], fourcc, fps, (640, 480), isColor=True
    # )
    # self.count = 0

    # def in_recorder_factory() -> MediaRecorder:
    #     return MediaRecorder(
    #         str(in_file), format="flv"
    #     )  # HLS does not work. See https://github.com/aiortc/aiortc/issues/331

    # def out_recorder_factory() -> MediaRecorder:
    #     return MediaRecorder(str(out_file), format="flv")

    if start_button:
        st.write("Starting...")
        uid = uuid.uuid4()
        print(uid)
        vitals_demo_db.set(uid, "uid")
        # Perform necessary actions when the Start button is clicked

    if stop_button:
        st.write("Stopping...")
        # Perform necessary actions when the Stop button is clicked

    if upload_button:
        uploaded_file = st.file_uploader("Upload video file", type=["mp4"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
                st.write(f"Uploaded file: {uploaded_file.name}")
        # Perform necessary actions when the Upload button is clicked

    if st.checkbox("Enable Webcam", value=True):
        self, track, stats, mode = None, None, None, None
        webrtc_streamer(
            key="webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoTransformTrack(
                self, track, stats, mode),
            async_processing=True,
            source_video_track=None,
            player_factory=create_player,
        )

    # desired_playing_state: Optional[bool] = None,
    # player_factory: Optional[MediaPlayerFactory] = None,
    # in_recorder_factory: Optional[MediaRecorderFactory] = None,
    # out_recorder_factory: Optional[MediaRecorderFactory] = None,
    # video_frame_callback: Optional[VideoFrameCallback] = None,
    # audio_frame_callback: Optional[AudioFrameCallback] = None,
    # queued_video_frames_callback: Optional[QueuedVideoFramesCallback] = None,
    # queued_audio_frames_callback: Optional[QueuedAudioFramesCallback] = None,
    # on_video_ended: Optional[MediaEndedCallback] = None,
    # video_receiver_size: int = 4,
    # audio_receiver_size: int = 4,
    # video_html_attrs: Optional[Union[VideoHTMLAttributes, Dict]] = None,
    # audio_html_attrs: Optional[Union[AudioHTMLAttributes, Dict]] = None,
    # on_change: Optional[Callable] = None,

        st.write("Recording data...")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Heart Rate', 'Respiratory Rate', 'Blood Pressure'])
        st.line_chart(chart_data)

    # components.html(
    #     '<div class="video-results"><div id="video-container"><video id="video" width="640" height="480" playsinline></video></div><div id="chart-container"><canvas id="chartDiv"></canvas></div></div>',
    #     height=600,
    # )


if __name__ == "__main__":
    main()
