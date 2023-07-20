import cv2
import pyramids
import numpy as np
import time
from scipy import signal
import scipy.fftpack as fftpack

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt0.xml")


def preprocess_frames(frames):
    video_frames = []
    face_rects = ()

    for frame in frames:
        roi_frame = frame
        # Detect face
        if len(video_frames) == 0:
            face_rects = faceCascade.detectMultiScale(frame, 1.3, 5)

        # Select ROI
        if len(face_rects) > 0:
            for x, y, w, h in face_rects:
                roi_frame = frame[y : y + h, x : x + w]
            if roi_frame.size != frame.size:
                roi_frame = cv2.resize(roi_frame, (500, 500))
                frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                frame[:] = roi_frame * (1.0 / 255)
                video_frames.append(frame)

    frame_ct = len(video_frames)
    fps = int(frame_ct / 3)
    print(frame_ct)

    return video_frames, frame_ct, fps


# Temporal bandpass filter with Fast-Fourier Transform
def fft_filter(video, freq_min, freq_max, fps):
    fft = fftpack.fft(video, axis=0)
    frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=0)
    result = np.abs(iff)
    result *= 100  # Amplification factor

    return result, fft, frequencies


def heart_rate_from_fft(fft, freqs, freq_min, freq_max):
    fft_maximums = []

    for i in range(fft.shape[0]):
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            fft_maximums.append(fftMap.max())
        else:
            fft_maximums.append(0)

    peaks, properties = signal.find_peaks(fft_maximums)
    max_peak = -1
    max_freq = 0

    # Find frequency with max amplitude in peaks
    for peak in peaks:
        if fft_maximums[peak] > max_freq:
            max_freq = fft_maximums[peak]
            max_peak = peak

    return freqs[max_peak] * 60


def heart_rate(frames):
    freq_min = 1
    freq_max = 1.8
    print(len(frames))

    face_frames, frame_ct, fps = preprocess_frames(frames)
    print("Building Laplacian video pyramid...")
    lap_video = pyramids.build_video_pyramid(face_frames)

    heart_rate = None

    for i, video in enumerate(lap_video):
        if i == 0 or i == len(lap_video) - 1:
            continue

        print("Running FFT and Eulerian magnification...")
        result, fft, frequencies = fft_filter(video, freq_min, freq_max, fps)
        print(result)
        lap_video[i] += result

        # Calculate heart rate
        print("Calculating heart rate...")
        try:
            heart_rate = heart_rate_from_fft(fft, frequencies, freq_min, freq_max)
            heart_rate = heart_rate
        except:
            heart_rate = None

    try:
        print(heart_rate)
    except:
        pass

    print(fps)

    return heart_rate
