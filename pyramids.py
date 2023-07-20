import cv2
import numpy as np

# Build Gaussian image pyramid
def build_gaussian_pyramid(img, levels):
    float_img = np.ndarray(shape=img.shape, dtype="float")
    float_img[:] = img
    pyramid = [float_img]

    for i in range(levels):
        float_img = cv2.pyrDown(float_img)
        pyramid.append(float_img)
    print("pyramid", len(pyramid))
    return pyramid


# Build Laplacian image pyramid from Gaussian pyramid
def build_laplacian_pyramid(img, levels):
    gaussian_pyramid = build_gaussian_pyramid(img, levels)
    laplacian_pyramid = []

    for i in range(levels - 1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1])
        (height, width, depth) = upsampled.shape
        gaussian_pyramid[i] = cv2.resize(gaussian_pyramid[i], (height, width))
        diff = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(diff)

    laplacian_pyramid.append(gaussian_pyramid[-1])
    print("laplacian_pyramid == ", len(laplacian_pyramid))

    return laplacian_pyramid


# Build video pyramid by building Laplacian pyramid for each frame
def build_video_pyramid(frames):
    lap_video = []

    for i, frame in enumerate(frames):
        pyramid = build_laplacian_pyramid(frame, 3)
        print("laplacian_pyramid == ", len(pyramid))
        for j in range(3):
            if i == 0:
                lap_video.append(
                    np.zeros((len(frames), pyramid[j].shape[0], pyramid[j].shape[1], 3))
                )
            lap_video[j][i] = pyramid[j]

    print("lap_video == ", len(lap_video))
    return lap_video
