import cv2
import numpy as np


class FramePreprocessor:
    """Crops and rescales an input image."""

    def __init__(self, scale: np.array, crop: np.array):
        top, right, bottom, left = crop
        scale_width, scale_height = scale

        def process(frame):
            frame = frame.transpose((1, 2, 0))
            frame = cv2.resize(frame[top:-(bottom + 1), left:-(right + 1), :],
                               None,
                               fx=scale_width,
                               fy=scale_height,
                               interpolation=cv2.INTER_AREA)
            return frame

        self.process = process

    def __call__(self, frame: np.ndarray):
        return self.process(frame)
