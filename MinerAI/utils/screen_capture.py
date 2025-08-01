# utils/screen_capture.py
import mss
import numpy as np

def capture_screen(region=None):
    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[0]
        else:
            monitor = {
                "top": region["top"],
                "left": region["left"],
                "width": region["width"],
                "height": region["height"]
            }
        img = np.array(sct.grab(monitor))
        return img[:, :, :3]  # Remove alpha