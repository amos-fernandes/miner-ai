# vision/vision_detector.py
import cv2
import numpy as np
from utils.screen_capture import capture_screen

class VisionDetector:
    def __init__(self, template_path="templates/miner_tile.png"):
        self.template = cv2.imread(template_path, 0)
        if self.template is None:
            raise FileNotFoundError(f"❌ Template não encontrado: {template_path}")
        self.template_w, self.template_h = self.template.shape[::-1]

    def detect_grid(self, region=None):
        screenshot = capture_screen(region)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

        points = []
        for pt in zip(*loc[::-1]):
            if not any(abs(p[0] - pt[0]) < 20 and abs(p[1] - pt[1]) < 20 for p in points):
                points.append((pt[0], pt[1]))
        points = sorted(points, key=lambda p: (p[1], p[0]))
        return points[:25]