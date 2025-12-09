import cv2
import numpy as np
import Config

class CloakEffect:
    def __init__(self):
        self.background = None

    def capture_background(self, cap, num_frames=30):
        backgrounds = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if ret:
                backgrounds.append(frame)
        
        if backgrounds:
            self.background = np.median(backgrounds, axis=0).astype(np.uint8)
            print("Arka plan başarıyla kaydedildi!")

    def apply_invisibility(self, frame, center_point):
        if self.background is None:
            return frame

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center_point, Config.CLOAK_RADIUS, 255, -1)

        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        mask_3d = cv2.merge([mask, mask, mask]) / 255.0
        
        foreground = frame.astype(float)
        background_part = self.background.astype(float)

        final_image = (background_part * mask_3d) + (foreground * (1 - mask_3d))
        
        return final_image.astype(np.uint8)