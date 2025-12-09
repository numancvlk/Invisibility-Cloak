import cv2
import mediapipe as mp
import Config

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=Config.DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands_and_info(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        hands_info = []

        if self.results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                
                label = hand_type.classification[0].label 
                
                h, w, c = img.shape
                cx, cy = int(hand_lms.landmark[9].x * w), int(hand_lms.landmark[9].y * h)

                hands_info.append({"type": label, "center": (cx, cy), "landmarks": hand_lms})

                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        
        return img, hands_info