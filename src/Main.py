import cv2
import time
import Config
from HandDetector import HandDetector
from CloakEffect import CloakEffect

def main():
    cap = cv2.VideoCapture(Config.CAMERA_ID)
    cap.set(3, Config.FRAME_WIDTH)
    cap.set(4, Config.FRAME_HEIGHT)

    detector = HandDetector()
    effect = CloakEffect()
    time.sleep(1)
    
    for _ in range(30):
        cap.read()

    effect.capture_background(cap)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)

        img, hands_info = detector.find_hands_and_info(img, draw=False)

        if hands_info:
            for hand in hands_info:
                
                if hand["type"] == "Right": 
                    center_x, center_y = hand["center"]
                    
                    img = effect.apply_invisibility(img, (center_x, center_y))

        cv2.imshow("Sol El Sihri", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()