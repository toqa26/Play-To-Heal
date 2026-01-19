import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

class Mode:
    name = "Gesture Detection (MediaPipe Tasks)"

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.connections = [
            (0,1),(1,2),(2,3),(3,4),       # إبهام
            (0,5),(5,6),(6,7),(7,8),       # سبابة
            (0,9),(9,10),(10,11),(11,12),  # وسطى
            (0,13),(13,14),(14,15),(15,16),# بنصر
            (0,17),(17,18),(18,19),(19,20) # خنصر
        ]

    @staticmethod
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _classify(self, lm):

        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        extended = 0

        for tip, pip in zip(finger_tips, finger_pips):
            if lm[tip][1] < lm[pip][1]:
                extended += 1

        thumb_tip = 4
        thumb_ip = 3
        index_mcp = 5

        d_tip = self.dist(lm[thumb_tip], lm[index_mcp])
        d_ip  = self.dist(lm[thumb_ip],  lm[index_mcp])

        thumb_extended = d_tip > d_ip
        if thumb_extended:
            extended += 1

        if extended == 0:
            return "Fist"
        elif extended == 5:
            return "Open"
        elif extended == 1:
            return "Thumb Up" if thumb_extended else "Pointing"
        elif extended == 2:
            if lm[8][1] < lm[6][1] and lm[12][1] < lm[10][1]:
                return "Peace"
            return "2-fingers"
        elif extended == 3:
            return "3-fingers"
        elif extended == 4:
            return "4-fingers"
        else:
            return f"{extended}-fingers"


    def process(self, frame):
        gesture = "N/A"
        handed = "N/A"
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            if result.handedness:
                try:
                    handed = result.handedness[0].category_name
                except AttributeError:
                    handed = result.handedness[0]

            h, w, _ = frame.shape
            lm_norm = [(lm.x, lm.y) for lm in hand_landmarks]

            gesture = self._classify(lm_norm)

            for start_idx, end_idx in self.connections:
                x1, y1 = int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h)
                x2, y2 = int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        cv2.putText(frame, f"Mode: {self.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(frame, f"Hand: {handed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        return frame, {"gesture": gesture, "handedness": handed}
