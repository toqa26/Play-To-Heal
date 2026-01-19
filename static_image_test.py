import cv2
from mode_gestureV2 import Mode

gesture_mode = Mode()

image = cv2.imread("sample.png")
if image is None:
    print("Not Found")
    exit()

processed = gesture_mode.process(image)

cv2.imshow("Static Hand Pipeline", processed)
cv2.waitKey(0)
cv2.destroyAllWindows()
