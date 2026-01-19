import cv2
from mode_gesture import Mode

gesture_mode = Mode()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, result = gesture_mode.process(frame)

    if result["gesture"] != "N/A":
        print(f"Hand: {result['handedness']}, Gesture: {result['gesture']}")

    cv2.imshow("Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
