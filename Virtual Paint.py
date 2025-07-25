import cv2
import numpy as np

# Open webcam at low resolution
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# HSV ranges for blue pen
blue_lower = np.array([100, 150, 50])
blue_upper = np.array([140, 255, 255])

# HSV ranges for red pen (two parts of hue)
red_lower1 = np.array([0, 150, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 150, 50])
red_upper2 = np.array([180, 255, 255])

canvas = np.zeros((240, 320, 3), dtype=np.uint8)
points = {"blue": [], "red": []}

def track_pen(mask, pts, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w // 2, y + h // 2)
            pts.append(center)
            for i in range(1, len(pts)):
                if pts[i - 1] and pts[i]:
                    cv2.line(canvas, pts[i - 1], pts[i], color, 3)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Combine two red masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    track_pen(blue_mask, points["blue"], (255, 0, 0))
    track_pen(red_mask, points["red"], (0, 0, 255))

    output = cv2.add(frame, canvas)
    cv2.imshow("Virtual Paint", output)

    # Optional: show mask windows for debugging
    # cv2.imshow("Red Mask", red_mask)
    # cv2.imshow("Blue Mask", blue_mask)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('c'):
        canvas[:] = 0
        points = {"blue": [], "red": []}

cap.release()
cv2.destroyAllWindows()
