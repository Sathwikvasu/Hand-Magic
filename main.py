import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

prev_distances = [0] * 5  # store for smoothing

# Fingertip landmarks
tips = [4, 8, 12, 16, 20]

# 🎨 Colors for each finger (BGR)
colors = [
    (255, 0, 0),    # Thumb
    (0, 255, 0),    # Index
    (0, 0, 255),    # Middle
    (255, 255, 0),  # Ring
    (255, 0, 255)   # Pinky
]

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hands_points = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            finger_points = []

            for tip in tips:
                x = int(handLms.landmark[tip].x * w)
                y = int(handLms.landmark[tip].y * h)
                finger_points.append((x, y))

            hands_points.append(finger_points)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # If two hands detected → draw styled lines
    if len(hands_points) == 2:
        hand1, hand2 = hands_points

        for i in range(5):
            (x1, y1) = hand1[i]
            (x2, y2) = hand2[i]

            # Distance
            dist = math.hypot(x2 - x1, y2 - y1)

            # Smooth
            dist = 0.7 * prev_distances[i] + 0.3 * dist
            prev_distances[i] = dist

            thickness = 2

            # ✨ Glow effect
            for t in range(4, 0, -1):
                cv2.line(img, (x1, y1), (x2, y2), colors[i], t * 2)

            # Main white line
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness, cv2.LINE_AA)

            # 🔵 Finger tip dots
            cv2.circle(img, (x1, y1), 6, colors[i], -1)
            cv2.circle(img, (x2, y2), 6, colors[i], -1)

    cv2.imshow("Stylized Finger Elastic", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()