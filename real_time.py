import cv2
import joblib
import mediapipe as mp
import numpy as np
import math

# Load trained model and encoder
model = joblib.load("asl_model_geometric.pkl")
encoder = joblib.load("label_encoder_geometric.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Landmark indexes for geometric features
key_points = {
    'thumb_tip': 4,
    'thumb_ip': 3,
    'index_tip': 8,
    'index_mcp': 5,
    'middle_tip': 12,
    'middle_mcp': 9,
    'ring_tip': 16,
    'ring_mcp': 13,
    'pinky_tip': 20,
    'pinky_mcp': 17,
    'wrist': 0
}

def calc_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def calc_angle(a, b, c):
    ba = [a.x - b.x, a.y - b.y, a.z - b.z]
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]
    dot = sum(i*j for i, j in zip(ba, bc))
    mag_ba = math.sqrt(sum(i**2 for i in ba))
    mag_bc = math.sqrt(sum(i**2 for i in bc))
    if mag_ba * mag_bc == 0:
        return 0
    angle_rad = math.acos(dot / (mag_ba * mag_bc))
    return math.degrees(angle_rad)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = []

            # Add raw landmarks
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            # Add geometric distances
            for k1 in key_points:
                for k2 in key_points:
                    if k1 < k2:
                        d = calc_distance(
                            hand_landmarks.landmark[key_points[k1]],
                            hand_landmarks.landmark[key_points[k2]]
                        )
                        features.append(d)

            # Add angles
            angle1 = calc_angle(
                hand_landmarks.landmark[key_points['thumb_tip']],
                hand_landmarks.landmark[key_points['index_mcp']],
                hand_landmarks.landmark[key_points['middle_mcp']]
            )
            angle2 = calc_angle(
                hand_landmarks.landmark[key_points['index_mcp']],
                hand_landmarks.landmark[key_points['middle_mcp']],
                hand_landmarks.landmark[key_points['ring_mcp']]
            )
            features.extend([angle1, angle2])

            # Predict
            if len(features) == 63 + 55 + 2:  # 63 raw + 55 distances + 2 angles
                prediction = model.predict([features])
                predicted_label = encoder.inverse_transform(prediction)[0]
                cv2.putText(frame, f"Sign: {predicted_label}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ASL Recognition (Geometric)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
