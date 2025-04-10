import os
import cv2
import csv
import math
import mediapipe as mp

# Set your dataset path
dataset_path = "signenv/Dataset_ASL"
csv_file = "asl_landmarks_geometric.csv"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Landmarks of interest (thumb, index, middle, ring, pinky tips & bases)
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
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def calc_angle(a, b, c):
    # returns angle ABC in degrees
    ba = [a.x - b.x, a.y - b.y, a.z - b.z]
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]
    dot = sum(i*j for i, j in zip(ba, bc))
    mag_ba = math.sqrt(sum(i**2 for i in ba))
    mag_bc = math.sqrt(sum(i**2 for i in bc))
    if mag_ba * mag_bc == 0:
        return 0
    angle_rad = math.acos(dot / (mag_ba * mag_bc))
    return math.degrees(angle_rad)

# Prepare CSV header
header = ['label']

# Raw landmark coords
for i in range(21):
    header.extend([f'{i}_x', f'{i}_y', f'{i}_z'])

# Add geometric features: distances and angles
for name1 in key_points:
    for name2 in key_points:
        if name1 < name2:
            header.append(f'dist_{name1}_{name2}')
header.append("angle_thumb_index_middle")
header.append("angle_index_middle_ring")

# Write header
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for set_folder in os.listdir(dataset_path):
        set_path = os.path.join(dataset_path, set_folder)
        if not os.path.isdir(set_path):
            continue

        for label in os.listdir(set_path):
            label_folder = os.path.join(set_path, label)
            if not os.path.isdir(label_folder):
                continue

            print(f"🟢 Processing label: {label}")
            count = 0

            for image_name in os.listdir(label_folder)[:300]:
                image_path = os.path.join(label_folder, image_name)
                image = cv2.imread(image_path)

                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    row = [label]

                    # Add raw landmarks
                    for lm in hand.landmark:
                        row.extend([lm.x, lm.y, lm.z])

                    # Add geometric features
                    for k1 in key_points:
                        for k2 in key_points:
                            if k1 < k2:
                                d = calc_distance(
                                    hand.landmark[key_points[k1]],
                                    hand.landmark[key_points[k2]]
                                )
                                row.append(d)

                    # Add angles
                    angle1 = calc_angle(
                        hand.landmark[key_points['thumb_tip']],
                        hand.landmark[key_points['index_mcp']],
                        hand.landmark[key_points['middle_mcp']]
                    )
                    angle2 = calc_angle(
                        hand.landmark[key_points['index_mcp']],
                        hand.landmark[key_points['middle_mcp']],
                        hand.landmark[key_points['ring_mcp']]
                    )
                    row.append(angle1)
                    row.append(angle2)

                    writer.writerow(row)
                    count += 1

            print(f"✅ {count} samples processed for label: {label}")

hands.close()
print("📄 CSV with geometric features created.")
