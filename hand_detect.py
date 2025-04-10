import os
import cv2
import csv
import mediapipe as mp

# Set your dataset path
dataset_path = "signenv/Dataset_ASL"

# Output CSV
csv_file = "asl_landmarks.csv"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Prepare CSV Header
header = ['label']
for i in range(21):  # 21 landmarks
    header.extend([f'{i}_x', f'{i}_y', f'{i}_z'])

# Write header to CSV
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for set_folder in os.listdir(dataset_path):  # 'asl_alphabet_train' and 'asl_alphabet_test'
        set_path = os.path.join(dataset_path, set_folder)
        if not os.path.isdir(set_path):
            continue

        for label in os.listdir(set_path):  # 'A', 'B', 'C', ...
            label_folder = os.path.join(set_path, label)
            if not os.path.isdir(label_folder):
                continue

            print(f"🟢 Processing label: {label}")
            count = 0

            for image_name in os.listdir(label_folder)[:300]:  # limit to 300 per label
                image_path = os.path.join(label_folder, image_name)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"⚠️ Could not read image: {image_path}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        row = [label]
                        for lm in hand_landmarks.landmark:
                            row.extend([lm.x, lm.y, lm.z])
                        writer.writerow(row)
                        count += 1
                        break  # Only use first detected hand

            print(f"✅ {count} samples processed for label: {label}")

hands.close()
print("📄 CSV creation complete.")
