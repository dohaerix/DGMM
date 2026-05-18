import cv2
import mediapipe as mp
import csv
import os
import time
from realsense_depth import DepthCamera   

# Create an instance of the DepthCamera
dc = DepthCamera()

# Define the landmark point for depth calculation
landmark_id_9 = 9

# Create OpenCV window
cv2.namedWindow("Image")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# FPS calculation
p_time = 0

# Larger sampling window
window_size = 15

# CSV Folder
csv_folder_path = 'input_gesture_new/Push'
os.makedirs(csv_folder_path, exist_ok=True)

# Function to get next CSV number
def get_next_file_number(folder):
    max_num = 0
    for file in os.listdir(folder):
        if file.startswith("all_landmarks_data_") and file.endswith(".csv"):
            try:
                num = int(file.split("_")[3].split(".")[0])
                max_num = max(max_num, num)
            except:
                pass
    return max_num + 1

file_number = get_next_file_number(csv_folder_path)
csv_path = os.path.join(csv_folder_path, f"all_landmarks_data_{file_number}.csv")
#video
# CSV fieldnames
fieldnames = ['X', 'Y', 'Z']

# Output video
video_folder = 'Output_Videos_new/Push'
os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, f'original_video_{file_number}.avi')

frame_width = 640
frame_height = 480
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

prev_depth = 50   # Default fallback depth (cm)

# Open CSV
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    frame_count = 0
    total_frames = 150

    while frame_count < total_frames:
        ret, depth_frame, color_frame = dc.get_frame()
        color_frame = cv2.flip(color_frame, 1)

        # Save raw video frame
        video_writer.write(color_frame)

        img_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Storage for landmarks
        landmarks_left = {}
        landmarks_right = {}

        if results.multi_hand_landmarks:
            for hand_lms, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):

                h, w, _ = color_frame.shape
                hand_label = handedness.classification[0].label

                # ---- DEPTH CALCULATION FOR LANDMARK 9 ----
                depth_sum = 0
                valid = 0
                px0 = int(hand_lms.landmark[landmark_id_9].x * w)
                py0 = int(hand_lms.landmark[landmark_id_9].y * h)

                for i in range(-window_size//2, window_size//2 + 1):
                    for j in range(-window_size//2, window_size//2 + 1):

                        px = px0 + i
                        py = py0 + j

                        if 0 <= px < w and 0 <= py < h:

                            d = depth_frame[py, px] / 10

                            # If invalid depth, search nearest valid pixel
                            if d == 0 or d < 15 or d > 120:
                                found = False
                                for dx in range(-3, 4):
                                    for dy in range(-3, 4):
                                        nx = px + dx
                                        ny = py + dy
                                        if 0 <= nx < w and 0 <= ny < h:
                                            nd = depth_frame[ny, nx] / 10
                                            if 15 < nd < 120:
                                                depth_sum += nd
                                                valid += 1
                                                found = True
                                                break
                                    if found:
                                        break
                            else:
                                depth_sum += d
                                valid += 1

                # Final depth
                if valid > 0:
                    avg_depth = int(depth_sum / valid)
                else:
                    avg_depth = prev_depth

                prev_depth = avg_depth

                # Store 21 landmarks for this hand
                for lid in range(21):
                    x = int(hand_lms.landmark[lid].x * w)
                    y = int(hand_lms.landmark[lid].y * h)
                    z = avg_depth

                    if hand_label == "Left":
                        landmarks_left[lid] = (x, y, z)
                    else:
                        landmarks_right[lid] = (x, y, z)

                # Draw skeleton
                mp_draw.draw_landmarks(color_frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                # Draw handedness label
                lx = int(hand_lms.landmark[0].x * w)
                ly = int(hand_lms.landmark[0].y * h) - 20
                cv2.putText(color_frame, hand_label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ---- WRITE LANDMARKS TO CSV ----
        for lid in range(21):
            writer.writerow({
                'X': landmarks_left.get(lid, (0, 0, 0))[0],
                'Y': landmarks_left.get(lid, (0, 0, 0))[1],
                'Z': landmarks_left.get(lid, (0, 0, 0))[2]
            })

        for lid in range(21):
            writer.writerow({
                'X': landmarks_right.get(lid, (0, 0, 0))[0],
                'Y': landmarks_right.get(lid, (0, 0, 0))[1],
                'Z': landmarks_right.get(lid, (0, 0, 0))[2]
            })

        # FPS Display
        c_time = time.time()
        fps_val = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(color_frame, f"FPS: {int(fps_val)}",
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # Show output
        cv2.imshow("Image", color_frame)

        if cv2.waitKey(1) == 27:
            break

        frame_count += 1

video_writer.release()
cv2.destroyAllWindows()

 