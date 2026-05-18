'''
dgmm_testing.py

# THIS DGMM FLOW:
#   record_landmarks()
#   feature_tensor = process_single_csv("landmark.csv")
#   features = mean + IQR
#   scaler.transform(features)
#   autoencoder.encode(features)
#   dgmm.predict(latent)
#   cluster_to_gesture[cluster]
#
# This tests ONE gesture at a time.
# It does NOT continuously predict while the model is running.
# ------------------------------------------------------------
'''

import os
import csv
import json
import time
from glob import glob

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TOTAL_FRAMES = 150
FPS = 30

CAMERA_INDEX = 0
FRAME_WIDTH = 480
FRAME_HEIGHT = 360

CSV_PATH = "landmark.csv"
VIDEO_PATH = "original_video.avi"

AUTOENCODER_PATH = "autoencoder.keras"
DGMM_PATH = "dgmm_model.pkl"
SCALER_PATH = "scaler.pkl"


CLUSTER_TO_GESTURE_JSON = "cluster_to_gesture.json"

CLUSTER_TO_GESTURE = {
    0: "Gesture_0",
    1: "Gesture_1",
    2: "Gesture_2",
    3: "Gesture_3",
    4: "Gesture_4",
    5: "Gesture_5",
    6: "Gesture_6",
    7: "Gesture_7",
}


# ============================================================
# FILE RESOLUTION
# ============================================================

def resolve_file(preferred_path, patterns, description):
    if os.path.exists(preferred_path):
        return preferred_path

    candidates = []
    for pattern in patterns:
        candidates.extend(glob(pattern))

    candidates = sorted(set([p for p in candidates if os.path.isfile(p)]))

    if len(candidates) == 1:
        print(f"[INFO] Auto-detected {description}: {candidates[0]}")
        return candidates[0]

    if len(candidates) > 1:
        print(f"\n[ERROR] Multiple possible {description} files found:")
        for i, p in enumerate(candidates, start=1):
            print(f"  {i}. {p}")
        raise FileExistsError(
            f"Multiple {description} files found. Rename the correct file to {preferred_path}"
        )

    print(f"\n[ERROR] Could not find {description}.")
    print("[INFO] Files in current folder:")
    for p in sorted(os.listdir(".")):
        print("  ", p)

    raise FileNotFoundError(
        f"Missing {description}. Expected {preferred_path}, or one of {patterns}"
    )


# ============================================================
# KERAS AUTOENCODER LOADING
# ============================================================

def load_keras_backend():
    try:
        import keras
        return keras
    except Exception:
        from tensorflow import keras
        return keras


keras = load_keras_backend()


@keras.saving.register_keras_serializable(package="models")
class Autoencoder(keras.Model):
    '''
    AutoEncoder:
        Input: 62 features (mean + IQR)
        Latent: 40 dimensions (configurable)
        Output: 62 features (reconstruction)
    '''
    
    def __init__(self, input_dim=62, latent_dim=40, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        self.encoder_dense1 = keras.layers.Dense(
            128,
            activation="relu",
            name="encoder_dense1"
        )
        self.encoder_dense2 = keras.layers.Dense(
            self.latent_dim,
            activation="relu",
            name="encoder_dense2"
        )
        self.decoder_dense1 = keras.layers.Dense(
            128,
            activation="relu",
            name="decoder_dense1"
        )
        self.decoder_dense2 = keras.layers.Dense(
            self.input_dim,
            activation="sigmoid",
            name="decoder_dense2"
        )

    def encode(self, x, training=False):
        x = self.encoder_dense1(x)
        return self.encoder_dense2(x)

    def decode(self, z, training=False):
        x = self.decoder_dense1(z)
        return self.decoder_dense2(x)

    def call(self, x, training=False):
        z = self.encode(x, training=training)
        return self.decode(z, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        })
        return config


def load_models():
    autoencoder_path = resolve_file(
        AUTOENCODER_PATH,
        ["autoencoder*.keras", "*autoencoder*.keras", "*.keras"],
        "autoencoder"
    )

    dgmm_path = resolve_file(
        DGMM_PATH,
        ["dgmm_model.pkl", "dgmm*.pkl", "*dgmm*.pkl", "*gmm*.pkl"],
        "DGMM/GMM model"
    )

    scaler_path = resolve_file(
        SCALER_PATH,
        ["scaler.pkl", "scaler*.pkl", "*scaler*.pkl"],
        "scaler"
    )

    print("[INFO] Loading scaler:", scaler_path)
    scaler = joblib.load(scaler_path)

    print("[INFO] Loading DGMM/GMM:", dgmm_path)
    dgmm = joblib.load(dgmm_path)

    print("[INFO] Loading autoencoder:", autoencoder_path)
    autoencoder = keras.models.load_model(
        autoencoder_path,
        custom_objects={
            "Autoencoder": Autoencoder,
            "models>Autoencoder": Autoencoder,
        },
        compile=False
    )

    dummy_x = np.zeros((1, getattr(scaler, "n_features_in_", 62)), dtype=np.float32)
    dummy_z = autoencoder.encode(dummy_x, training=False)
    if hasattr(dummy_z, "numpy"):
        dummy_z = dummy_z.numpy()

    print("[INFO] Scaler input dim       :", getattr(scaler, "n_features_in_", "unknown"))
    print("[INFO] Autoencoder latent dim :", dummy_z.shape[1])
    print("[INFO] DGMM input dim         :", getattr(dgmm, "n_features_in_", "unknown"))
    print("[INFO] DGMM components        :", getattr(dgmm, "n_components", "unknown"))

    if getattr(scaler, "n_features_in_", 62) != 62:
        raise ValueError(
            f"Scaler expects {getattr(scaler, 'n_features_in_', 'unknown')} features, "
            "but this testing code produces 62 features after mean+IQR."
        )

    if hasattr(dgmm, "n_features_in_") and dgmm.n_features_in_ != dummy_z.shape[1]:
        raise ValueError(
            f"DGMM expects {dgmm.n_features_in_} features, "
            f"but autoencoder gives {dummy_z.shape[1]}."
        )

    return scaler, autoencoder, dgmm


def load_cluster_to_gesture():
    if os.path.exists(CLUSTER_TO_GESTURE_JSON):
        with open(CLUSTER_TO_GESTURE_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)

        mapping = {int(k): str(v) for k, v in raw.items()}
        print("[INFO] Loaded cluster label file:", CLUSTER_TO_GESTURE_JSON)
        return mapping

    print("[WARNING] cluster_to_gesture.json not found.")
    print("[WARNING] Using CLUSTER_TO_GESTURE dictionary from this script.")
    return CLUSTER_TO_GESTURE


# ============================================================
# 1. RECORD LANDMARKS
# Exact same style as your old MediaPipe code:
# write 42 rows per frame:
#   first 21 = Left hand
#   next 21  = Right hand
# ============================================================

def record_landmarks(total_frames=150, csv_path="landmark.csv", video_path="original_video.avi"):
    print("[INFO] Initializing MediaPipe...")

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mp_draw = mp.solutions.drawing_utils

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(
        video_path,
        fourcc,
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    if os.name == "nt":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        hands.close()
        video_writer.release()
        raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("[INFO] Camera opened.")
    print(f"[INFO] Recording {total_frames} frames.")
    print("[INFO] Perform ONE gesture now.")
    print("[INFO] Press ESC or Q to cancel.")

    fieldnames = ["X", "Y", "Z"]

    p_time = time.time()
    frame_count = 0
    cancelled = False

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame capture failed.")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)

            video_writer.write(frame)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            landmarks_left = {}
            landmarks_right = {}

            if results.multi_hand_landmarks:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    h, w, _ = frame.shape
                    hand_label = handedness.classification[0].label

                    for lid in range(21):
                        x = int(hand_lms.landmark[lid].x * w)
                        y = int(hand_lms.landmark[lid].y * h)
                        z = int(hand_lms.landmark[lid].z * 1000)

                        if hand_label == "Left":
                            landmarks_left[lid] = (x, y, z)
                        else:
                            landmarks_right[lid] = (x, y, z)

                    mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS
                    )

                    lx = int(hand_lms.landmark[0].x * w)
                    ly = int(hand_lms.landmark[0].y * h) - 20

                    cv2.putText(
                        frame,
                        hand_label,
                        (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            for lid in range(21):
                x, y, z = landmarks_left.get(lid, (0, 0, 0))
                writer.writerow({"X": x, "Y": y, "Z": z})

            for lid in range(21):
                x, y, z = landmarks_right.get(lid, (0, 0, 0))
                writer.writerow({"X": x, "Y": y, "Z": z})

            frame_count += 1

            c_time = time.time()
            fps_val = 1.0 / (c_time - p_time + 1e-6)
            p_time = c_time

            cv2.putText(
                frame,
                f"FPS: {int(fps_val)}",
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Recording: {frame_count}/{total_frames}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Do one gesture. Prediction after recording.",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow("Image", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q"), ord("Q")]:
                cancelled = True
                print("[INFO] Cancelled by user.")
                break

    hands.close()
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    if cancelled:
        return False

    if frame_count != total_frames:
        print(f"[ERROR] Only recorded {frame_count}/{total_frames} frames.")
        return False

    print("[INFO] Recording finished.")
    print(f"[INFO] CSV saved: {csv_path}")
    return True


# ============================================================
# 2. FEATURE EXTRACTION
# This is copied to match your training features.py logic.
# It processes ONE csv gesture only.
# ============================================================

def preprocess_frame(frame_landmarks, eps=1e-6):
    assert frame_landmarks.shape == (42, 2)

    normalized = np.zeros_like(frame_landmarks)

    hands = {
        0: {"indices": np.arange(0, 21), "wrist": 0, "index_mcp": 5, "pinky_mcp": 17},
        1: {"indices": np.arange(21, 42), "wrist": 21, "index_mcp": 26, "pinky_mcp": 38},
    }

    for hand_id, hand in hands.items():
        idx = hand["indices"]
        hand_landmarks = frame_landmarks[idx]

        if np.allclose(hand_landmarks, 0.0):
            continue

        wrist = frame_landmarks[hand["wrist"]]
        hand_landmarks = hand_landmarks - wrist

        p1 = frame_landmarks[hand["index_mcp"]] - wrist
        p2 = frame_landmarks[hand["pinky_mcp"]] - wrist
        palm_width = np.linalg.norm(p1 - p2)

        if palm_width < eps:
            continue

        hand_landmarks = hand_landmarks / (palm_width + eps)
        normalized[idx] = hand_landmarks

    return normalized


def preprocess_video(csv_data, num_frames=150):
    assert csv_data.shape[0] == num_frames * 42, (
        f"CSV row mismatch. Expected {num_frames * 42}, got {csv_data.shape[0]}"
    )

    video = csv_data.reshape(num_frames, 42, 3)[..., :2]

    normalized_video = np.zeros_like(video)
    raw_wrist_positions = np.zeros((num_frames, 2, 2))

    for t in range(num_frames):
        frame = video[t]

        raw_wrist_positions[t, 0] = frame[0]
        raw_wrist_positions[t, 1] = frame[21]

        normalized_video[t] = preprocess_frame(frame)

    return normalized_video, raw_wrist_positions


def compute_angle(a, b, c, eps=1e-6):
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba) + eps
    norm_bc = np.linalg.norm(bc) + eps

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)


def extract_frame_features(frame_landmarks):
    features = []

    fingers = [
        (1, 2, 4),
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20),
    ]

    fingertip_indices = [4, 8, 12, 16, 20]

    for hand_id in range(2):
        offset = 0 if hand_id == 0 else 21
        hand = frame_landmarks[offset:offset + 21]

        if np.allclose(hand, 0.0):
            features.extend([0.0] * 11)
            continue

        wrist = hand[0]

        for (a, b, c) in fingers:
            angle = compute_angle(hand[a], hand[b], hand[c])
            features.append(angle)

        fingertip_points = []

        for idx in fingertip_indices:
            dist = np.linalg.norm(hand[idx] - wrist)
            features.append(dist)
            fingertip_points.append(hand[idx])

        fingertip_points = np.array(fingertip_points)
        dists = []

        for i in range(len(fingertip_points)):
            for j in range(i + 1, len(fingertip_points)):
                dists.append(np.linalg.norm(fingertip_points[i] - fingertip_points[j]))

        features.append(np.mean(dists))

    return np.array(features, dtype=np.float32)


def compute_wrist_velocity(wrist_positions, fps=30):
    T = wrist_positions.shape[0]
    velocity = np.zeros_like(wrist_positions)

    dt = 1.0 / fps

    for t in range(1, T):
        velocity[t] = (wrist_positions[t] - wrist_positions[t - 1]) / dt

    return velocity


def build_video_feature_tensor(normalized_video, raw_wrist_positions, fps=30):
    T = normalized_video.shape[0]
    frame_features = []

    wrist_velocity = compute_wrist_velocity(raw_wrist_positions, fps)

    for t in range(T):
        static_feat = extract_frame_features(normalized_video[t])

        wrist_pos_hand1 = raw_wrist_positions[t, 0]
        wrist_pos_hand2 = raw_wrist_positions[t, 1]

        wrist_vel_hand1 = wrist_velocity[t, 0]
        wrist_vel_hand2 = wrist_velocity[t, 1]

        if (
            not np.allclose(raw_wrist_positions[t, 0], 0.0)
            and not np.allclose(raw_wrist_positions[t, 1], 0.0)
        ):
            inter_hand_dist = np.linalg.norm(raw_wrist_positions[t, 0] - raw_wrist_positions[t, 1])
        else:
            inter_hand_dist = 0.0

        combined = np.concatenate([
            static_feat,
            wrist_pos_hand1,
            wrist_pos_hand2,
            wrist_vel_hand1,
            wrist_vel_hand2,
            [inter_hand_dist],
        ])

        frame_features.append(combined)

    return np.array(frame_features, dtype=np.float32)


def process_single_csv(csv_path, num_frames=150, fps=30):
    csv_data = pd.read_csv(csv_path, skiprows=1, header=None).values

    normalized_video, raw_wrist_positions = preprocess_video(
        csv_data,
        num_frames=num_frames
    )

    video_features = build_video_feature_tensor(
        normalized_video,
        raw_wrist_positions,
        fps=fps
    )

    if video_features.shape != (num_frames, 31):
        raise ValueError(
            f"Expected one gesture feature shape ({num_frames}, 31), "
            f"got {video_features.shape}"
        )

    return np.expand_dims(video_features, axis=0).astype(np.float32)


def aggregate_sequence(feature_tensor):
    mean = np.mean(feature_tensor, axis=1)
    iqr = np.percentile(feature_tensor, 75, axis=1) - np.percentile(feature_tensor, 25, axis=1)
    aggregated = np.concatenate([mean, iqr], axis=1)
    return aggregated.astype(np.float32)


# ============================================================
# 3. DGMM PREDICTION
# ============================================================

def predict_one_gesture(csv_path, scaler, autoencoder, dgmm, cluster_to_gesture):
    print("[INFO] Processing landmark.csv...")

    feature_tensor = process_single_csv(
        csv_path,
        num_frames=TOTAL_FRAMES,
        fps=FPS
    )

    print("[INFO] Feature tensor shape:", feature_tensor.shape)

    features = aggregate_sequence(feature_tensor)
    print("[INFO] Aggregated feature shape:", features.shape)

    scaled = scaler.transform(features).astype(np.float32)

    latent = autoencoder.encode(scaled, training=False)
    if hasattr(latent, "numpy"):
        latent = latent.numpy()
    latent = np.asarray(latent, dtype=np.float32)

    print("[INFO] Latent shape:", latent.shape)

    cluster = int(dgmm.predict(latent)[0])

    if hasattr(dgmm, "predict_proba"):
        probs = dgmm.predict_proba(latent)[0]
        confidence = float(np.max(probs))

        print("\n[DEBUG] DGMM posterior probabilities:")
        for i, p in enumerate(probs):
            name = cluster_to_gesture.get(i, f"Cluster {i}")
            print(f"  Cluster {i:02d} / {name}: {p:.6f}")
    else:
        probs = None
        confidence = float("nan")

    if hasattr(dgmm, "score_samples"):
        log_likelihood = float(dgmm.score_samples(latent)[0])
    else:
        log_likelihood = float("nan")

    label = cluster_to_gesture.get(cluster, f"Cluster {cluster}")

    return {
        "cluster": cluster,
        "label": label,
        "confidence": confidence,
        "log_likelihood": log_likelihood,
        "probs": probs,
    }


def show_final_result(result):
    img = np.zeros((330, 720, 3), dtype=np.uint8)

    cv2.putText(img, "FINAL DGMM RESULT", (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img, f"Predicted label: {result['label']}", (40, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(img, f"Cluster ID: {result['cluster']}", (40, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)

    cv2.putText(img, f"Confidence: {result['confidence']:.6f}", (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(img, f"Log-likelihood: {result['log_likelihood']:.6f}", (40, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Final Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("DGMM TESTING - EXACT MEDIAPIPE STYLE")
    print("=" * 80)

    scaler, autoencoder, dgmm = load_models()
    cluster_to_gesture = load_cluster_to_gesture()

    print("\n[INFO] Current cluster label mapping:")
    for k in sorted(cluster_to_gesture.keys()):
        print(f"  Cluster {k} -> {cluster_to_gesture[k]}")

    recorded = record_landmarks(
        total_frames=TOTAL_FRAMES,
        csv_path=CSV_PATH,
        video_path=VIDEO_PATH
    )

    if not recorded:
        print("[INFO] Prediction skipped.")
        return

    result = predict_one_gesture(
        CSV_PATH,
        scaler,
        autoencoder,
        dgmm,
        cluster_to_gesture
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"Predicted label : {result['label']}")
    print(f"Cluster ID      : {result['cluster']}")
    print(f"Confidence      : {result['confidence']:.6f}")
    print(f"Log-likelihood  : {result['log_likelihood']:.6f}")
    print("=" * 80)

    show_final_result(result)


if __name__ == "__main__":
    main()
