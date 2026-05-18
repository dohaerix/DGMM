import os
import glob
import numpy as np
import pandas as pd

# ============================================================
# 1. FRAME-LEVEL NORMALIZATION (SHAPE FEATURES ONLY)
# ============================================================

def preprocess_frame(frame_landmarks, eps=1e-6):
    """
    Normalize a single frame of hand landmarks.

    Operations:
    - Detect hand presence
    - Wrist-centered translation normalization
    - Scale normalization using palm width (index MCP to pinky MCP)

    Parameters
    ----------
    frame_landmarks : np.ndarray
        Shape (42, 2), raw x-y landmarks for one frame

    eps : float
        Small constant to prevent division by zero

    Returns
    -------
    normalized : np.ndarray
        Shape (42, 2), wrist- and scale-normalized landmarks
        Absent hands remain zeros.
    """

    assert frame_landmarks.shape == (42, 2)
    normalized = np.zeros_like(frame_landmarks)

    # Hand definitions
    hands = {
        0: {"indices": np.arange(0, 21), "wrist": 0, "index_mcp": 5, "pinky_mcp": 17},
        1: {"indices": np.arange(21, 42), "wrist": 21, "index_mcp": 26, "pinky_mcp": 38},
    }

    for hand_id, hand in hands.items():
        idx = hand["indices"]
        hand_landmarks = frame_landmarks[idx]

        # If hand is absent → skip
        if np.allclose(hand_landmarks, 0.0):
            continue

        # ---- Translation normalization (wrist centered) ----
        wrist = frame_landmarks[hand["wrist"]]
        hand_landmarks = hand_landmarks - wrist

        # ---- Scale normalization (palm width) ----
        p1 = frame_landmarks[hand["index_mcp"]] - wrist
        p2 = frame_landmarks[hand["pinky_mcp"]] - wrist
        palm_width = np.linalg.norm(p1 - p2)

        if palm_width < eps:
            continue

        hand_landmarks = hand_landmarks / (palm_width + eps)
        normalized[idx] = hand_landmarks

    return normalized


# ============================================================
# 2. VIDEO PREPROCESSING
# ============================================================

def preprocess_video(csv_data, num_frames=150):
    """
    Process entire video CSV.

    Returns:
    - Normalized landmarks (shape features)
    - Raw wrist positions (global trajectory)

    Parameters
    ----------
    csv_data : np.ndarray
        Shape (num_frames*42, >=2)

    num_frames : int

    Returns
    -------
    normalized_video : np.ndarray
        (T, 42, 2)

    raw_wrist_positions : np.ndarray
        (T, 2, 2)
        raw_wrist_positions[t, hand_id] = [x, y]
    """

    assert csv_data.shape[0] == num_frames * 42

    # Reshape to (T, 42, 3) and drop Z
    video = csv_data.reshape(num_frames, 42, 3)[..., :2]

    normalized_video = np.zeros_like(video)
    raw_wrist_positions = np.zeros((num_frames, 2, 2))

    for t in range(num_frames):
        frame = video[t]

        # Store raw wrist positions BEFORE normalization
        raw_wrist_positions[t, 0] = frame[0]      # Hand 1 wrist
        raw_wrist_positions[t, 1] = frame[21]     # Hand 2 wrist

        normalized_video[t] = preprocess_frame(frame)

    return normalized_video, raw_wrist_positions


# ============================================================
# 3. GEOMETRIC UTILITIES
# ============================================================

def compute_angle(a, b, c, eps=1e-6):
    """
    Compute angle (in radians) at point b formed by a-b-c.
    """

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba) + eps
    norm_bc = np.linalg.norm(bc) + eps

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)

# ============================================================
# 4. FRAME-LEVEL FEATURE EXTRACTION (SHAPE ONLY)
# ============================================================

def extract_frame_features(frame_landmarks):
    """
    Extract static geometric features from normalized landmarks.

    Features per hand:
    - 5 joint angles
    - 5 fingertip-to-wrist distances
    - 1 hand spread measure

    Total:
    11 features per hand × 2 hands = 22 features

    Returns
    -------
    features : np.ndarray
        Shape (22,)
    """

    features = []

    fingers = [
        (1, 2, 4),    # Thumb
        (5, 6, 8),    # Index
        (9, 10, 12),  # Middle
        (13, 14, 16), # Ring
        (17, 18, 20)  # Pinky
    ]

    fingertip_indices = [4, 8, 12, 16, 20]

    for hand_id in range(2):
        offset = 0 if hand_id == 0 else 21
        hand = frame_landmarks[offset:offset + 21]

        if np.allclose(hand, 0.0):
            features.extend([0.0] * 11)
            continue

        wrist = hand[0]

        # ---- Joint angles ----
        for (a, b, c) in fingers:
            angle = compute_angle(hand[a], hand[b], hand[c])
            features.append(angle)

        # ---- Fingertip distances ----
        fingertip_points = []

        for idx in fingertip_indices:
            dist = np.linalg.norm(hand[idx] - wrist)
            features.append(dist)
            fingertip_points.append(hand[idx])

        # ---- Spread (mean fingertip distance) ----
        fingertip_points = np.array(fingertip_points)
        dists = []

        for i in range(len(fingertip_points)):
            for j in range(i + 1, len(fingertip_points)):
                dists.append(np.linalg.norm(fingertip_points[i] - fingertip_points[j]))

        features.append(np.mean(dists))

    return np.array(features, dtype=np.float32)


# ============================================================
# 5. GLOBAL MOTION FEATURES
# ============================================================

def compute_wrist_velocity(wrist_positions, fps=30):
    """
    Compute wrist velocity using discrete derivative.

    Parameters
    ----------
    wrist_positions : np.ndarray
        Shape (T, 2, 2)

    fps : int

    Returns
    -------
    velocity : np.ndarray
        Shape (T, 2, 2)
    """

    T = wrist_positions.shape[0]
    velocity = np.zeros_like(wrist_positions)
    dt = 1.0 / fps      # Time is inverse of frame rate!

    for t in range(1, T):
        velocity[t] = (wrist_positions[t] - wrist_positions[t - 1]) / dt

    return velocity


# ============================================================
# 6. COMBINE STATIC + DYNAMIC FEATURES
# ============================================================

def build_video_feature_tensor(normalized_video, raw_wrist_positions, fps=30):
    """
    Construct final feature tensor per video.

    Per frame features:
    - 22 static shape features
    - 4 wrist positions (2 hands × x,y)
    - 4 wrist velocities
    - 1 inter-hand distance

    Total per frame = 33 features
    """

    T = normalized_video.shape[0]
    frame_features = []

    # Compute wrist velocity
    wrist_velocity = compute_wrist_velocity(raw_wrist_positions, fps)

    for t in range(T):
        # <---- Static shape features ---->
        static_feat = extract_frame_features(normalized_video[t])

        # <---- Wrist position ---->
        wrist_pos_hand1 = raw_wrist_positions[t, 0]
        wrist_pos_hand2 = raw_wrist_positions[t, 1]
        
        # <---- Wrist velocity ---->
        wrist_vel_hand1 = wrist_velocity[t, 0]
        wrist_vel_hand2 = wrist_velocity[t, 1]

        # <---- Inter-hand distance (RAW, not normalized) ---->
        if not np.allclose(raw_wrist_positions[t, 0], 0.0) and not np.allclose(raw_wrist_positions[t, 1], 0.0):
            inter_hand_dist = np.linalg.norm(raw_wrist_positions[t, 0] - raw_wrist_positions[t, 1])
        else:
            inter_hand_dist = 0.0

        combined = np.concatenate([static_feat, wrist_pos_hand1, wrist_pos_hand2, wrist_vel_hand1, wrist_vel_hand2, [inter_hand_dist]])
        frame_features.append(combined)

    return np.array(frame_features, dtype=np.float32)


# ============================================================
# 7. BUILD DATASET FROM FOLDERS
# ============================================================

def build_feature_tensor_from_folders(root_dir, num_frames=150, fps=30):
    """
    Build dataset tensor from gesture-organized folders.

    Returns
    -------
    X : np.ndarray
        Shape (N, T, 31)

    y : np.ndarray
        Labels

    video_paths : list
        File paths

    label_map : dict
        Mapping label index → gesture name
    """

    gesture_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    X, y, video_paths = [], [], []
    label_map = {i: g for i, g in enumerate(gesture_folders)}

    for label, gesture_name in label_map.items():
        gesture_dir = os.path.join(root_dir, gesture_name)
        csv_files = sorted(glob.glob(os.path.join(gesture_dir, "*.csv")))

        for csv_path in csv_files:
            csv_data = pd.read_csv(csv_path, skiprows=1, header=None).values

            normalized_video, raw_wrist_positions = preprocess_video(csv_data, num_frames)

            video_features = build_video_feature_tensor(normalized_video, raw_wrist_positions, fps)

            X.append(video_features)
            y.append(label)
            video_paths.append(csv_path)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    return X, y, video_paths, label_map

# ============================================================
# 8. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    feature_tensor, labels, files, label_map = build_feature_tensor_from_folders("HGRM_dataset\csv")
    np.save("features.npy", feature_tensor)
    np.save("labels.npy", labels)
    np.save("mappings.npy", label_map, allow_pickle=True)

    print("Feature tensor shape:", feature_tensor.shape)
    print("Label shape:", labels.shape)
    print("Label Map:", label_map)
