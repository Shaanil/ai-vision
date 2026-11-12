import cv2
import json
import time
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp

MODEL_PATH = "ssl_model.keras"
LABELS_PATH = "labels.json"
HAND_TASK_PATH = "hand_landmarker.task"  # download this file

# ---- Load model + labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

print("Model input shape:", model.input_shape)
EXPECTED_DIM = model.input_shape[-1]   # should be 126 for two hands
SEQ_LEN = model.input_shape[1]         # usually 30

if EXPECTED_DIM != 126:
    raise ValueError(
        f"Your model expects {EXPECTED_DIM} features, but two-hands needs 126.\n"
        f"Retrain your model with 126-dim hand landmarks."
    )

sequence = deque(maxlen=SEQ_LEN)

# Smoothing / stability
pred_history = deque(maxlen=12)
CONF_THRESH = 0.70
HOLD_FRAMES = 7

# ---- MediaPipe Tasks: HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_TASK_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
landmarker = HandLandmarker.create_from_options(options)
t0 = time.monotonic()

def extract_twohand_features(frame):
    """
    Returns:
      feat: (126,) float32 = [LeftHand(63), RightHand(63)] padded with zeros if missing
      pts:  list of hands, each is list of (x_px, y_px) for drawing
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.monotonic() - t0) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    feat = np.zeros(126, dtype=np.float32)
    draw_points = []

    if result.hand_landmarks:
        for i, hand_lms in enumerate(result.hand_landmarks):
            # handedness: 'Left' or 'Right' if available
            hand_name = None
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                hand_name = result.handedness[i][0].category_name  # e.g., 'Left'/'Right'

            # choose slot: Left->0..62, Right->63..125
            if hand_name == "Left":
                base = 0
            elif hand_name == "Right":
                base = 63
            else:
                # fallback: first goes Left slot, second goes Right slot
                base = 0 if i == 0 else 63

            vals = []
            pts = []
            for lm in hand_lms:  # 21 landmarks
                vals.extend([lm.x, lm.y, lm.z])
                pts.append((int(lm.x * w), int(lm.y * h)))

            vals = np.array(vals, dtype=np.float32)  # (63,)
            feat[base:base+63] = vals
            draw_points.append(pts)

    return feat, draw_points

def stable_label_from_history(hist):
    if not hist:
        return None
    vals, counts = np.unique(np.array(hist), return_counts=True)
    best = int(vals[np.argmax(counts)])
    best_count = int(np.max(counts))
    return best, best_count

cap = cv2.VideoCapture(0)

display_label = ""
display_score = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feat, hands_pts = extract_twohand_features(frame)
    sequence.append(feat)

    # draw points (simple, no connections)
    for pts in hands_pts:
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Predict once we have 30 frames
    if len(sequence) == SEQ_LEN:
        x = np.array(sequence, dtype=np.float32)    # (30, 126)
        x = np.expand_dims(x, axis=0)               # (1, 30, 126)

        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        score = float(probs[idx])

        if score >= CONF_THRESH:
            pred_history.append(idx)

        stable = stable_label_from_history(pred_history)
        if stable is not None:
            stable_idx, stable_count = stable
            if stable_count >= HOLD_FRAMES:
                display_label = label_map.get(str(stable_idx), str(stable_idx))
                display_score = score
        else:
            display_label = ""
            display_score = 0.0

    text = f"{display_label} ({display_score:.2f})" if display_label else "..."
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.imshow("Two-Hand Sign Guess", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()