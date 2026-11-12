import os, time, json
import cv2
import numpy as np
from collections import deque
import mediapipe as mp

HAND_TASK_PATH = "hand_landmarker.task"
OUT_DIR = "data_126"
SEQ_LEN = 30

labels = [
    "hello",
    "thanks",
    "yes",
    "no",
]
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
    json.dump({str(i): name for i, name in enumerate(labels)}, f, indent=2)

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

def extract_twohand(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int((time.monotonic() - t0) * 1000)
    res = landmarker.detect_for_video(mp_image, ts)

    feat = np.zeros(126, dtype=np.float32)
    if res.hand_landmarks:
        for i, hand_lms in enumerate(res.hand_landmarks):
            name = None
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                name = res.handedness[i][0].category_name

            base = 0 if name == "Left" else 63 if name == "Right" else (0 if i == 0 else 63)

            vals = []
            pts = []
            for lm in hand_lms:
                vals.extend([lm.x, lm.y, lm.z])
                pts.append((int(lm.x * w), int(lm.y * h)))
            feat[base:base+63] = np.array(vals, dtype=np.float32)
    return feat

cap = cv2.VideoCapture(0)
current_class = 0
samples_per_class = 0

print("Keys: number = select class, r = record 30 frames, q = quit")
print("Classes:", {i: n for i, n in enumerate(labels)})

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.putText(frame, f"Class: {current_class} {labels[current_class]}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(frame, "Press r to record (30 frames)",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Collect", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if ord('0') <= k <= ord('9'):
        idx = k - ord('0')
        if idx < len(labels):
            current_class = idx

    if k == ord('r'):
        seq = []
        print(f"Recording class {current_class} ({labels[current_class]})...")
        for _ in range(SEQ_LEN):
            ok, fr = cap.read()
            if not ok:
                break
            feat = extract_twohand(fr)
            seq.append(feat)
            cv2.imshow("Collect", fr)
            cv2.waitKey(1)

        if len(seq) == SEQ_LEN:
            seq = np.stack(seq, axis=0)  # (30,126)
            cls_dir = os.path.join(OUT_DIR, str(current_class))
            os.makedirs(cls_dir, exist_ok=True)
            fname = os.path.join(cls_dir, f"{int(time.time()*1000)}.npy")
            np.save(fname, seq)
            samples_per_class += 1
            print("Saved:", fname)
        else:
            print("Failed to record full sequence.")

cap.release()
cv2.destroyAllWindows()