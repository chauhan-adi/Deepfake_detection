import os
import numpy as np
import dlib
import cv2
from imutils import face_utils
from eye_blink_utils import eye_aspect_ratio

# ===== Constants =====
EAR_THRESHOLD = 0.25       # below this → eye closed
EAR_CONSEC_FRAMES = 3      # min consecutive frames to count a blink

# ===== Paths =====
DATASET_DIR = "Dataset/face++dataset"  # your FF++ root
OUTPUT_BLINK_FILE = "data/processed/blink_features.npy"

# ===== Landmark predictor =====
predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(predictor_path):
    raise FileNotFoundError(
        f"Download shape_predictor_68_face_landmarks.dat from "
        f"http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and place it here: {predictor_path}"
    )

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ===== Load previous results if exist =====
if os.path.exists(OUTPUT_BLINK_FILE):
    blink_features = np.load(OUTPUT_BLINK_FILE, allow_pickle=True).item()
    print(f"Loaded existing blink features for {len(blink_features)} videos")
else:
    blink_features = {}

# ===== Process a single video =====
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    blink_count = 0
    frame_counter = 0
    ear_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            ear_list.append(ear)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= EAR_CONSEC_FRAMES:
                    blink_count += 1
                frame_counter = 0

    cap.release()

    avg_ear = float(np.mean(ear_list)) if ear_list else 0
    var_ear = float(np.var(ear_list)) if ear_list else 0

    return [blink_count, avg_ear, var_ear, 0.0]  # last value reserved for future features

# ===== Loop over FF++ folders =====
for label in ['ffpp_real', 'ffpp_fake']:
    input_folder = os.path.join(DATASET_DIR, label)
    for video_file in os.listdir(input_folder):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        print(f"Processing video: {video_file}")

        # Skip if already processed
        if video_name in blink_features:
            print(f"Skipped (already processed): {video_name}")
            continue

        features = process_video(video_path)
        blink_features[video_name] = features

        # Save intermediate results in case of interruption
        os.makedirs(os.path.dirname(OUTPUT_BLINK_FILE), exist_ok=True)
        np.save(OUTPUT_BLINK_FILE, blink_features)

# ===== Save all results =====
np.save(OUTPUT_BLINK_FILE, blink_features)
print(f"Blink features saved to {OUTPUT_BLINK_FILE} ({len(blink_features)} videos processed)")
