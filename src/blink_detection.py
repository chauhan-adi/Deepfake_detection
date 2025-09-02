import cv2
import dlib
import numpy as np
from imutils import face_utils
from eye_blink_utils import eye_aspect_ratio

# constants
EAR_THRESHOLD = 0.25   # if below → eye closed
EAR_CONSEC_FRAMES = 3  # min frames to count as blink

def extract_blink_features(video_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # get landmark indices for left/right eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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

    # features to save
    features = {
        "blink_count": blink_count,
        "avg_ear": float(np.mean(ear_list)) if ear_list else 0,
        "var_ear": float(np.var(ear_list)) if ear_list else 0
    }
    return features


