import cv2
import dlib
import os
import numpy as np

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

def extract_faces_from_video(video_path, output_dir, frame_skip=5):
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processes frames
        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for i, face in enumerate(faces):
                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                # Clip coordinates so they don’t go outside frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                if w > 0 and h > 0:  # ensure valid crop
                    face_img = frame[y:y+h, x:x+w]

                    if face_img.size != 0:  # double check it’s not empty
                        face_filename = os.path.join(output_dir, f"face_{frame_count}_{i}.jpg")
                        success = cv2.imwrite(face_filename, face_img)
                        if success:
                            face_count += 1
                        else:
                            print(f"[WARN] Failed to save {face_filename}")
                else:
                    print(f"[WARN] Invalid face dimensions in {video_path}, frame {frame_count}")

        frame_count += 1

    cap.release()
    print(f"Extracted {face_count} faces from {video_path}")


def preprocess_faces(input_dir, output_dir, size=(224, 224)):
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, size)
        img = img / 255.0  # normalize pixel values to [0,1]
        np.save(os.path.join(output_dir, img_file.split('.')[0] + '.npy'), img)

    print(f"Preprocessed faces saved in {output_dir}")


