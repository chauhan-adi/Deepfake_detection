import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

def extract_faces_from_video_mtcnn(video_path, output_dir, frame_skip=5):
  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_frame)

            if boxes is not None:
                h_frame, w_frame, _ = frame.shape

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(b) for b in box]

                    # Clip coordinates to frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w_frame, x2)
                    y2 = min(h_frame, y2)

                    # Only save valid crops
                    if x2 > x1 and y2 > y1:
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size != 0:  # double check
                            face_filename = os.path.join(output_dir, f"face_{frame_count}_{i}.jpg")
                            cv2.imwrite(face_filename, face_img)
                            face_count += 1
                    else:
                        print(f"[WARN] Skipping invalid face at frame {frame_count}")

        frame_count += 1

    cap.release()
    print(f"Extracted {face_count} faces from {video_path}")



def preprocess_faces_mtcnn(input_dir, output_dir, size=(224, 224)):
    """
    Resize and normalize extracted faces, save as numpy arrays.

    Parameters:
        input_dir: directory with extracted face images
        output_dir: directory to save .npy arrays
        size: target image size
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, size)
        img = img / 255.0  # normalize
        np.save(os.path.join(output_dir, img_file.split('.')[0] + '.npy'), img)

    print(f"Preprocessed faces saved in {output_dir}")


