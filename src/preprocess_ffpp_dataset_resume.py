
import os
from video_preprocess import extract_faces_from_video, preprocess_faces

# Paths
dataset_dir = "Dataset/face++dataset"
output_faces_dir = "data/processed/faces"
output_numpy_dir = "data/processed/numpy_faces"

for label in ['ffpp_real', 'ffpp_fake']:
    input_folder = os.path.join(dataset_dir, label)
    output_folder_faces = os.path.join(output_faces_dir, 'real' if 'real' in label else 'fake')
    output_folder_numpy = os.path.join(output_numpy_dir, 'real' if 'real' in label else 'fake')

    if not os.path.exists(output_folder_faces):
        os.makedirs(output_folder_faces)
    if not os.path.exists(output_folder_numpy):
        os.makedirs(output_folder_numpy)

    for video_file in os.listdir(input_folder):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(input_folder, video_file)
        temp_face_dir = os.path.join(output_folder_faces, video_file.split('.')[0])

        # Checks for already processed frames
        if os.path.exists(temp_face_dir) and len(os.listdir(temp_face_dir)) > 0:
            print(f"Skipping {video_file}, already processed.")
            continue

        print(f"Processing video: {video_path}")

        # Step 1: Extract faces
        extract_faces_from_video(video_path, temp_face_dir, frame_skip=5)

        # Step 2: Preprocess faces and save as numpy arrays
        preprocess_faces(temp_face_dir, output_folder_numpy)

print("FaceForensics++ dataset preprocessing with dlib (resume fixed) completed!")
