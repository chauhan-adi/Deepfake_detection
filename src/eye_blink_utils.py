import numpy as np
from scipy.spatial import distance as dist

# Eye aspect ratio (EAR) function
def eye_aspect_ratio(eye):
    # compute distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear
