from ultralytics import YOLO
import torch
import cv2
import random
import numpy as np
import csv
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'Models')))

from deep_sort.deep_sort import DeepSort
from utils import visualize_processed_frame, detect_persons, load_yolo_model, detect_faces, extract_face_embeddings, match_embeddings_with_users
from models import SCRFD, ArcFace
import ast

# Set the model paths
coco_model_path = './Models/yolo11m-pose.pt'
coco_model = load_yolo_model(coco_model_path)

# Configuration
DETECTION_WEIGHT_PATH = "./Models/det_10g.onnx"
RECOGNITION_WEIGHT_PATH = "./Models/w600k_r50.onnx"

SIMILARITY_THRESHOLD = 0.2  # Face match score
FACE_CONFIDENCE_THRESHOLD = 0.5  # Face confidence score

EMBEDDINGS_CSV = "user_data.csv"  # CSV file containing embeddings

def load_user_data(csv_file_path):
    """ Load the user data from CSV and store embeddings and bounding boxes as NumPy arrays in a dictionary """
    user_data_dict = {}
    user_data = pd.read_csv(csv_file_path)

    for _, row in user_data.iterrows():
        # Parse the face_embedding column
        face_embedding_str = row['face_embedding']
        if pd.isna(face_embedding_str) or face_embedding_str == 'None':
            continue

        face_embedding = np.array([float(x) for x in face_embedding_str.split(',')])

        # Parse the face_bbox column
        face_bbox_str = row['face_bbox']
        face_bbox = np.array(ast.literal_eval(face_bbox_str))

        # Store data in a dictionary with person_id as the key
        user_data_dict[row['person_id']] = {
            'face_embedding': face_embedding,
            'face_bbox': face_bbox,
            'face_score': row['face_score']
        }

    return user_data_dict


# Load user data as a dictionary at the start
user_data_dict = load_user_data(EMBEDDINGS_CSV)

detector = SCRFD(DETECTION_WEIGHT_PATH, input_size=(640, 640), conf_thres=FACE_CONFIDENCE_THRESHOLD)

# Load DeepSort tracker
deep_sort_weights = 'Models/deep_sort/deep/checkpoint/ckpt.t7'
mot_tracker = DeepSort(
    model_path=deep_sort_weights,
    max_age=35,  
    use_cuda=True
)

# Set the person bounding box threshold and similarity threshold
person_threshold = 0.55

# Open the live video stream (0 for default webcam or replace with CCTV feed URL)
cap = cv2.VideoCapture("Face1.mp4")  

fps = cap.get(cv2.CAP_PROP_FPS)
skip_frames = int(fps * 0)  # N seconds of video
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)  # Skip to N-second mark

frame_number = skip_frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or corrupted frame encountered.")
        break

    try:

        # Detect people in the frame
        results = detect_persons(frame, frame_number, coco_model, mot_tracker, person_threshold)
        
        # Detect faces of each person
        detect_faces(results, frame, detector, FACE_CONFIDENCE_THRESHOLD)

        # Extract face embeddings of each person
        extract_face_embeddings(results, frame, recognizer)

        # Match embeddings with frames
        match_embeddings_with_users(results, user_data_dict)

        # Visualize the frame with bounding boxes and IDs
        processed_frame = visualize_processed_frame(frame, results)

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error processing frame: {e}")
        continue  # Skip to the next frame

    # Increment the frame number
    frame_number += 1

cap.release()
cv2.destroyAllWindows()
