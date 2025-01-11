import math
from deep_sort.deep_sort import DeepSort
import cv2
import numpy as np
import torch
from ultralytics import YOLO 
import threading
import time
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity


def load_yolo_model(coco_model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(coco_model_path).to(device)
    print(f"Model loaded on: {device}")
    return model

keypoint_conf_threshold = 0.7

show_keypoints = False

def detect_persons(frame, frame_nmr, model, tracker, person_threshold):
    detected_results = {}

    # Run YOLO inference
    results = model(frame, conf=person_threshold, imgsz=1280)

    # Prepare detections for tracker
    bboxes = []  # Bounding boxes for tracking
    confs = []  # Confidence scores
    keypoints = []  # Keypoints for each person

    for result in results:
        for person in range(len(result.boxes)):
            bbox = result.boxes[person]
            conf = bbox.conf.item()  # Confidence score

            if conf < person_threshold:
                continue  # Skip low-confidence detections

            # Keep bbox.xywh[0] as tensor
            bbox_xywh = bbox.xywh[0]  # Center_x, center_y, width, height
            bboxes.append(bbox_xywh)
            confs.append(conf)

            # Extract keypoints directly if available
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                kp = result.keypoints.data[person]  # Keypoints remain as tensors
                keypoints.append(kp)
            else:
                keypoints.append(None)  # No keypoints available

    # Update tracker with detections
    if bboxes:
        tracked_objects = tracker.update(
            torch.stack(bboxes).cpu().numpy(),  # Convert bounding boxes to numpy
            torch.tensor(confs),
            frame  # Pass the original frame as 'ori_img'
        )
    else:
        tracked_objects = []

    # Match tracked objects with detected keypoints and bounding boxes
    for track, bbox, kp in zip(tracked_objects, bboxes, keypoints):
        person_id = int(track[4])  # Get person ID from tracker
        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])  # Convert bbox to corners

        # Add tracked results to the dictionary
        detected_results[person_id] = {
            'frame_nmr': frame_nmr,
            'person': {
                'xywh': bbox,  # Keep xywh as tensor
                'bbox': [x1, y1, x2, y2],  # Top-left and bottom-right coordinates
                'confidence': conf,  # Confidence score
                'keypoints': kp if kp is not None else None,  # Keypoints remain as tensors
            }
        }

    return detected_results




def detect_faces(results, frame, detector, FACE_CONFIDENCE_THRESHOLD):
    for person_id in results:
        if 'person' in results[person_id]:
            person_bbox = results[person_id]['person']['bbox']
            x1, y1, x2, y2 = map(int, person_bbox)
            person_crop = frame[y1:y2, x1:x2]

            # Detect faces within the cropped person bounding box
            try:
                bboxes, kpss = detector.detect(person_crop, 0)  # Detect faces in cropped person region
                if bboxes is not None and len(bboxes) > 0:
                    # Assuming the first detected face is the one we care about
                    bbox, kps = bboxes[0], kpss[0] if kpss is not None else None
                    *face_bbox, conf_score = bbox  # Directly unpack bbox into coordinates and confidence

                    if conf_score >= FACE_CONFIDENCE_THRESHOLD:
                        fx1, fy1, fx2, fy2 = map(int, face_bbox)  # Unpack global coordinates
                        results[person_id]['face'] = {
                            'bbox': [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2],  # Global coordinates
                            'score': conf_score,
                            'keypoints': kps if kps is not None else None  # Optional keypoints
                        }
            except Exception as e:
                print(f"Face detection error for person {person_id}: {str(e)}")

    return results

def extract_face_embeddings(results, frame, recognizer):
    for person_id in results:
        if 'face' in results[person_id] and 'person' in results[person_id]:
            face_info = results[person_id]['face']
            # Get keypoints from the detected face
            kps = face_info.get('keypoints', None)

            # Get person crop from person bounding box
            person_bbox = results[person_id]['person']['bbox']
            x1, y1, x2, y2 = map(int, person_bbox)
            person_crop = frame[y1:y2, x1:x2]

            try:
                # Extract embedding using the recognizer and person crop
                embedding = recognizer(person_crop, kps)

                # Add embedding to face information
                results[person_id]['face']['embedding'] = embedding
                # print(f"Saved embedding for person {person_id}")
            except Exception as e:
                print(f"Error extracting embedding for person {person_id}: {str(e)}")
                results[person_id]['face']['embedding'] = None
    
    return results

# Calculate cosine similarity
def calculate_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


def find_closest_match(face_embedding, user_data_dict, threshold=0.2):
    max_similarity = -1  # Cosine similarity is higher for better matches
    closest_name = None

    for person_id, data in user_data_dict.items():
        stored_embedding = data['face_embedding']

        # Calculate similarity and check if it is the closest match
        similarity = calculate_similarity(face_embedding, stored_embedding)

        # Debugging: Print the calculated similarity
        print(f"Cosine Similarity to {person_id}: {similarity}")

        if similarity > max_similarity and similarity > threshold:
            max_similarity = similarity
            closest_name = person_id

    return closest_name, max_similarity

person_name_map = {}
name_similarity_map = {}

def match_embeddings_with_users(results_with_embeddings, user_data_dict):
    global person_name_map
    global name_similarity_map
   
    # Iterate through the tracked persons
    for person_id in results_with_embeddings:
        if 'face' in results_with_embeddings[person_id]:
            face_embedding = results_with_embeddings[person_id]['face'].get('embedding')

            # If an embedding exists, try to find a match
            if face_embedding is not None:
                # Check if the person_id has already been matched
                if person_id not in person_name_map.values():
                    # Find the closest match from the user data
                    matched_name, similarity = find_closest_match(face_embedding, user_data_dict)

                    # If a match is found, store it in the person_name_map
                    if matched_name:
                        if matched_name in name_similarity_map and name_similarity_map[matched_name] > similarity:
                            continue

                        person_name_map[matched_name] = person_id
                        name_similarity_map[matched_name] = similarity

    # If person_id exists in person_name_map, use the stored name
    for person_name, person_id in person_name_map.items():
        if person_id in results_with_embeddings:
            results_with_embeddings[person_id]['name'] = person_name

def visualize_processed_frame(frame, tracked_person_data):

    for person_id, person_data in tracked_person_data.items():

        # Visualize person bounding boxes
        x1, y1, x2, y2 = map(int, person_data['person']['bbox'])
        keypoints = person_data['person']['keypoints']
 
        # Draw the person bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # Retrieve the name if assigned, otherwise show the ID
        name = person_data.get('name', f'ID: {person_id}')

        # Check if there is additional overlay information
        if "overlay" in person_data:
            overlay_data = person_data["overlay"]
            overlay_text = f"{name}\n{overlay_data['status']}\n" \
                           f"Student ID: {overlay_data['student_id']}\n" \
                           f"Course: {overlay_data['course']}\n" \
                           f"Year: {overlay_data['year']}\n" \
                           f"Disciplinary Action: {overlay_data['action']}\n" \
                           f"Expulsion Date: {overlay_data['expulsion_date']}"

            # Define overlay box dimensions
            overlay_x1 = max(0, x1 - 5)
            overlay_y1 = max(0, y1 - 150)
            overlay_x2 = min(frame.shape[1], x1 + 300)
            overlay_y2 = y1 - 5

            # Draw translucent rectangle for overlay
            overlay_color = (0, 0, 255)  # Red overlay
            overlay_alpha = 0.6
            overlay_box = frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2].copy()
            cv2.rectangle(frame, (overlay_x1, overlay_y1), (overlay_x2, overlay_y2), overlay_color, -1)
            cv2.addWeighted(frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2], overlay_alpha,
                            overlay_box, 1 - overlay_alpha, 0, frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2])

            # Draw overlay text
            font_scale = 0.5
            font_thickness = 1
            text_y = overlay_y1 + 20
            for line in overlay_text.split("\n"):
                cv2.putText(frame, line, (overlay_x1 + 10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                text_y += 20

        # Display the name and status label
        label = f'{name}'
        font_scale = 0.9  # Increased font scale for larger text
        font_thickness = 3  # Adjust thickness for better visibility
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        # Draw the white background rectangle with increased padding
        padding = 5  # Adjust padding as needed
        cv2.rectangle(frame, (label_x - padding, label_y - label_size[1] - padding),
                      (label_x + label_size[0] + padding, label_y + padding), (255, 255, 255), -1)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

        if show_keypoints:
            for kp in keypoints:
                x, y, confidence = int(kp[0]), int(kp[1]), kp[2]
                if confidence > keypoint_conf_threshold:
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            head, left_shoulder, right_shoulder = keypoints[0], keypoints[5], keypoints[6]
            left_hip, right_hip = keypoints[11], keypoints[12]
            if left_shoulder[2] > keypoint_conf_threshold and right_shoulder[2] > keypoint_conf_threshold:
                cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])),
                         (int(right_shoulder[0]), int(right_shoulder[1])), (255, 255, 0), 2)
            if left_hip[2] > keypoint_conf_threshold and right_hip[2] > keypoint_conf_threshold:
                cv2.line(frame, (int(left_hip[0]), int(left_hip[1])),
                         (int(right_hip[0]), int(right_hip[1])), (255, 255, 0), 2)
            if head[2] > keypoint_conf_threshold and left_shoulder[2] > keypoint_conf_threshold and right_shoulder[
                2] > keypoint_conf_threshold:
                shoulder_center = (
                    (left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
                cv2.line(frame, (int(head[0]), int(head[1])), (int(shoulder_center[0]), int(shoulder_center[1])),
                         (255, 255, 0), 2)

        # Visualize face bounding boxes (if detected for the person)
        if 'face' in person_data:
            face_x1, face_y1, face_x2, face_y2 = map(int, person_data['face']['bbox'])
            # face_score = person_data['face']['score']

            # Draw the face bounding box in blue
            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)

    return frame