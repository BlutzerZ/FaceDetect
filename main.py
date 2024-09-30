import cv2
import torch
import pickle
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import numpy as np
from PIL import Image

# Load YOLOv8-face model and move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8-face.pt').to(device)  # YOLOv8 will automatically use GPU if available

# Load FaceNet model and move to GPU
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the saved embeddings from the .pkl file
with open('embeddings.pkl', 'rb') as f:
    embeddings_db = pickle.load(f)

# Function to crop face using YOLOv8-face
def crop_face(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    face_crop = image[y1:y2, x1:x2]
    return face_crop

# Function to extract face embedding using FaceNet and GPU
def extract_embedding(face_image):
    img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    img = img.resize((160, 160))  # FaceNet input size requirement
    img = np.array(img).astype(np.float32)
    img = (img - 127.5) / 128.0  # Normalize like FaceNet expects

    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)  # Convert to tensor and move to GPU

    with torch.no_grad():
        embedding = facenet_model(img_tensor).cpu().numpy().flatten()  # Transfer to CPU for further processing

    return embedding

# Function to match the face embedding with the database
def match_face(embedding, threshold=0.6):
    min_distance = float('inf')
    best_match = "Unknown"
    for person_name, saved_embedding in embeddings_db.items():
        distance = cosine(embedding, saved_embedding)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            best_match = person_name

    return best_match

# Open webcam using OpenCV
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame and move image to GPU
    results = model(frame)

    # Process each face detected
    for result in results:
        for box in result.boxes.xyxy:
            # Crop face based on bounding box
            cropped_face = crop_face(frame, box)

            # Extract embedding for the cropped face
            embedding = extract_embedding(cropped_face)

            # Match the embedding with the saved database
            name = match_face(embedding)

            # Draw a rectangle around the face and put the name
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result in real-time
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
