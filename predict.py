# predict_facenet.py
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# === LOAD MODEL ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=False, device=device)

# === LOAD EMBEDDINGS ===
with open("face_embeddings.pickle", "rb") as f:
    known_embeddings = pickle.load(f)

names = list(known_embeddings.keys())
embeddings = np.array([known_embeddings[name] for name in names])

# === CAPTURE ===
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)

    if face is not None:
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()

        # === COSINE SIMILARITY ===
        sims = cosine_similarity(embedding, embeddings)[0]
        max_sim_idx = np.argmax(sims)
        max_sim = sims[max_sim_idx]

        if max_sim >= 0.75:  # You can raise this to 0.85+ for stricter matching
            identity = names[max_sim_idx]
            label = f"{identity} ({max_sim:.2f})"
        else:
            label = "Unknown"

        # Draw box and label
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            (x1, y1, x2, y2) = map(int, boxes[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition - FaceNet", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
