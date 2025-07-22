import cv2
import dlib
import time
import os

# === CONFIG ===
SAVE_THRESHOLD = 0.40
FACE_PADDING = 30
min_face_size = (120, 120)
MAX_IMAGES = 200

# === SELECT NAME FROM 'faces/' ===
faces_dir = "faces"
os.makedirs(faces_dir, exist_ok=True)
users = sorted(os.listdir(faces_dir))
print("Available Users:")
for i, user in enumerate(users):
    print(f"[{i}] {user}")
print("[n] New user")

choice = input("Enter index or name: ").strip()
if choice.isdigit() and int(choice) < len(users):
    name = users[int(choice)]
elif choice.lower() == "n" or choice.strip() == "":
    name = input("Enter new name: ").strip()
    if not name:
        print("Name required.")
        exit()
else:
    name = choice

base_dir = os.path.join(faces_dir, name)
parts = ['face', 'left_eye', 'right_eye', 'nose', 'mouth']
for part in parts:
    os.makedirs(os.path.join(base_dir, part), exist_ok=True)

# === LOAD MODELS ===
cascade_paths = [
    "models/haarcascade_frontalface_default.xml",
    "models/haarcascade_frontalface_alt.xml",
    "models/haarcascade_profileface.xml"
]
detectors = [cv2.CascadeClassifier(p) for p in cascade_paths if not cv2.CascadeClassifier(p).empty()]

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    print("Missing shape predictor. Please place it in models folder.")
    exit()

dlib_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

# === INIT CAMERA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not access webcam.")
    exit()

prev_face = None
stable_start_time = None
img_count = 0

print(f"[INFO] Starting capture for {name}. Press 'q' to quit.")

while True:
    if img_count >= MAX_IMAGES:
        print(f"[✔] Finished capturing {MAX_IMAGES} images for {name}")
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = None

    for detector in detectors:
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_face_size)
        if len(faces) > 0:
            face = faces[0]
            break

    if face is not None:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = shape_predictor(gray, rect)

        for i in range(68):
            (px, py) = (shape.part(i).x, shape.part(i).y)
            cv2.circle(frame, (px, py), 1, (255, 0, 0), -1)

        if prev_face is not None:
            dx = abs(x - prev_face[0])
            dy = abs(y - prev_face[1])
            if dx < 20 and dy < 20:
                if stable_start_time is None:
                    stable_start_time = time.time()
                elif time.time() - stable_start_time >= SAVE_THRESHOLD:
                    # === SAVE FACE ===
                    px1 = max(0, x - FACE_PADDING)
                    py1 = max(0, y - FACE_PADDING)
                    px2 = min(frame.shape[1], x + w + FACE_PADDING)
                    py2 = min(frame.shape[0], y + h + FACE_PADDING)

                    face_crop = frame[py1:py2, px1:px2]
                    if face_crop.size > 0:
                        face_path = os.path.join(base_dir, 'face', f"{name}_{img_count}.jpg")
                        cv2.imwrite(face_path, face_crop)

                    # === SAVE FEATURES ===
                    def save_region(indexes, part_name):
                        pts = [(shape.part(i).x, shape.part(i).y) for i in indexes]
                        xs, ys = zip(*pts)
                        rx1, ry1 = max(min(xs) - 5, 0), max(min(ys) - 5, 0)
                        rx2, ry2 = min(max(xs) + 5, frame.shape[1]), min(max(ys) + 5, frame.shape[0])
                        roi = frame[ry1:ry2, rx1:rx2]
                        if roi.size > 0:
                            save_path = os.path.join(base_dir, part_name, f"{name}_{img_count}.jpg")
                            cv2.imwrite(save_path, roi)

                    save_region(range(36, 42), 'left_eye')
                    save_region(range(42, 48), 'right_eye')
                    save_region(range(27, 36), 'nose')
                    save_region(range(48, 68), 'mouth')

                    print(f"[✔] Saved image {img_count + 1}/{MAX_IMAGES}")
                    img_count += 1
                    stable_start_time = None
            else:
                stable_start_time = None
        else:
            stable_start_time = time.time()

        prev_face = (x, y)
    else:
        prev_face = None
        stable_start_time = None

    cv2.imshow("Capture Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
