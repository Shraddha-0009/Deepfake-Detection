import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from efficientnet_pytorch import EfficientNet
import mediapipe as mp
import os

# ------------------------------- Config -------------------------------
MODEL_DIR = r"D:\hi\backend\models"
SEQ_LEN = 16
IMG_SIZE = 380
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------- Model Definition -------------------------------
class CNN_LSTM_Model(nn.Module):
    def __init__(self, lstm_hidden=256, num_classes=2, cnn_trainable=False):
        super().__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b4')
        self.cnn._fc = nn.Identity()
        for param in self.cnn.parameters():
            param.requires_grad = cnn_trainable

        self.lstm = nn.LSTM(input_size=1792, hidden_size=lstm_hidden,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        return self.fc2(x)

# ------------------------------- Load Models -------------------------------
def load_all_models(model_dir=MODEL_DIR):
    model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
                   if f.endswith(".pth") or f.endswith(".pt")]
    models = {}
    for path in model_paths:
        try:
            model = CNN_LSTM_Model().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models[path] = model
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")
    return models

models = load_all_models()

# ------------------------------- Preprocessing -------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------- MediaPipe Setup -------------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

# ------------------------------- Frame Extraction -------------------------------
def extract_frames(video_path, seq_len=SEQ_LEN):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // seq_len)

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) < seq_len:
        frames = (frames * ((seq_len + len(frames) - 1) // len(frames)))[:seq_len]
    elif len(frames) > seq_len:
        frames = frames[:seq_len]

    return frames

# ------------------------------- Feature Computation -------------------------------
def compute_features(frames, is_video=True):
    facial_movements, eye_blinks, lighting_vals = [], [], []
    prev_landmarks = None
    prev_gray = None
    left_eye_idx = [33, 160, 158, 133, 153, 144]
    right_eye_idx = [362, 385, 387, 263, 373, 380]

    def eye_aspect_ratio(pts):
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C)

    for frame in frames:
        frame_small = cv2.resize(frame, (480, 480))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)
        results = face_mesh.process(frame_small)

        # Lighting
        if prev_gray is not None:
            lighting_change = np.mean(np.abs(gray - prev_gray)) / 255.0
            lighting_vals.append(lighting_change)
        prev_gray = gray

        movement = 0.0
        ear_avg = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            lm = np.array([[p.x * frame_small.shape[1], p.y * frame_small.shape[0]] for p in landmarks])
            if prev_landmarks is not None:
                movement = np.mean(np.linalg.norm(lm - prev_landmarks, axis=1))
            facial_movements.append(movement)
            prev_landmarks = lm

            pts_left = lm[left_eye_idx]
            pts_right = lm[right_eye_idx]
            ear_left = eye_aspect_ratio(pts_left)
            ear_right = eye_aspect_ratio(pts_right)
            ear_avg = (ear_left + ear_right) / 2.0
            eye_blinks.append(ear_avg)
        else:
            facial_movements.append(0.0)
            eye_blinks.append(0.0)

    features = {}
    if facial_movements:
        features['facial_movement'] = float(np.clip(np.mean(facial_movements) / 20.0, 0, 1))
    if eye_blinks:
        features['eye_blink'] = float(np.clip(1 - np.mean(eye_blinks), 0, 1))
    if lighting_vals:
        features['lighting'] = float(np.clip(np.mean(lighting_vals), 0, 1))

    return features

# ------------------------------- Human-readable Explanation -------------------------------
def explain_features(prediction, features):
    explanation_parts = []

    if "eye_blink" in features:
        if prediction == "FAKE":
            if features["eye_blink"] < 0.3:
                explanation_parts.append("low eye blinks")
            else:
                explanation_parts.append("eye blink pattern unusual")
        else:
            explanation_parts.append("normal eye blinks")

    if "facial_movement" in features:
        if prediction == "FAKE":
            if features["facial_movement"] < 0.3:
                explanation_parts.append("low facial movement")
            else:
                explanation_parts.append("facial movement unusual")
        else:
            explanation_parts.append("normal facial movement")

    if "lighting" in features:
        if prediction == "FAKE":
            if features["lighting"] < 0.3:
                explanation_parts.append("dark or inconsistent lighting")
            else:
                explanation_parts.append("lighting variations unusual")
        else:
            explanation_parts.append("normal lighting")

    explanation = f"Video predicted as {prediction} due to " + ", ".join(explanation_parts) + "."
    return explanation

# ------------------------------- Model Selection -------------------------------
def select_best_model(seq):
    model_scores = {}
    for path, model in models.items():
        try:
            with torch.no_grad():
                logits = model(seq)
                probs = torch.softmax(logits, dim=1)
                prob_fake = float(probs[0, 1].item())
                model_scores[path] = prob_fake
        except Exception as e:
            print(f"Error evaluating {path}: {e}")

    if not model_scores:
        raise RuntimeError("No models produced predictions!")

    best_model_path = max(model_scores, key=lambda x: model_scores[x])
    print(f"✅ Using best model: {os.path.basename(best_model_path)} with fake prob={model_scores[best_model_path]:.3f}")
    return models[best_model_path], model_scores[best_model_path]

# ------------------------------- Media Analysis -------------------------------
def analyze_media(filepath):
    ext = filepath.split('.')[-1].lower()

    if ext in ['mp4', 'mov', 'avi']:
        frames = extract_frames(filepath)
        if not frames:
            raise RuntimeError("No frames extracted from video.")
        tensor_frames = [transform(Image.fromarray(f)) for f in frames]
        seq = torch.stack(tensor_frames).unsqueeze(0).to(device)
    else:
        img = Image.open(filepath).convert("RGB")
        img_t = transform(img)
        seq = torch.stack([img_t] * SEQ_LEN).unsqueeze(0).to(device)
        frames = [np.array(img)] * SEQ_LEN

    model, prob_fake = select_best_model(seq)

    # Model-driven prediction
    prediction = "FAKE" if prob_fake > 0.5 else "REAL"
    confidence = prob_fake if prediction == "FAKE" else 1 - prob_fake
    confidence = round(confidence, 3)

    # Features for reference
    features = compute_features(frames, is_video=True)

    # Human-readable explanation
    explanation = explain_features(prediction, features)

    return {
        "prediction": prediction,
        "is_deepfake": prob_fake > 0.5,
        "confidence": confidence,
        "features": features,
        "explanation": explanation
    }


# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from PIL import Image
# from efficientnet_pytorch import EfficientNet
# import mediapipe as mp
# import os

# # ------------------------------- Config -------------------------------
# MODEL_DIR = r"D:\hi\backend\models"
# SEQ_LEN = 16
# IMG_SIZE = 380
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ------------------------------- Model Definition -------------------------------
# class CNNLSTM(nn.Module):
#     def __init__(self, num_classes=2, hidden_size=512, num_layers=1):
#         super().__init__()
#         self.cnn = EfficientNet.from_pretrained('efficientnet-b4')
#         self.cnn_out = 1792
#         for param in self.cnn.parameters():
#             param.requires_grad = True

#         self.lstm = nn.LSTM(self.cnn_out, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         B, T, C, H, W = x.size()
#         feats = []
#         for t in range(T):
#             f = self.cnn.extract_features(x[:, t])
#             f = torch.nn.functional.adaptive_avg_pool2d(f, 1).view(B, -1)
#             feats.append(f)
#         feats = torch.stack(feats, dim=1)
#         lstm_out, _ = self.lstm(feats)
#         last_out = lstm_out[:, -1, :]
#         return self.fc(last_out)

# # ------------------------------- Load Models -------------------------------
# def load_all_models(model_dir=MODEL_DIR):
#     model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
#                    if f.endswith(".pth") or f.endswith(".pt")]
#     models = {}
#     for path in model_paths:
#         try:
#             model = CNNLSTM().to(device)
#             # strict=False avoids mismatched key errors
#             model.load_state_dict(torch.load(path, map_location=device), strict=False)
#             model.eval()
#             models[path] = model
#         except Exception as e:
#             print(f"⚠️ Failed to load {path}: {e}")
#     return models

# models = load_all_models()

# # ------------------------------- Preprocessing -------------------------------
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # ------------------------------- MediaPipe Setup -------------------------------
# mp_face = mp.solutions.face_mesh
# face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

# # ------------------------------- Frame Extraction -------------------------------
# def extract_frames(video_path, seq_len=SEQ_LEN):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     step = max(1, total // seq_len)
#     for i in range(0, total, step):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     cap.release()

#     # Ensure exact seq_len
#     if len(frames) < seq_len:
#         frames = (frames * ((seq_len + len(frames) - 1) // len(frames)))[:seq_len]
#     elif len(frames) > seq_len:
#         frames = frames[:seq_len]
#     return frames

# # ------------------------------- Feature Computation -------------------------------
# def compute_features(frames):
#     facial_movements, eye_blinks, lighting_vals = [], [], []
#     prev_landmarks = None
#     prev_gray = None
#     left_eye_idx = [33, 160, 158, 133, 153, 144]
#     right_eye_idx = [362, 385, 387, 263, 373, 380]

#     def eye_aspect_ratio(pts):
#         A = np.linalg.norm(pts[1] - pts[5])
#         B = np.linalg.norm(pts[2] - pts[4])
#         C = np.linalg.norm(pts[0] - pts[3])
#         return (A + B) / (2.0 * C)

#     for frame in frames:
#         frame_small = cv2.resize(frame, (480, 480))
#         gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)
#         results = face_mesh.process(frame_small)

#         if prev_gray is not None:
#             lighting_vals.append(np.mean(np.abs(gray - prev_gray)) / 255.0)
#         prev_gray = gray

#         movement = 0.0
#         ear_avg = 0.0

#         if results.multi_face_landmarks:
#             landmarks = results.multi_face_landmarks[0].landmark
#             lm = np.array([[p.x * frame_small.shape[1], p.y * frame_small.shape[0]] for p in landmarks])
#             if prev_landmarks is not None:
#                 movement = np.mean(np.linalg.norm(lm - prev_landmarks, axis=1))
#             facial_movements.append(movement)
#             prev_landmarks = lm

#             pts_left = lm[left_eye_idx]
#             pts_right = lm[right_eye_idx]
#             ear_avg = (eye_aspect_ratio(pts_left) + eye_aspect_ratio(pts_right)) / 2
#             eye_blinks.append(ear_avg)
#         else:
#             facial_movements.append(0.0)
#             eye_blinks.append(0.0)

#     features = {
#         'facial_movement': float(np.clip(np.mean(facial_movements) / 20.0, 0, 1)),
#         'eye_blink': float(np.clip(1 - np.mean(eye_blinks), 0, 1)),
#         'lighting': float(np.clip(np.mean(lighting_vals) if lighting_vals else 0, 0, 1))
#     }
#     return features

# # ------------------------------- Explanation -------------------------------
# def explain_features(prediction, features):
#     explanation_parts = []
#     if "eye_blink" in features:
#         explanation_parts.append("normal eye blinks" if prediction == "REAL" else "low/unusual eye blinks")
#     if "facial_movement" in features:
#         explanation_parts.append("normal facial movement" if prediction == "REAL" else "low/unusual facial movement")
#     if "lighting" in features:
#         explanation_parts.append("normal lighting" if prediction == "REAL" else "dark/inconsistent lighting")
#     return f"Video predicted as {prediction} due to " + ", ".join(explanation_parts) + "."

# # ------------------------------- Model Selection -------------------------------
# def select_best_model(seq):
#     model_scores = {}
#     for path, model in models.items():
#         try:
#             with torch.no_grad():
#                 logits = model(seq)
#                 probs = torch.softmax(logits, dim=1)
#                 model_scores[path] = float(probs[0, 1].item())  # FAKE prob
#         except Exception as e:
#             print(f"Error evaluating {path}: {e}")

#     if not model_scores:
#         raise RuntimeError("No models produced predictions!")

#     best_model_path = max(model_scores, key=lambda x: model_scores[x])
#     print(f"✅ Using best model: {os.path.basename(best_model_path)} with fake prob={model_scores[best_model_path]:.2f}")
#     return models[best_model_path], model_scores[best_model_path]

# # ------------------------------- Media Analysis -------------------------------
# def analyze_media(filepath):
#     ext = filepath.split('.')[-1].lower()

#     if ext in ['mp4', 'mov', 'avi']:
#         frames = extract_frames(filepath)
#         if not frames:
#             raise RuntimeError("No frames extracted from video.")
#         tensor_frames = [transform(Image.fromarray(f)) for f in frames]
#         seq = torch.stack(tensor_frames).unsqueeze(0).to(device)
#     else:
#         img = Image.open(filepath).convert("RGB")
#         img_t = transform(img)
#         seq = torch.stack([img_t] * SEQ_LEN).unsqueeze(0).to(device)
#         frames = [np.array(img)] * SEQ_LEN

#     model, prob_fake = select_best_model(seq)
#     prediction = "FAKE" if prob_fake > 0.5 else "REAL"
#     confidence = round(prob_fake if prediction == "FAKE" else 1 - prob_fake, 3)
#     features = compute_features(frames)
#     explanation = explain_features(prediction, features)

#     return {
#         "prediction": prediction,
#         "is_deepfake": prob_fake > 0.5,
#         "confidence": confidence,
#         "features": features,
#         "explanation": explanation
#     }
