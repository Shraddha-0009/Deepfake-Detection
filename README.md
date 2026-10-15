# 🔍 AI-Based Deepfake Detection System

An AI-powered web application for detecting manipulated images and videos using **Deep Learning, Computer Vision, and Temporal Analysis**.

The system combines **EfficientNet-B4** for visual feature extraction with a **Bidirectional LSTM** to analyze patterns across a sequence of video frames and classify media as **REAL or FAKE**.

---

## 🎯 Project Overview

Deepfake technology can be used to manipulate digital media and create realistic-looking fake videos or images.

This project aims to provide an automated approach for identifying potentially manipulated media by analyzing visual and temporal patterns.

The system accepts an image or video as input and performs:

- Frame extraction
- Image preprocessing
- Deep visual feature extraction
- Temporal sequence analysis
- Facial landmark analysis
- Eye-related feature analysis
- Facial movement analysis
- Lighting variation analysis
- REAL/FAKE classification

---

## 🚀 Key Features

- 🎥 Video-based deepfake detection
- 🖼️ Image-based analysis
- 🧠 EfficientNet-B4 feature extraction
- 🔄 Bidirectional LSTM temporal analysis
- 👤 Facial landmark detection using MediaPipe
- 👁️ Eye-blink pattern analysis
- 🙂 Facial movement analysis
- 💡 Lighting variation analysis
- 📊 Prediction confidence score
- 📝 Human-readable prediction explanation
- 🌐 Flask-based web interface
- ⚡ CUDA/GPU support when available
- 📁 Supports MP4, MOV, AVI, JPG, JPEG and PNG files

---

## 🧠 System Architecture

```text
              ┌──────────────────────┐
              │   Image / Video      │
              │       Input          │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Frame Extraction    │
              │   & Preprocessing    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   EfficientNet-B4    │
              │ Visual Feature       │
              │ Extraction           │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Bidirectional LSTM   │
              │ Temporal Analysis    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Classification     │
              │    REAL / FAKE       │
              └──────────┬───────────┘
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
       Confidence Score       Feature Analysis
                              • Eye pattern
                              • Facial movement
                              • Lighting
```

---

## 🔬 Model Architecture

### EfficientNet-B4

EfficientNet-B4 is used as the CNN backbone for extracting high-level visual features from input frames.

The final classification layer of EfficientNet is replaced so that the network acts as a feature extractor.

### Bidirectional LSTM

The extracted frame-level features are passed to a Bidirectional LSTM.

This allows the system to analyze temporal information across a sequence of frames rather than treating every frame independently.

### Classification Layer

The temporal features are passed through fully connected layers to produce two classes:

```text
0 → REAL
1 → FAKE
```

---

## 🔄 Detection Pipeline

```text
Input Media
     ↓
Validate File Type
     ↓
Upload to Flask Server
     ↓
Extract Video Frames
     ↓
Resize & Normalize Frames
     ↓
EfficientNet-B4
     ↓
Feature Sequence
     ↓
Bidirectional LSTM
     ↓
Fake Probability
     ↓
REAL / FAKE Prediction
     ↓
Confidence Score
     ↓
Facial & Lighting Feature Analysis
     ↓
Human-Readable Explanation
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| CNN | EfficientNet-B4 |
| Sequence Model | Bidirectional LSTM |
| Computer Vision | OpenCV |
| Facial Analysis | MediaPipe Face Mesh |
| Image Processing | PIL, NumPy |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Hardware Acceleration | CUDA/GPU when available |

---

## 📂 Project Structure

```text
Deepfake-Detection/
│
├── backend/
│   ├── app1.py
│   ├── deepfake_model3.py
│   ├── train.py
│   │
│   ├── templates/
│   │   └── t.html
│   │
│   └── static/
│       └── videos/
│           └── demo.mp4
│
├── README.md
└── .gitignore
```

> Model weight files are not included in this repository. Configure the model directory in the backend before running the application.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shraddha-0009/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch torchvision
pip install flask
pip install opencv-python
pip install numpy
pip install pillow
pip install efficientnet-pytorch
pip install mediapipe
```

### 4. Configure the model directory

The application loads trained `.pth` / `.pt` model files from the configured model directory.

Update the `MODEL_DIR` value in:

```text
backend/deepfake_model3.py
```

to the location containing your trained model weights.

### 5. Start the Flask application

```bash
python backend/app1.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📤 Supported Input Formats

### Video

- MP4
- MOV
- AVI

### Image

- JPG
- JPEG
- PNG

Maximum upload size:

```text
50 MB
```

---

## 📊 Output

After processing the uploaded media, the system provides:

### Prediction

```text
REAL
```

or

```text
FAKE
```

### Confidence

A confidence value representing the model's prediction.

### Feature Analysis

The system additionally analyzes:

- Facial movement
- Eye-related patterns
- Lighting variations

These features are used to generate a human-readable explanation of the prediction.

---

## 🧪 Model Training

The project includes a training pipeline using a sequence of video frames.

Training configuration includes:

```text
Sequence Length: 16 frames
Image Size: 380 × 380
Architecture: EfficientNet-B4 + BiLSTM
LSTM Hidden Size: 256
Classes: 2
Optimizer: Adam
Learning Rate: 0.0001
Epochs: 30
Loss Function: Cross Entropy Loss
```

The training pipeline supports separate training and validation datasets organized into:

```text
train/
├── real/
└── fake/

val/
├── real/
└── fake/
```

---

## 🔐 Relevance to Digital Forensics & Cybersecurity

Deepfake detection can support digital investigation by helping identify potentially manipulated digital evidence.

Potential applications include:

- Digital media verification
- Cybercrime investigation
- Misinformation analysis
- Fraud investigation
- Social media content verification
- Digital evidence screening
- Media authenticity assessment

> The system should be treated as an automated screening/decision-support tool rather than a standalone forensic proof of authenticity.

---

## 💡 Future Enhancements

- Improve model performance using larger and more diverse datasets
- Add audio-based deepfake detection
- Add real-time webcam analysis
- Add multiple deepfake model architectures
- Generate downloadable forensic reports
- Store analysis history and timestamps
- Add explainable AI visualizations
- Add stronger model evaluation metrics
- Containerize the application using Docker
- Deploy the system as a secure cloud application

---

## 👩‍💻 My Contribution

**Role: AI/ML Developer**

Key areas of contribution:

- Designed and implemented the deepfake detection pipeline
- Worked with CNN-based visual feature extraction
- Implemented EfficientNet-B4 feature extraction
- Implemented Bidirectional LSTM temporal analysis
- Integrated OpenCV for video/frame processing
- Integrated MediaPipe for facial landmark analysis
- Developed Flask-based backend integration
- Implemented REAL/FAKE prediction and confidence calculation
- Added facial movement, eye-pattern and lighting analysis
- Worked on model training and validation pipeline

---

## 📌 Project Status

**Status:** Working Prototype

The project demonstrates an end-to-end deepfake detection workflow from media upload to AI-based classification and feature analysis.

---

## 📜 Disclaimer

This project is developed for educational and research purposes.

Deepfake detection models may produce false positives or false negatives. Results should therefore be verified using additional forensic techniques before being used for legal, investigative, or high-stakes decisions.
