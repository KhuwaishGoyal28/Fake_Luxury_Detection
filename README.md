
# Fake Luxury Brand Detection

## Features
- Uses Deep Learning (CNN - MobileNetV2)
- Real-time camera-based detection
- Integrates Google Gemini AI for additional verification

## Setup Instructions

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python train_model.py
```

### Start Flask API
```bash
python app.py
```

### Open the Camera for Detection
```bash
python camera_client.py
```

## How It Works
1. The user captures an image of a luxury product.
2. The image is sent to a CNN model for fake detection.
3. Google Gemini AI also verifies based on product name and brand.
4. The final decision is displayed (Real or Fake).
