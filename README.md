# Norwegian Politician Image Classifier

End-to-end image classification system for identifying Norwegian political figures. Combines OpenCV face detection, wavelet feature extraction, and machine learning with a Flask REST API for real-time inference.

## Overview

Multi-class classifier trained to identify five prominent Norwegian politicians:
- Erna Solberg
- Jens Stoltenberg
- Jonas Gahr Støre
- Guri Melby
- Sylvi Listhaug

## Architecture

```
Input Image --> Face Detection --> Eye Validation --> Wavelet Transform --> Feature Vector --> Classification --> Prediction
     |              |                   |                   |                    |                |               |
   Base64      Haar Cascade        >=2 eyes           db1 level 5         32x32 raw +      SVM/RF/LR      Class + 
   or Path     Face + Eye          required           decomposition       32x32 wavelet    Pipeline       Probability
```

## Technical Pipeline

### 1. Preprocessing

**Face Detection:**
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
```

**Quality Filter:**
Images are only processed if:
- Face is detected
- At least 2 eyes are detected within the face region
- This filters out obstructed, low-quality, or non-frontal images

### 2. Feature Engineering

**Dual Feature Approach:**

| Feature Type | Dimensions | Purpose |
|--------------|------------|---------|
| Raw Pixels | 32x32x3 = 3072 | Color and texture information |
| Wavelet Transform | 32x32 = 1024 | Edge and structural features |
| **Combined** | **4096** | Final feature vector |

**Wavelet Transform Implementation:**
```python
def w2d(img, mode='db1', level=5):
    # Convert to grayscale and normalize
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray) / 255
    
    # Compute wavelet coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    
    # Zero out approximation coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    
    # Reconstruct with detail coefficients only
    imArray_H = pywt.waverec2(coeffs_H, mode)
    return np.uint8(imArray_H * 255)
```

The wavelet transform captures facial structural features (eyes, nose, lips contours) that are critical for distinguishing between individuals.

### 3. Model Training

**Algorithm Comparison via GridSearchCV:**

| Model | Hyperparameters | Cross-Validation |
|-------|-----------------|------------------|
| SVM | C: [1,10,100,1000], kernel: [rbf, linear] | 5-fold |
| Random Forest | n_estimators: [1,5,10] | 5-fold |
| Logistic Regression | C: [1,5,10] | 5-fold |

**Pipeline:**
```python
pipe = make_pipeline(StandardScaler(), model)
clf = GridSearchCV(pipe, params, cv=5)
```

**Best Model:** SVM with RBF kernel selected based on test accuracy.

### 4. Inference API

**Flask Endpoint:**
```
POST /classify_image
Content-Type: multipart/form-data
Body: image_data (base64 encoded)

Response:
{
  "class": "jonas_gahr_store",
  "class_probability": [12.5, 8.3, 72.1, 4.2, 2.9],
  "class_dictionary": {"erna_solberg": 0, ...}
}
```

## Running the Application

### Server
```bash
cd server
pip install flask opencv-python numpy pywt joblib scikit-learn
python server.py
```

### Client
Open `client/app.html` in browser or serve via HTTP server.

## Dependencies

| Package | Purpose |
|---------|---------|
| opencv-python | Face and eye detection |
| pywt | Wavelet transforms |
| scikit-learn | SVM, GridSearchCV, StandardScaler |
| joblib | Model serialization |
| numpy | Array operations |
| flask | REST API |

## Dataset

Custom curated dataset with:
- Balanced classes across all five politicians
- Variation in angle, lighting, and context
- Images sourced from public media
- Automatic filtering via eye detection

## Model Artifacts

| File | Description |
|------|-------------|
| saved_model.pkl | Serialized sklearn Pipeline (StandardScaler + SVM) |
| class_dictionary.json | Mapping of politician names to class indices |

## Evaluation

Confusion matrix analysis performed on held-out test set. Model evaluated on:
- Per-class precision and recall
- Overall accuracy
- False positive control across all five classes

## Author

Kesara Rathnasiri
