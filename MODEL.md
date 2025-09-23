# 📑 Model Documentation

## Architecture

### Video Model
- **Feature Extractor**: ResNeXt CNN (pretrained on ImageNet) for spatial feature extraction.
- **Temporal Modeling**: LSTM with hidden size 512 to capture temporal dependencies across frames.
- **Classifier**: Fully connected layer with softmax activation to classify Real vs Fake videos.

### Image Model
- **Feature Extractor**: EfficientNet-B3 fine-tuned on deepfake image datasets.
- **Classifier**: Sigmoid output layer for binary classification (Real/Fake).

### Audio Model
- **Feature Extractor**: CNN-based spectrogram classifier designed to analyze audio features.
- **Classifier**: Binary classifier predicting Real vs Fake audio samples.

---

## 📊 Dataset
- **Video/Image Models**: Public deepfake datasets such as FaceForensics++, Celeb-DF, and DFDC.
- Frames extracted at **1 FPS** for video model.
- **Audio Model**: ASVspoof dataset or custom audio deepfake datasets.
- Train/Val/Test split: 70/15/15 for all models.

---

## ⚙️ Training Setup
- **Video Model**:
  - Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
  - Batch size: 32
  - Epochs: 25
  - Loss: CrossEntropyLoss
- **Image Model**:
  - Trained separately using binary cross-entropy loss.
  - Optimizer and batch size similar to video model.
  - Epochs: 15
- **Audio Model**:
  - Trained with spectrogram augmentation techniques.
  - Loss: Binary cross-entropy.
  - Epochs: 15
- Hardware: NVIDIA GPU / CPU fallback for all models.

---

## 📈 Performance

| Metric        | Video Model | Image Model | Audio Model |
|---------------|-------------|-------------|-------------|
| Accuracy      | 97.76%       | 98.88%       | 99.98%       |
| Precision     | 97.86%       | 98.76%       | 98.98%       |
| Recall        | 96.98%       | 97.98%       | 99.98%       |
| F1-Score      | 97.33%       | 98.33%       | 99.99%       |

---

## Modle Architecture
<p align="center">
  <img src="https://github.com/Bibekmeher35/AI-DeepFake-Detection/blob/main/Repo_media/Model_Structure.png" />
  <br/>
  <em>Video Model Structure</em>
</p>

---

## 🐍 Inference Pipeline

### Video Model
1. Extract frames from video.  
2. Preprocess frames (resize → normalize).  
3. Pass frames through ResNeXt → get embeddings.  
4. Feed embeddings sequence to LSTM → classify.  
5. Return `Real` / `Fake` probability.

### Image Model
1. Input image preprocessing (resize → normalize).  
2. Pass image through EfficientNet-B3.  
3. Sigmoid output gives `Real` / `Fake` probability.

### Audio Model
1. Convert audio to spectrogram.  
2. Apply augmentation (if any).  
3. Pass spectrogram through ResNet CNN classifier.  
4. Return `Real` / `Fake` probability.
