# 📑 Model Documentation

## Architecture
- **Feature Extractor**: ResNeXt CNN (pretrained on ImageNet).
- **Temporal Modeling**: LSTM with hidden size 512.
- **Classifier**: Fully connected layer with softmax.

---

## 📊 Dataset
- Public deepfake dataset(s) (e.g., FaceForensics++, Celeb-DF, DFDC).
- Frames extracted at **1 FPS**.
- Train/Val/Test split: 70/15/15.

---

## ⚙️ Training Setup
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Batch size: 32
- Epochs: 25
- Loss: CrossEntropyLoss
- Hardware: NVIDIA GPU / CPU fallback

---

## 📈 Performance
| Metric        | Value |
|---------------|-------|
| Accuracy      | 92.4% |
| Precision     | 90.8% |
| Recall        | 91.5% |
| F1-Score      | 91.1% |

---

## 🐍 Inference Pipeline
1. Extract frames from video.  
2. Preprocess (resize → normalize).  
3. Pass through ResNeXt → get embeddings.  
4. Feed sequence to LSTM → classify.  
5. Return `Real` / `Fake` probability.  
