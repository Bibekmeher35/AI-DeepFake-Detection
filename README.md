# AI-DeepFake-Detection
Deepfakes pose a significant threat to digital media authenticity.  
This project provides an **AI-powered DeepFake Detection System** built with **PyTorch, Django, and Transfer Learning**.

---

## 1. Introduction
This project aims to detect video deepfakes using deep learning techniques like ResNext and LSTM. We achieved deepfake detection by using transfer learning where the pretrained ResNext CNN is used to obtain a feature vector, and the LSTM layer is trained using the features. 

This project has been developed by Bibek Meher (Team: Anshuman Mishra, Karan Dev Gorai, Satwik Shivam, Aditi Kumari Singh, Prabhati Karmakar) at C V Raman Global University, Bhubaneswar.

---

## ✨ Features
- 🎥 **Video DeepFake Detection** using CNN (ResNeXt) + LSTM sequence modeling.
- 🌐 **Web Application** (Django) for uploading and detecting fakes in videos.
- 🐳 **Dockerized** for easy deployment on any machine.
- ⚡ Works on **CPU-only systems** (Non-CUDA/AMD GPU supported).
- 📊 Provides evaluation metrics: Accuracy, Precision, Recall, F1.

---

## 📂 Project Structure
```
AI-DeepFake-Detection/
│── Django_Application/    # Web app for video upload & detection
│── requirements.txt       # Python dependencies
│── Dockerfile             # Container setup
│── README.md              # Project description
```

---

## ⚙️ Installation

### 1. Clone repo
```bash
git clone https://github.com/Bibekmeher35/AI-DeepFake-Detection.git
cd AI-DeepFake-Detection
```

### 2. Create environment
```bash
pip install -r requirements.txt
```

### 3. Run Django app
```bash
cd Django_Application
python manage.py migrate
python manage.py runserver
```
Open: **http://127.0.0.1:8000/**

---

## 🐳 Docker Setup
```bash
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector
```
---

## Demo 
### You can watch [this Youtube video]([https://www.youtube.com/watch?v=_q16aJTXVRE](https://youtu.be/zxbPvGVXMJw)) the demo video
<p align="center">
  <img src="https://github.com/Bibekmeher35/AI-DeepFake-Detection/blob/main/Repo_media/DEMO_Videos.gif" />
</p> 

---

## 📊 Model Details
- **Base CNN**: ResNeXt (transfer learning from ImageNet).
- **Sequence Layer**: LSTM (captures temporal frame dependencies).
- **Input**: Extracted video frames (1 per second or every nth frame).
- **Output**: `Real` or `DeepFake` probability.

# Structure
<p align="center">
  <img src="https://github.com/Bibekmeher35/AI-DeepFake-Detection/blob/main/Repo_media/Full_Model.png" />
</p>
More details: <a href="https://github.com/Bibekmeher35/AI-DeepFake-Detection/edit/main/MODEL.md">MODEL.md</a>

---

## 🚀 Roadmap
- [ ] Add live streaming detection  
- [ ] Optimize inference with ONNX / TensorRT  
- [ ] Add progress bar for large video uploads  
- [ ] REST API endpoints with Django REST Framework  

---

## 👨‍💻 Contributors
- **Bibek Meher** (Maintainer)

---

## 📜 License
GNU GPL v3.0
