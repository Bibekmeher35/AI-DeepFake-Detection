# Deep fake detection Django Application

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
### You can watch the demo video
<p align="center">
  <img src="https://github.com/Bibekmeher35/AI-DeepFake-Detection/blob/main/Demo_Video.mp4" />
</p>  
