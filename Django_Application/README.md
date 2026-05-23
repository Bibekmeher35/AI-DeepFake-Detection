# AI DeepFake Detection System

Production-ready Django application for detecting deepfakes in videos, images, and audio using deep learning, featuring a state-of-the-art interactive user interface.

## 🚀 Quick Start

```bash
cd Django_Application
pip install -r requirements.txt
python3 manage.py runserver
```

Visit: **http://127.0.0.1:8000/**

## ✨ Features

### Deep Learning Models
- **Video Detection**: ResNeXt50 + LSTM with FOMM forensics
- **Image Detection**: EfficientNet B3 model
- **Audio Detection**: ResNet18 audio analysis
- **Face Detection**: YOLOv8 integration
- **Real-time Processing**: Fast inference on CPU/GPU

### User Interface & Experience
- **Glassmorphic Design**: Sleek, modern aesthetic with translucent panels and dynamic blurring.
- **Dynamic Theme System**: Seamless toggle between Light Mode and Dark Mode.
- **Particle Physics Cursor**: Custom interactive cursor with gravity and trailing particle effects.
- **Cinematic Processing**: Animated scanning overlays and progress visualization during model inference.
- **Responsive Layout**: Full-screen, two-column layout optimized for all modern screens.

## 📋 Requirements

- Python 3.11+
- FFmpeg
- 8GB RAM minimum
- 2GB storage

## 📚 Documentation

- **Setup Guide**: [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) - Complete installation and usage
- **Quick Start**: [`QUICK_START.md`](QUICK_START.md) - Get running in 5 minutes
- **Troubleshooting**: [`docs/`](docs/) - Common issues and solutions

## 🎯 System Status

✅ All dependencies installed and verified  
✅ All models present and loading correctly  
✅ Python 3.13.5 compatible  
✅ PyTorch 2.11.0 with soundfile backend  
✅ Production ready

## 🏗️ Project Structure

```
Django_Application/
├── ml_app/              # Core application logic
├── models/              # ML model files (222MB total)
├── media/               # Runtime uploads
├── static/              # CSS, JS, images
├── templates/           # HTML templates (Glassmorphic UI)
├── docs/                # Documentation
├── requirements.txt     # Dependencies
└── manage.py            # Django management
```

## 🔧 Tech Stack

- **Framework**: Django 5.0.6
- **ML**: PyTorch 2.11.0, Ultralytics, TIMM
- **CV**: OpenCV, face-recognition, dlib
- **Audio**: soundfile, pydub, ffmpeg
- **Frontend**: HTML5, Vanilla CSS3 (Custom Variables), JavaScript (Canvas API)

## 📊 Model Performance

- **Video**: ~97% accuracy (FaceForensics++)
- **Image**: High accuracy on synthetic faces
- **Audio**: Detects synthetic speech

## 🤝 Contributing

Contributions welcome! See main project README for guidelines.

## 👥 Team

- Bibek Meher (Lead)
- Anshuman Mishra
- Karan Dev Gorai
- Satwik Shivam
- Aditi Kumari Singh
- Prabhati Karmakar

**Institution**: C V Raman Global University, Bhubaneswar

## 📄 License

GPL v3 - See LICENSE file

## 🌟 Star this repo if you find it useful!

---

**For detailed setup instructions, see [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md)**
