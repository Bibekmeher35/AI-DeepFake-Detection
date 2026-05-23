# Complete Setup Guide - AI DeepFake Detection System

## Table of Contents
1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [Features](#features)
6. [Troubleshooting](#troubleshooting)
7. [Project Structure](#project-structure)

---

## Quick Start

### For Impatient Users:
```bash
cd Django_Application
python3 manage.py runserver
```
Then visit: **http://127.0.0.1:8000/**

---

## System Requirements

### Software
- **Python**: 3.13.5 (or 3.11+)
- **Operating System**: macOS, Linux, or Windows
- **FFmpeg**: Required for audio extraction

### Hardware
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional (CUDA/MPS for faster processing)

### Verified Environment
- ✅ Python 3.13.5
- ✅ macOS ARM64 (Apple Silicon)
- ✅ Django 5.0.6
- ✅ PyTorch 2.11.0

---

## Installation

### Step 1: Install System Dependencies

**macOS:**
```bash
brew install ffmpeg cmake
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg cmake build-essential
```

**Windows:**
- Download FFmpeg from https://ffmpeg.org/download.html
- Add to PATH

### Step 2: Install Python Dependencies

```bash
cd Django_Application
pip install -r requirements.txt
```

**Installation Time:** 5-10 minutes (depending on internet speed)

### Step 3: Verify Installation

```bash
python3 manage.py check
```

Expected output:
```
System check identified no issues (0 silenced).
```

---

## Running the Application

### Development Server

```bash
python3 manage.py runserver
```

Access at: **http://127.0.0.1:8000/**

### Custom Port

```bash
python3 manage.py runserver 8080
```

### Production Deployment

See `docs/DEPLOYMENT.md` for production setup with Gunicorn and Nginx.

---

## Features

### 1. Video Deepfake Detection

**How it works:**
- Upload video file (mp4, avi, mov, etc.)
- Set sequence length (20-60 frames recommended)
- System analyzes:
  - Face detection with YOLOv8
  - ResNeXt50 + LSTM model prediction
  - FOMM forensic analysis
  - Identity consistency check
  - Audio analysis (if present)

**Supported formats:** mp4, avi, mov, mkv, wmv, flv, webm, 3gp, gif

**Processing time:** 10-90 seconds (depends on video length and sequence)

### 2. Image Deepfake Detection

**How it works:**
- Upload image file (jpg, png, etc.)
- EfficientNet B3 model analyzes the image
- Instant results

**Supported formats:** jpg, jpeg, png, bmp, tiff

**Processing time:** 1-3 seconds

### 3. Audio Deepfake Detection

**How it works:**
- Upload audio file (wav, mp3)
- ResNet18 audio model analyzes mel-spectrogram
- Detects synthetic/manipulated audio

**Supported formats:** wav, mp3, flac, ogg

**Processing time:** 2-5 seconds

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'django'"
**Solution:**
```bash
pip install -r requirements.txt
```

#### 2. "TorchCodec is required"
**Solution:** Already fixed in current version. If you see this:
```bash
pip install soundfile
```

#### 3. "Model file not found"
**Check:**
```bash
ls -lh models/model_97_acc_100_frames_FF_data.pt
```
Should show ~216MB file.

#### 4. "FFmpeg not found"
**Solution:**
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

#### 5. Port already in use
**Solution:**
```bash
python3 manage.py runserver 8080
```

#### 6. Out of memory
**Solution:** Reduce sequence length to 20-30 frames

### Debug Mode

Enable detailed logging:
```python
# In project_settings/settings.py
DEBUG = True
```

### Check System Status

```bash
python3 -c "
import django
import torch
import cv2
print('Django:', django.__version__)
print('PyTorch:', torch.__version__)
print('OpenCV:', cv2.__version__)
print('CUDA available:', torch.cuda.is_available())
"
```

---

## Project Structure

```
Django_Application/
├── ml_app/                      # Core application
│   ├── views.py                 # Main logic
│   ├── predict_audio.py         # Audio detection
│   ├── predict_image.py         # Image detection
│   ├── forensics.py             # FOMM analysis
│   ├── forms.py                 # Django forms
│   ├── urls.py                  # URL routing
│   ├── best.pt                  # Audio model (ResNet18)
│   ├── best_deepfake_model.pth  # Image model (EfficientNet)
│   ├── static/                  # CSS, JS, images
│   └── templates/               # HTML templates
│
├── models/                      # ML models
│   ├── model_97_acc_100_frames_FF_data.pt  # Video model (216MB)
│   └── yolov8n-face.pt          # Face detection (6MB)
│
├── media/                       # Runtime uploads
│   ├── uploaded_videos/
│   ├── uploaded_images/
│   └── uploaded_audios/
│
├── project_settings/            # Django configuration
│   ├── settings.py              # Main settings
│   ├── urls.py                  # Root URL config
│   └── wsgi.py                  # WSGI config
│
├── static/                      # Static assets
│   ├── bootstrap/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/                   # Base templates
│   ├── base.html
│   ├── nav-bar.html
│   └── footer.html
│
├── docs/                        # Documentation
│   ├── SETUP_GUIDE.md          # This file
│   ├── TROUBLESHOOTING.md      # Detailed troubleshooting
│   └── API.md                  # API documentation
│
├── tests/                       # Test scripts
│
├── requirements.txt             # Python dependencies
├── manage.py                    # Django management
├── Dockerfile                   # Docker configuration
├── db.sqlite3                   # SQLite database
└── README.md                    # Quick reference
```

---

## Model Files

### Required Models (All Present ✅)

1. **Video Model**
   - File: `models/model_97_acc_100_frames_FF_data.pt`
   - Size: 216MB
   - Architecture: ResNeXt50 + LSTM
   - Purpose: Video deepfake detection

2. **Face Detection**
   - File: `models/yolov8n-face.pt`
   - Size: 6MB
   - Architecture: YOLOv8
   - Purpose: Face detection in videos/images

3. **Audio Model**
   - File: `ml_app/best.pt`
   - Architecture: ResNet18 Audio
   - Purpose: Audio deepfake detection

4. **Image Model**
   - File: `ml_app/best_deepfake_model.pth`
   - Architecture: EfficientNet B3
   - Purpose: Image deepfake detection

---

## Dependencies

### Core Frameworks
- Django 5.0.6
- PyTorch 2.11.0
- TorchVision 0.26.0
- TorchAudio 2.11.0

### Deep Learning
- ultralytics (YOLOv8)
- timm (EfficientNet)
- torchvision.models (ResNeXt)

### Computer Vision
- opencv-python
- face-recognition
- dlib

### Audio Processing
- soundfile
- pydub
- ffmpeg-python
- audioop-lts

### Scientific Computing
- numpy
- scikit-learn
- pandas
- matplotlib

**Full list:** See `requirements.txt`

---

## Performance Tips

### 1. Optimize Sequence Length
- **Fast**: 20 frames (~10 seconds)
- **Balanced**: 40 frames (~30 seconds)
- **Accurate**: 60 frames (~60 seconds)

### 2. File Size Limits
- Keep videos under 100MB
- Compress large files before upload

### 3. Hardware Acceleration
- **Apple Silicon**: Automatic MPS acceleration
- **NVIDIA GPU**: Automatic CUDA acceleration
- **CPU Only**: Works but slower

### 4. Batch Processing
For multiple files, use the API (see `docs/API.md`)

---

## Security Notes

### For Production Deployment:

1. **Change SECRET_KEY** in `settings.py`
2. **Set DEBUG = False**
3. **Update ALLOWED_HOSTS**
4. **Use HTTPS**
5. **Add authentication**
6. **Use PostgreSQL/MySQL** instead of SQLite
7. **Set up proper logging**
8. **Enable CSRF protection**

---

## API Endpoints

- `/` - Home page
- `/video/` - Video upload
- `/image/` - Image upload
- `/audio/` - Audio upload
- `/predict/` - Prediction results
- `/about/` - About page

---

## Testing

### Run System Check
```bash
python3 manage.py check
```

### Test Video Detection
1. Upload a test video
2. Set sequence length to 20
3. Submit and wait for results

### Test Image Detection
1. Upload a test image
2. View instant results

### Test Audio Detection
1. Upload a WAV or MP3 file
2. View results

---

## Support

### Documentation
- Setup Guide: `docs/SETUP_GUIDE.md` (this file)
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- Model Training: `../Model Creation/Readme.md`
- Project Overview: `../README.md`

### Common Questions

**Q: How accurate is the detection?**
A: Depends on the model training. Current model achieves ~97% accuracy on FaceForensics++ dataset.

**Q: Can I train my own model?**
A: Yes! See `Model Creation/` directory for training notebooks.

**Q: Does it work offline?**
A: Yes, once dependencies are installed.

**Q: Can I use it commercially?**
A: Check the LICENSE file. GPL v3 license applies.

---

## Updates and Maintenance

### Check for Updates
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Database Migrations
```bash
python3 manage.py makemigrations
python3 manage.py migrate
```

### Collect Static Files
```bash
python3 manage.py collectstatic
```

---

## Contributing

Contributions welcome! See main `README.md` for guidelines.

---

## License

GPL v3 - See LICENSE file

---

## Credits

**Developed by:**
- Bibek Meher (Lead)
- Anshuman Mishra
- Karan Dev Gorai
- Satwik Shivam
- Aditi Kumari Singh
- Prabhati Karmakar

**Institution:** C V Raman Global University, Bhubaneswar

---

**Last Updated:** 2025
**Version:** 2.0
**Status:** Production Ready ✅
