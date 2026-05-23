# Quick Start Guide - AI DeepFake Detection

## ✅ Setup Complete!

All dependencies have been successfully installed and verified.

---

## Current Environment

- **Python**: 3.13.5
- **Django**: 5.0.6
- **PyTorch**: 2.11.0
- **TorchVision**: 0.26.0
- **TorchAudio**: 2.11.0
- **OpenCV**: 4.11.0
- **TIMM**: 1.0.27

---

## Running the Application

### Start Development Server
```bash
cd Django_Application
python3 manage.py runserver
```

The server will start at: **http://127.0.0.1:8000/**

---

## Application Features

1. **Video Deepfake Detection**
   - Upload video files (mp4, avi, mov, etc.)
   - Uses ResNeXt + LSTM model
   - Face detection with YOLOv8
   - FOMM forensic analysis

2. **Image Deepfake Detection**
   - Upload image files (jpg, png, etc.)
   - Uses EfficientNet model
   - Single image analysis

3. **Audio Deepfake Detection**
   - Upload audio files (wav, mp3)
   - Uses ResNet18 audio model
   - Mel-spectrogram analysis

---

## Project Structure

```
Django_Application/
├── ml_app/
│   ├── views.py              # Main application logic
│   ├── predict_audio.py      # Audio detection
│   ├── predict_image.py      # Image detection
│   ├── forensics.py          # FOMM forensic analysis
│   ├── best_deepfake_model.pth  # Image model weights
│   └── best.pt               # Audio model weights
├── models/
│   ├── yolov8n-face.pt       # Face detection model
│   └── model_97_acc_100_frames_FF_data.pt  # Video model
├── media/
│   ├── uploaded_videos/
│   ├── uploaded_images/
│   └── uploaded_audios/
├── manage.py
└── requirements.txt
```

---

## Testing the Application

### 1. Test Video Detection
- Navigate to http://127.0.0.1:8000/
- Click "Video" button
- Upload a video file
- Set sequence length (default: 20-60 frames)
- Click "Submit"

### 2. Test Image Detection
- Navigate to http://127.0.0.1:8000/
- Click "Image" button
- Upload an image file
- View results

### 3. Test Audio Detection
- Navigate to http://127.0.0.1:8000/
- Click "Audio" button
- Upload an audio file (wav or mp3)
- View results

---

## Model Files Required

Ensure these model files exist in the correct locations:

1. **Video Model**: `models/model_97_acc_100_frames_FF_data.pt`
2. **Face Detection**: `models/yolov8n-face.pt` ✅
3. **Image Model**: `ml_app/best_deepfake_model.pth`
4. **Audio Model**: `ml_app/best.pt`

---

## Common Commands

### Run Server
```bash
python3 manage.py runserver
```

### Run on Different Port
```bash
python3 manage.py runserver 8080
```

### Run System Check
```bash
python3 manage.py check
```

### Create Migrations
```bash
python3 manage.py makemigrations
python3 manage.py migrate
```

### Collect Static Files
```bash
python3 manage.py collectstatic
```

---

## Troubleshooting

### Server Won't Start
```bash
# Check for errors
python3 manage.py check

# Verify imports
python3 -c "from ml_app import views"
```

### Port Already in Use
```bash
# Use different port
python3 manage.py runserver 8080
```

### Model Files Missing
- Check if model files exist in `models/` and `ml_app/` directories
- Download missing models from project repository

### FFmpeg Not Found
```bash
# Install FFmpeg
brew install ffmpeg
```

---

## Performance Tips

1. **GPU Acceleration**: If you have a Mac with Apple Silicon, PyTorch will automatically use Metal Performance Shaders (MPS) for acceleration

2. **Sequence Length**: For faster video processing, use lower sequence length (20-40 frames)

3. **File Size**: Keep uploaded videos under 100MB for optimal performance

---

## API Endpoints

- `/` - Home page
- `/video/` - Video upload page
- `/image/` - Image upload page
- `/audio/` - Audio upload page
- `/predict/` - Prediction results page
- `/about/` - About page

---

## Security Notes

⚠️ **Important for Production**:
- Change `SECRET_KEY` in settings.py
- Set `DEBUG = False`
- Update `ALLOWED_HOSTS`
- Use proper database (PostgreSQL/MySQL)
- Enable HTTPS
- Add authentication

---

## Next Steps

1. ✅ Dependencies installed
2. ✅ System check passed
3. 🔄 Start the server: `python3 manage.py runserver`
4. 🔄 Test with sample files
5. 🔄 Review and customize settings
6. 🔄 Deploy to production

---

## Support

For issues or questions:
- Check `REQUIREMENTS_UPDATE_SUMMARY.md` for detailed dependency information
- Review project documentation in `Documentation/` folder
- Check GitHub issues

---

**Ready to detect deepfakes! 🚀**
