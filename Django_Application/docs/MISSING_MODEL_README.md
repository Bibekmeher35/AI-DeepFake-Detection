# Missing Model File - IMPORTANT

## ⚠️ Video Deepfake Detection Model Required

The application is missing the trained video deepfake detection model file.

### Required File:
- **Filename**: `model_97_acc_100_frames_FF_data.pt`
- **Location**: `Django_Application/models/`
- **Full Path**: `/Users/bibekmeher/Documents/COLLEGE/Projects/AIML/Video_Deepfake_detection_using_deep_learning-master/Django_Application/models/model_97_acc_100_frames_FF_data.pt`

### Current Status:
✅ YOLOv8 Face Detection Model: `models/yolov8n-face.pt` (Present)
✅ Audio Detection Model: `ml_app/best.pt` (Present)
✅ Image Detection Model: `ml_app/best_deepfake_model.pth` (Present)
❌ Video Detection Model: `models/model_97_acc_100_frames_FF_data.pt` (MISSING)

---

## How to Obtain the Model

### Option 1: Train Your Own Model
Follow the instructions in the `Model Creation/` directory to train your own deepfake detection model using the ResNeXt + LSTM architecture.

1. Navigate to `Model Creation/` directory
2. Follow the notebooks in order:
   - `preprocessing.ipynb` - Prepare your dataset
   - `Model_and_train_csv.ipynb` - Train the model
3. Save the trained model as `model_97_acc_100_frames_FF_data.pt`
4. Copy it to `Django_Application/models/`

### Option 2: Download Pre-trained Model
If the project team has shared a pre-trained model:
1. Download the model file from the shared location
2. Place it in `Django_Application/models/` directory
3. Ensure the filename is exactly: `model_97_acc_100_frames_FF_data.pt`

### Option 3: Use Alternative Model (Temporary)
If you have another trained model file:
1. Place any `.pt` model file in `Django_Application/models/` directory
2. The application will automatically detect and use it
3. Note: Results may vary depending on the model's training

---

## Model Specifications

The expected model should be:
- **Architecture**: ResNeXt50 + LSTM
- **Input**: Sequence of face images (default 20-60 frames)
- **Output**: Binary classification (Real/Fake)
- **Framework**: PyTorch
- **Image Size**: 112x112 pixels

---

## Temporary Workaround

The application has been modified to handle the missing model gracefully:
- ✅ Image detection will still work (uses separate model)
- ✅ Audio detection will still work (uses separate model)
- ⚠️ Video detection will use an untrained model (random weights)
  - Results will not be accurate
  - This is only for testing the application flow

---

## Training Dataset

To train your own model, you'll need:
- **Real Videos**: Genuine, unmanipulated videos
- **Fake Videos**: Deepfake videos (e.g., from FaceForensics++, Celeb-DF datasets)
- **Recommended**: At least 1000+ videos of each class
- **Format**: Common video formats (mp4, avi, etc.)

Popular deepfake datasets:
- FaceForensics++ (FF++)
- Celeb-DF
- DFDC (Deepfake Detection Challenge)
- DeeperForensics-1.0

---

## Verification

Once you've placed the model file, verify it's working:

```bash
cd Django_Application
python3 -c "
import os
import torch
model_path = 'models/model_97_acc_100_frames_FF_data.pt'
if os.path.exists(model_path):
    print('✅ Model file found!')
    try:
        model = torch.load(model_path, map_location='cpu')
        print('✅ Model loads successfully!')
    except Exception as e:
        print(f'❌ Error loading model: {e}')
else:
    print('❌ Model file not found!')
"
```

---

## Next Steps

1. **Obtain the model file** using one of the options above
2. **Place it in the correct directory**: `Django_Application/models/`
3. **Restart the Django server**: `python3 manage.py runserver`
4. **Test video upload** to verify it works

---

## Support

For questions about:
- **Training the model**: Check `Model Creation/Readme.md`
- **Model architecture**: Check `Documentation/` folder
- **Application setup**: Check `QUICK_START.md`

---

**Note**: Without the proper trained model, video deepfake detection will not produce accurate results. The other features (image and audio detection) will continue to work normally.
