# Imports
import os
import sys
import time
import shutil
import logging
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch import nn
import cv2
from PIL import Image as pImage
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from .forms import VideoUploadForm, ImageUploadForm
from ultralytics import YOLO
import face_recognition
from .predict_audio import predict as audio_predict
from .predict_image import predict_image_from_file
from .forensics import fomm_likelihood, bg_face_flow_ratio, boundary_flicker, spectral_slope
from django.views.decorators.csrf import csrf_exempt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from sklearn.cluster import DBSCAN
logger = logging.getLogger(__name__)

# For SimpleNamespace
from types import SimpleNamespace

## === AUDIO HELPER FUNCTIONS ===
def ensure_wav(input_path, output_dir=None):
    """Convert MP3 → WAV if needed. output_dir is the full path where to save the wav."""
    ext = os.path.splitext(input_path)[1].lower()
    if output_dir is None:
        # Default to uploaded_audios folder under MEDIA_ROOT
        output_dir = os.path.join(settings.MEDIA_ROOT, "uploaded_audios")
    os.makedirs(output_dir, exist_ok=True)
    if ext == ".wav":
        return input_path
    elif ext == ".mp3":
        audio = AudioSegment.from_file(input_path, format="mp3")
        wav_basename = os.path.basename(input_path).replace(".mp3", ".wav")
        wav_path = os.path.join(output_dir, wav_basename)
        audio.export(wav_path, format="wav")
        return wav_path
    else:
        raise ValueError("Unsupported audio format")

import subprocess

# --- VIDEO AUDIO EXTRACTION HELPER ---
def extract_audio_from_video(video_path, output_dir):
    """
    Extract audio from video file and save as WAV in output_dir using ffmpeg.
    Returns the path to the extracted WAV file, or None if no audio stream present.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_basename = os.path.splitext(os.path.basename(video_path))[0] + ".wav"
    audio_path = os.path.join(output_dir, audio_basename)
    try:
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output file if exists
            "-i", video_path,
            "-vn",  # disable video
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # sample rate 16kHz
            "-ac", "1",      # mono
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(audio_path):
            return audio_path
        else:
            return None
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return None


def predict_page(request):
    """
    Unified predict page: handles video detection result (always), and audio detection if audio POSTed.
    """
    # Ensure video file exists in session
    if 'file_name' not in request.session:
        return redirect("ml_app:index")
    video_file = request.session['file_name']
    sequence_length = request.session['sequence_length']
    path_to_videos = [video_file]
    video_filename = os.path.basename(video_file)
    video_url = request.session.get('video_url', os.path.join("uploaded_videos", video_filename))
    production_video_name = os.path.join(settings.MEDIA_URL, video_url)

    # === AUDIO HANDLING (extracted + optional uploaded audio) ===
    audio_result = {"label": "No audio prediction available", "confidence": 0.0}
    audio_file_url = None
    audio_upload_dir = os.path.join(settings.MEDIA_ROOT, "uploaded_audios")
    os.makedirs(audio_upload_dir, exist_ok=True)

    # Extract audio from video
    audio_from_video_path = extract_audio_from_video(video_file, output_dir=audio_upload_dir)
    if audio_from_video_path:
        audio_result = detect_audio_file(audio_from_video_path)
        audio_file_url = os.path.join(settings.MEDIA_URL, "uploaded_audios", os.path.basename(audio_from_video_path))

    # Optional: handle user-uploaded audio to override
    if request.method == "POST" and request.FILES.get("audio_file", None):
        uploaded_audio_file = request.FILES["audio_file"]
        base_audio_filename = f"{int(time.time())}_{uploaded_audio_file.name}"
        saved_audio_path = os.path.join(audio_upload_dir, base_audio_filename)
        with open(saved_audio_path, "wb+") as out_audio:
            for chunk in uploaded_audio_file.chunks():
                out_audio.write(chunk)
        wav_path = ensure_wav(saved_audio_path, output_dir=audio_upload_dir)
        audio_result = detect_audio_file(wav_path)
        audio_file_url = os.path.join(settings.MEDIA_URL, "uploaded_audios", os.path.basename(wav_path))

    video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)
    model = load_global_model()
    start_time = time.time()
    preprocessed_images = []
    faces_cropped_images = []
    padding = 60
    faces_found = 0
    scores = {"bg_flow_ratio": [], "boundary_flicker": [], "spec_slope": []}
    prev_frame = None
    # --- Begin: Accurate face summary logic ---
    face_predictions = []
    embeddings = []
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    video_file_name_only = os.path.splitext(video_filename)[0]
    while cap.isOpened() and frame_count < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_name = f"{video_file_name_only}_preprocessed_{frame_count}.png"
        pImage.fromarray(rgb_frame, 'RGB').save(os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name))
        preprocessed_images.append(image_name)

        # --- New: process all faces per frame, avoid double-counting ---
        results = yolo_face_detector(frame, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            continue

        for box_tensor in results[0].boxes:  # iterate over all detected faces
            try:
                box = box_tensor.xyxy[0].cpu().numpy().astype(int)
                left, top, right, bottom = box[0], box[1], box[2], box[3]

                # Apply padding
                h_frame, w_frame = frame.shape[:2]
                pad_left = max(0, left - padding)
                pad_top = max(0, top - padding)
                pad_right = min(w_frame, right + padding)
                pad_bottom = min(h_frame, bottom + padding)
                frame_face = frame[pad_top:pad_bottom, pad_left:pad_right]
                if frame_face is None or frame_face.size == 0:
                    continue

                # Convert to RGB
                rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)

                # Save cropped face
                image_name = f"{video_file_name_only}_cropped_faces_{frame_count}_{faces_found+1}.png"
                pImage.fromarray(rgb_face, 'RGB').save(os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name))
                faces_cropped_images.append(image_name)
                faces_found += 1

                # Compute embedding
                encs = face_recognition.face_encodings(rgb_face)
                if encs:
                    embeddings.append(encs[0])

                # Predict face label for this cropped face
                face_tensor = train_transforms(rgb_face).unsqueeze(0).unsqueeze(0)  # shape (1,1,C,H,W)
                prediction = predict(model, face_tensor, './', video_file_name_only)
                face_label = prediction[0]
                face_predictions.append(face_label)

                # Optional: FOMM metrics for face (as before)
                if prev_frame is not None:
                    try:
                        scores["bg_flow_ratio"].append(bg_face_flow_ratio(prev_frame, frame, box))
                        scores["boundary_flicker"].append(boundary_flicker(prev_frame, frame, box))
                    except Exception as e:
                        logger.info(f"FOMM metric failed: {e}")
                try:
                    face_gray = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
                    scores["spec_slope"].append(spectral_slope(cv2.resize(face_gray, (112,112))))
                except Exception as e:
                    logger.info(f"Spectral slope failed: {e}")

            except Exception as e:
                logger.info(f"Face processing failed: {e}")
        prev_frame = frame
    cap.release()
    # --- End: Accurate face summary logic ---
    logger.info("<=== | Videos Splitting and Face Cropping Done | ===>")
    logger.info("--- %s seconds ---", time.time() - start_time)
    if faces_found == 0:
        return render(request, predict_template_name, {"no_faces": True})
    logger.debug("FOMM raw arrays: flow=%s, flick=%s, spec=%s",
                 str(scores.get('bg_flow_ratio', []))[:200],
                 str(scores.get('boundary_flicker', []))[:200],
                 str(scores.get('spec_slope', []))[:200])
    try:
        heatmap_images = []
        output = ""
        confidence = 0.0
        for i in range(len(path_to_videos)):
            logger.info("<=== | Started Prediction | ===>")
            prediction = predict(model, video_dataset[i], './', video_file_name_only)
            confidence = round(prediction[1], 1)
            output = prediction[0]
            base_pred_idx = prediction[2]
            logger.info("Base model index returned: %d", base_pred_idx)
            logger.info("Prediction: %s Confidence: %s", output, confidence)
            logger.info("<=== | Prediction Done | ===>")
            logger.info("--- %s seconds ---", time.time() - start_time)
        fomm_score = fomm_likelihood(scores)
        from .forensics import identity_variance
        identity_score = identity_variance(embeddings)
        logger.info("Identity variance score: %.2f", identity_score)
        logger.info("FOMM counts: flow=%d, flick=%d, spec=%d",
                    len(scores.get('bg_flow_ratio', [])),
                    len(scores.get('boundary_flicker', [])),
                    len(scores.get('spec_slope', [])))
        final_confidence = confidence
        logger.info("Final decision (base model only): %s (Confidence=%.2f)", output, final_confidence)
        # --- Begin: Cluster embeddings and count unique faces ---
        if embeddings:
            X = np.stack(embeddings)
            clustering = DBSCAN(eps=0.6, min_samples=1).fit(X)
            labels = clustering.labels_

            unique_faces = {}
            for idx, cluster_label in enumerate(labels):
                if cluster_label not in unique_faces:
                    unique_faces[cluster_label] = []
                unique_faces[cluster_label].append(face_predictions[idx])

            final_face_predictions = {}
            for cluster, preds in unique_faces.items():
                if preds.count("FAKE") > preds.count("REAL"):
                    final_face_predictions[cluster] = "FAKE"
                else:
                    final_face_predictions[cluster] = "REAL"

            num_real = sum(1 for v in final_face_predictions.values() if v == "REAL")
            num_fake = sum(1 for v in final_face_predictions.values() if v == "FAKE")
            total_faces = len(final_face_predictions)
            face_summary = f"{total_faces} faces detected: {num_real} Real, {num_fake} Fake"
        else:
            face_summary = "No faces detected"
        # --- End: Cluster embeddings and count unique faces ---
        context = {
            'preprocessed_images': preprocessed_images,
            'faces_cropped_images': faces_cropped_images,
            'heatmap_images': heatmap_images if heatmap_images else None,
            'original_video': production_video_name,
            'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
            'output': output,
            'confidence': final_confidence,
            'fomm_score': fomm_score,
            'identity_score': identity_score,
            'base_pred_idx': base_pred_idx,
            # Audio results for predict.html (if present)
            'result': audio_result,
            'audio_file_url': audio_file_url,
            'video_result': output,
            'face_summary': face_summary
        }
        # Store prediction details in session for later retrieval
        request.session['prediction_details'] = context
        return render(request, predict_template_name, context)
    except Exception as e:
        logger.error("Exception occurred during prediction: %s", e)
        return render(request, 'cuda_full.html')
    

def image_upload_view(request):
    """
    Handles the 'Image' button:
    - GET: Show drag & drop upload form.
    - POST: Run deepfake prediction on uploaded image.
    """
    result_sentence = None
    image_file_url = None

    if request.method == "POST" and request.FILES.get("upload_image_file"):
        image_file = request.FILES["upload_image_file"]
        # Save uploaded image to MEDIA_ROOT/uploaded_images/
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploaded_images")
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"{int(time.time())}_{image_file.name}"
        saved_path = os.path.join(upload_dir, filename)
        with open(saved_path, "wb+") as out_file:
            for chunk in image_file.chunks():
                out_file.write(chunk)

        # Run deepfake prediction using ResNeXt+LSTM pipeline on single image
        try:
            # Load image with OpenCV
            frame = cv2.imread(saved_path)
            if frame is None:
                raise Exception("Could not read uploaded image file.")
            # Detect faces with YOLO
            results = yolo_face_detector(frame, verbose=False)
            if len(results) == 0 or len(results[0].boxes) == 0:
                raise Exception("No faces detected in uploaded image.")
            # Use first detected face
            box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            left, top, right, bottom = box[0], box[1], box[2], box[3]
            face_img = frame[top:bottom, left:right, :]
            if face_img is None or face_img.size == 0:
                raise Exception("Face crop failed.")
            # Apply transforms
            img_tensor = train_transforms(face_img)
            # The model expects shape (1, sequence_length, C, H, W).
            # For single image, sequence_length=1.
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            model = load_global_model()
            prediction = predict(model, img_tensor, './', filename)
            label, conf, pred_idx = prediction[0], round(prediction[1], 2), prediction[2]
            image_file_url = os.path.join(settings.MEDIA_URL, "uploaded_images", filename)
            result_sentence = f"The uploaded image was predicted as {label} with {conf:.2f}% confidence."
        except Exception as e:
            result_sentence = f"Image prediction failed: {e}"

    return render(request, "image_upload.html", {
        "form": ImageUploadForm(),
        "result": result_sentence,
        "image_file_url": image_file_url
    })

# --- AUDIO DEEPFAKE DETECTION UTILITY & VIEW ---
def detect_audio_file(audio_path, request=None):
    """
    Utility to handle audio deepfake detection given a path to an audio file.
    Returns: result dict with 'label' and 'confidence' keys.
    """
    try:
        wav_path = ensure_wav(audio_path)
    except Exception as e:
        return {"label": "Error", "confidence": 0.0, "error": f"Audio conversion failed: {e}"}

    try:
        model_path = os.path.join(settings.BASE_DIR, "ml_app", "best.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = audio_predict(model_path, wav_path, device=device)

        confidence = max(result["probs"].values()) * 100
        return {"label": result["class"], "confidence": round(confidence, 2)}
    except Exception as e:
        return {"label": "Error", "confidence": 0.0, "error": f"Audio detection failed: {e}"}


# --- VIDEO MODEL, DATASET, AND UTILITIES ---
index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
PREDICTION_THRESHOLD = 50.0
INVERT_LABELS = True
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

global_model = None
def load_global_model():
    global global_model
    if global_model is None:
        model = Model(2).to(device)
        model_name = 'model_97_acc_100_frames_FF_data.pt'
        model.load_state_dict(torch.load(os.path.join(settings.PROJECT_DIR, 'models', model_name), map_location=device))
        model.eval()
        global_model = model
        logger.info("Global model loaded")
    return global_model

yolo_face_detector = YOLO(os.path.join(settings.PROJECT_DIR, "models", "yolov8n-face.pt"))

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional, batch_first=True)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        out = self.dp(self.linear1(x_lstm[:, -1, :]))
        return fmap, out

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            results = yolo_face_detector(frame, verbose=False)
            if len(results) == 0 or len(results[0].boxes) == 0:
                continue
            try:
                box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
                left, top, right, bottom = box[0], box[1], box[2], box[3]
                frame = frame[top:bottom, left:right, :]
            except Exception as e:
                logger.info(f"Face crop failed (YOLO): {e}")
                continue
            if i % a == first_frame:
                frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def im_convert(tensor, video_file_name):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def predict(model, img, path='./', video_file_name=""):
    with torch.inference_mode():
        fmap, logits = model(img.to(device))
    logits = sm(logits)
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    _, prediction = torch.max(logits, 1)
    pred_idx = int(prediction.item())
    if INVERT_LABELS:
        idx2label = {0: "REAL", 1: "FAKE"}
    else:
        idx2label = {0: "FAKE", 1: "REAL"}
    output_label = idx2label.get(pred_idx, "UNCERTAIN")
    confidence = logits[:, pred_idx].item() * 100
    return [output_label, confidence, pred_idx]


def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    import glob
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))
    for model_path in list_models:
        model_name.append(os.path.basename(model_path))
    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except IndexError:
            pass
    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]
            accuracy.append(acc)
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[0])
    else:
        logger.info("No model found for the specified sequence length.")
    return final_model

ALLOWED_VIDEO_EXTENSIONS = set(['mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv', 'mov'])
def allowed_video_file(filename):
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        for key in ['file_name', 'preprocessed_images', 'faces_cropped_images']:
            request.session.pop(key, None)
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})
            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            if not allowed_video_file(video_file.name):
                video_upload_form.add_error("upload_video_file", "Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            saved_video_file = 'uploaded_file_' + str(int(time.time())) + "." + video_file_ext
            save_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_videos')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, saved_video_file)
            with open(save_path, 'wb') as vFile:
                shutil.copyfileobj(video_file, vFile)
            request.session['file_name'] = save_path
            request.session['sequence_length'] = sequence_length
            request.session['video_url'] = os.path.join('uploaded_videos', saved_video_file)
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})


def about(request):
    return render(request, about_template_name)

def handler404(request, exception):
    return render(request, '404.html', status=404)

def cuda_full(request):
    return render(request, 'cuda_full.html')

def video_upload_page(request):
    """
    Handles the 'Video' button from index.html.
    - GET: Render video_upload.html.
    - POST: Handle video upload and sequence length.
    """
    if request.method == "GET":
        form = VideoUploadForm()
        return render(request, "video_upload.html", {"form": form})
    elif request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data["upload_video_file"]
            sequence_length = form.cleaned_data["sequence_length"]
            # Validate file extension
            if not allowed_video_file(video_file.name):
                form.add_error("upload_video_file", "Only video files are allowed")
                return render(request, "video_upload.html", {"form": form})
            # Validate sequence length
            if sequence_length <= 0:
                form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, "video_upload.html", {"form": form})
            # Validate file size
            if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                form.add_error("upload_video_file", "Maximum file size 100 MB")
                return render(request, "video_upload.html", {"form": form})
            # Save with timestamped filename in uploaded_videos/
            video_file_ext = video_file.name.split('.')[-1]
            ts = int(time.time())
            saved_video_file = f"uploaded_videos_{ts}.{video_file_ext}"
            save_dir = os.path.join(settings.MEDIA_ROOT, "uploaded_videos")
            os.makedirs(save_dir, exist_ok=True)
            video_save_path = os.path.join(save_dir, saved_video_file)
            with open(video_save_path, "wb+") as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)

            # Store relative path and sequence_length in session
            request.session['file_name'] = video_save_path
            request.session['sequence_length'] = sequence_length
            request.session['video_url'] = os.path.join("uploaded_videos", saved_video_file)

            # Redirect to predict page
            return redirect('ml_app:predict')
        else:
            return render(request, "video_upload.html", {"form": form})

# --- AUDIO UPLOAD PAGE VIEW ---
def audio_upload_view(request):
    result = None
    audio_file_url = None
    if request.method == "POST" and request.FILES.get("audio_file"):
        audio_file = request.FILES["audio_file"]
        # Save audio file in uploaded_audios/ under MEDIA_ROOT
        audio_upload_dir = os.path.join(settings.MEDIA_ROOT, "uploaded_audios")
        os.makedirs(audio_upload_dir, exist_ok=True)
        base_audio_filename = f"{int(time.time())}_{audio_file.name}"
        saved_audio_path = os.path.join(audio_upload_dir, base_audio_filename)
        with open(saved_audio_path, "wb+") as out_audio:
            for chunk in audio_file.chunks():
                out_audio.write(chunk)
        # Ensure wav in uploaded_audios
        wav_path = ensure_wav(saved_audio_path, output_dir=audio_upload_dir)
        # Run detection
        result = detect_audio_file(wav_path)
        # Generate audio URL for client
        audio_filename = os.path.basename(wav_path)
        audio_file_url = os.path.join(settings.MEDIA_URL, "uploaded_audios", audio_filename)

        # Format output sentence for template
        if result and isinstance(result, dict) and "error" not in result:
            formatted_result = f"Detected: {result['label']} (Confidence: {result['confidence']:.2f}%)"
        else:
            formatted_result = result.get("error", "Unknown error") if isinstance(result, dict) else "Unknown error"
    else:
        formatted_result = None

    return render(request, "audio_upload.html", {
        "result": formatted_result,
        "audio_file_url": audio_file_url
    })


# --- PREDICTION DETAILS VIEW ---
def prediction_details_view(request):
    """
    View to display the most recent prediction details stored in the session, including extra context variables.
    """
    details = request.session.get('prediction_details')
    if not details:
        messages.info(request, "No prediction details available.")
        return redirect('ml_app:index')
    # Retrieve additional items from session if present, fallback to details dict if not found
    video_result = details.get('output') if details else None
    face_summary = details.get('face_summary') if details else None
    fomm_score = details.get('fomm_score') if details else None
    raw_identity_score = details.get('identity_score') if details else None
    identity_score = None
    if raw_identity_score is not None:
    # Scale to 0–100
        identity_score = round(raw_identity_score * 10, 2)
    audio_result = details.get('result') if details else None
    audio_file_url = details.get('audio_file_url') if details else None

    # Format video result as readable sentence
    if video_result:
        video_conf = details.get('confidence', 0.0)
        video_result_sentence = f"The video prediction is {video_result} with {video_conf:.2f}% confidence."
    else:
        video_result_sentence = "No video prediction available."

    # Format face summary as: "X faces detected: Y Real, Z Deepfake."
    # But original face_summary is like "3 faces detected, 3 Real"
    if face_summary and "faces detected" in face_summary:
        try:
            # Try to parse face_summary string
            # e.g., "3 faces detected, 3 Real" or "3 faces detected, 3 Deepfake"
            import re
            match = re.match(r"(\d+)\s+faces detected,\s+(\d+)\s+(Real|Deepfake)", face_summary)
            if match:
                total_faces = int(match.group(1))
                count = int(match.group(2))
                label = match.group(3)
                if label == "Real":
                    face_summary_sentence = f"{total_faces} faces detected: {count} Real, 0 Deepfake."
                else:
                    face_summary_sentence = f"{total_faces} faces detected: 0 Real, {count} Deepfake."
            else:
                face_summary_sentence = face_summary
        except Exception:
            face_summary_sentence = face_summary
    else:
        face_summary_sentence = face_summary or "No face summary available."

    # Format audio result as: "The audio was analyzed and predicted as Real with 92.34% confidence."
    if isinstance(audio_result, dict):
        if "error" in audio_result:
            audio_result_sentence = audio_result["error"]
        else:
            audio_label = audio_result.get('label', 'Unknown')
            audio_conf = audio_result.get('confidence', 0.0)
            audio_result_sentence = f"The audio was analyzed and predicted as {audio_label} with {audio_conf:.2f}% confidence."
    elif audio_result:
        audio_result_sentence = audio_result
    else:
        audio_result_sentence = "No audio prediction available."

    context = {
        "details": details,
        "video_result": video_result_sentence,
        "face_summary": face_summary_sentence,
        "fomm_score": fomm_score,
        "identity_score": identity_score,
        "audio_result": audio_result_sentence,
        "audio_file_url": audio_file_url,
    }
    return render(request, "prediction_details.html", context)
