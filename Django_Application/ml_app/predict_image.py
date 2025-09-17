import torch
import timm
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os
from termcolor import colored

# ---- Pretty printing helpers ----
def light_red(text):
    LIGHT_RED_BOLD = '\033[1;91m'  # Bold bright red
    RESET = '\033[0m'
    return f"{LIGHT_RED_BOLD}{text}{RESET}"

def bold_green(text):
    BOLD_GREEN = '\033[1;32m'  # Bold green
    RESET = '\033[0m'
    return f"{BOLD_GREEN}{text}{RESET}"

# ---- Arch-aware config to mirror Train_model.py ----
MODEL_CONFIGS = {
    "efficientnet_b3": {
        "size": 300,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}

def get_config_for_arch(arch: str):
    return MODEL_CONFIGS.get(arch, MODEL_CONFIGS["efficientnet_b3"])

def build_transform(arch: str):
    cfg = get_config_for_arch(arch)
    size, mean, std = cfg["size"], cfg["mean"], cfg["std"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def validate_image_file(file_path):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    _, ext = os.path.splitext(file_path.lower())
    if ext not in valid_extensions:
        raise ValueError(f"File '{file_path}' is not a supported image type.")
    return True

# ---- Model loading ----
def load_model(model_path, device, arch=None):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    checkpoint = torch.load(model_path, map_location=device)

    if "epoch" in checkpoint:
        print(colored(f"Loaded model trained up to epoch {checkpoint['epoch']}", 'cyan'))
    if "best_acc" in checkpoint:
        print(colored(f"Best validation accuracy: {checkpoint['best_acc']:.2f}", 'cyan'))

    # Get architecture from checkpoint if not provided (training saves 'arch' in checkpoints)
    if arch is None and "arch" in checkpoint:
        arch = checkpoint["arch"]
        print(colored(f"Using architecture from checkpoint: {arch}", 'cyan'))

    # Fallback to efficientnet_b3 if still None (mirrors training’s default)
    if arch is None:
        arch = "efficientnet_b3"
        print(colored(f"No arch provided or stored; defaulting to {arch}", 'yellow'))

    try:
        model = timm.create_model(arch, pretrained=False, num_classes=1)
    except Exception as e:
        raise RuntimeError(f"Error creating model architecture: {e}")

    # Load weights from state_dict if available
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")

    model.to(device)
    model.eval()
    return model, arch

# ---- Prediction with TTA ----
def predict_image(image_path, model, device, transform, threshold=0.55, use_tta=True):
    """
    Returns: label (str), prob_fake (float in [0,1])
    Semantics: sigmoid(logit) = P(fake), matching training labels (0=Real, 1=Fake)
    Uses simple Test-Time Augmentation (original + horizontal flip) if use_tta is True.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{image_path}' not found.")
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify image file '{image_path}'.")
    except Exception as e:
        raise RuntimeError(f"Error opening image '{image_path}': {e}")

    probs = []
    if use_tta:
        flips = [False, True]
    else:
        flips = [False]
    for flip in flips:
        img = image.transpose(Image.FLIP_LEFT_RIGHT) if flip else image
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)          # shape [1,1]
            probs.append(torch.sigmoid(outputs).item())  # scalar

    # Average probability across TTA
    prob_fake = sum(probs) / len(probs)

    if prob_fake >= threshold:
        return "Fake", prob_fake
    else:
        return "Real", prob_fake

def predict_image_from_file(image_path, model_path="best_deepfake_model.pth", device=None, threshold=0.55):
    """
    High-level function to load model, build transform, validate image, and predict label & confidence.
    Returns dictionary: {'label': 'Fake' or 'Real', 'confidence': float in [0,1]}
    """
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validate_image_file(image_path)

    model, arch = load_model(model_path, device)

    transform = build_transform(arch)

    label, prob_fake = predict_image(image_path, model, device, transform, threshold=threshold, use_tta=True)

    confidence = prob_fake if label == "Fake" else 1 - prob_fake

    return {'label': label, 'confidence': confidence}

def main():
    import argparse
    import sys
    import tkinter as tk
    from tkinter import filedialog

    parser = argparse.ArgumentParser(description="Deepfake Image Detection (matched to training pipeline)")
    parser.add_argument('--model', '-m', type=str, default='best_deepfake_model.pth',
                        help="Path to the model weights file (default: best_deepfake_model.pth)")
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help="Decision threshold for P(fake) (default 0.5)")
    parser.add_argument('--arch', '-a', type=str, default=None,
                        help="Model architecture; if omitted, will use the 'arch' stored in the checkpoint")
    parser.add_argument('--image', '-i', type=str, default=None,
                        help="Optional: path to an image file; if omitted, a file picker will open")
    parser.add_argument('--no-tta', action='store_true', help="Disable Test-Time Augmentation (default: enabled)")
    args = parser.parse_args()

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(colored("GPU detected. Using CUDA for inference.", 'green'))
    else:
        print(colored("GPU not detected. Using CPU for inference.", 'yellow'))

    # Get image path (CLI or file picker)
    image_path = args.image
    if not image_path:
        print("Select one image file for deepfake detection.")
        root = tk.Tk()
        root.withdraw()
        default_dir = ""  # change if a specific default folder is desired
        image_path = filedialog.askopenfilename(
            title="Select an image for Deepfake Detection",
            initialdir=default_dir,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not image_path:
            print(colored("No image selected. Exiting.", 'yellow'))
            sys.exit(0)

    try:
        validate_image_file(image_path)
    except ValueError as e:
        print(colored(f"Error: {e}", 'red'))
        sys.exit(1)

    try:
        model, used_arch = load_model(args.model, device, arch=args.arch)
    except (FileNotFoundError, RuntimeError) as e:
        print(colored(f"Error loading model: {e}", 'red'))
        sys.exit(1)

    transform = build_transform(used_arch)

    try:
        label, prob_fake = predict_image(image_path, model, device, transform, threshold=args.threshold, use_tta=not args.no_tta)
    except Exception as e:
        print(colored(f"Error during prediction: {e}", 'red'))
        sys.exit(1)

    # Print result with color and consistent confidence reporting
    if label == "Fake":
        # Confidence is P(fake) and P(real)
        print(light_red(f"{image_path} → Fake: {prob_fake*100:.1f}% | Real: {(1 - prob_fake)*100:.1f}%"))
    else:
        # Confidence is P(real) and P(fake)
        print(bold_green(f"{image_path} → Real: {(1 - prob_fake)*100:.1f}% | Fake: {prob_fake*100:.1f}%"))

if __name__ == "__main__":
    main()
