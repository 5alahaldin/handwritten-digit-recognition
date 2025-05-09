import os
import sys
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from src.model import DigitCNN
import torch.nn.functional as F

def get_latest_model(folder="models"):
  try:
    models = [f for f in os.listdir(folder) if f.endswith(".pth")]
    if not models:
      raise FileNotFoundError(f"No model file found in {folder}")
    models.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return os.path.join(folder, models[0])
  except Exception as e:
    raise RuntimeError(f"Error locating model in {folder}: {e}")

def load_model(model_path=None):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = DigitCNN().to(device)
  try:
    model_path = model_path or get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device
  except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

def preprocess_image(image, is_path=True):
  transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Normalize((0.1307,), (0.3081,))
  ])
  try:
    img = Image.open(image).convert('L') if is_path else Image.fromarray(image.astype(np.uint8)).convert('L')
    return transform(img).unsqueeze(0)
  except FileNotFoundError:
    raise FileNotFoundError(f"Image not found: {image}")
  except UnidentifiedImageError:
    raise ValueError(f"Unrecognized image file: {image}")
  except Exception as e:
    raise RuntimeError(f"Failed to preprocess image: {e}")

def predict(image=None, image_path=None, model_path=None):
  model, device = load_model(model_path)
  if image_path:
    tensor = preprocess_image(image_path).to(device)
  elif image is not None:
    tensor = preprocess_image(image, is_path=False).to(device)
  else:
    raise ValueError("Either `image_path` or `image` (NumPy array) must be provided.")
  try:
    with torch.no_grad():
      output = model(tensor)
      probs = F.softmax(output, dim=1)
      pred = probs.argmax(dim=1).item()
      confidence = probs[0][pred].item()
      return pred, confidence
  except Exception as e:
    raise RuntimeError(f"Prediction failed: {e}")

def predict_digit(pil_img):
  digit, confidence = predict(image=np.array(pil_img))
  print(f"[INFO] Predicted Digit: {digit} (Confidence: {confidence:.2%})")
  return digit, confidence

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)
  image_path = sys.argv[1]
  if not os.path.exists(image_path):
    print(f"Error: File not found: {image_path}")
    sys.exit(1)
  try:
    digit, confidence = predict(image_path=image_path)
    print(f"Predicted Digit: {digit} (Confidence: {confidence:.2%})")
  except Exception as e:
    print("Error:", e)
    sys.exit(1)
