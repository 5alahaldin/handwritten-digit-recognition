import os
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from model import DigitCNN

def get_latest_model(folder="models"):
  try:
    models = [f for f in os.listdir(folder) if f.endswith(".pth")]
    if not models:
      raise FileNotFoundError("No model file found in models/")
    models.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return os.path.join(folder, models[0])
  except Exception as e:
    raise RuntimeError(f"Error locating model in {folder}: {e}")

def load_and_preprocess_image(image_path):
  transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Normalize((0.1307,), (0.3081,))
  ])
  try:
    img = Image.open(image_path).convert('L')
    return transform(img).unsqueeze(0)
  except FileNotFoundError:
    raise FileNotFoundError(f"Image not found: {image_path}")
  except UnidentifiedImageError:
    raise ValueError(f"Unrecognized image file: {image_path}")
  except Exception as e:
    raise RuntimeError(f"Failed to load image: {e}")

def predict_image(image_path, model_path=None):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = DigitCNN().to(device)
  try:
    model_path = model_path or get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
  except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

  try:
    image = load_and_preprocess_image(image_path).to(device)
    model.eval()
    with torch.no_grad():
      output = model(image)
      pred = output.argmax(dim=1).item()
      print(f"Predicted Digit: {pred}")
      return pred
  except Exception as e:
    raise RuntimeError(f"Prediction failed: {e}")

if __name__ == "__main__":
  predict_image("data/test/sample1_28x28.png")
