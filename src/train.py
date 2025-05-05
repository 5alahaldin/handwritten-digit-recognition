import os
import sys
import itertools
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from dataset import DigitDataset
from model import DigitCNN
from utils import plot_metrics, compute_accuracy

def train_model(data_dir, batch_size=128, epochs=10, lr=0.001):
  torch.backends.cudnn.benchmark = True
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"[INFO] Training on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

  transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Normalize((0.1307,), (0.3081,))
  ])

  try:
    dataset = DigitDataset(root_dir=data_dir, transform=transform)
  except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    return

  try:
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, pin_memory=True)

  except Exception as e:
    print(f"[ERROR] Data loader setup failed: {e}")
    return

  try:
    model = DigitCNN().to(device)
  except Exception as e:
    print(f"[ERROR] Model initialization failed: {e}")
    return

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

  train_losses, val_losses, val_accuracies = [], [], []
  spinner = itertools.cycle(['|', '/', '-', '\\'])

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"\n[~] Epoch {epoch + 1}/{epochs}")

    for i, (images, labels) in enumerate(train_loader):
      try:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      except Exception as e:
        print(f"\n[ERROR] Training batch failed: {e}")
        continue

      if i % 10 == 0: spin_char = next(spinner)
      if i == len(train_loader) - 1: spin_char = "-"
      sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} {spin_char} Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
      sys.stdout.flush()

    scheduler.step()
    val_loss, all_preds, all_labels = 0.0, [], []
    model.eval()
    with torch.no_grad():
      for images, labels in val_loader:
        try:
          images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
          outputs = model(images)
          loss = criterion(outputs, labels)
          val_loss += loss.item()
          all_preds.append(outputs.argmax(dim=1).cpu())
          all_labels.append(labels.cpu())
        except Exception as e:
          print(f"\n[ERROR] Validation batch failed: {e}")
          continue

    val_acc = compute_accuracy(all_preds, all_labels)
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"\n[✔] Epoch {epoch+1}/{epochs} Done - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Acc: {val_acc:.2f}%")

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

  os.makedirs("models", exist_ok=True)
  os.makedirs("visualizations", exist_ok=True)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  filename = f"model_{timestamp}_{val_acc:.2f}.pth"
  model_path = os.path.join("models", filename)
  plot_path = os.path.join("visualizations", filename.replace(".pth", ".png"))

  try:
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
  except Exception as e:
    print(f"[ERROR] Failed to save model: {e}")

  try:
    plot_metrics(train_losses, val_losses, val_accuracies, save_path=plot_path)
    print(f"Plot saved to {plot_path}")
  except Exception as e:
    print(f"[ERROR] Failed to save plot: {e}")

if __name__ == "__main__":
  train_model("data/train")
