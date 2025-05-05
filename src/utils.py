import torch
import matplotlib.pyplot as plt

def compute_accuracy(preds, labels):
  preds = torch.cat(preds)
  labels = torch.cat(labels)
  correct = (preds == labels).sum().item()
  return 100.0 * correct / labels.size(0)

def plot_metrics(train_losses, val_losses, val_accuracies, save_path=None):
  epochs = range(1, len(train_losses) + 1)
  fig, ax1 = plt.subplots()

  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Loss")
  ax1.plot(epochs, train_losses, label="Train Loss")
  ax1.plot(epochs, val_losses, label="Val Loss")
  ax1.legend(loc="upper left")

  ax2 = ax1.twinx()
  ax2.set_ylabel("Val Accuracy (%)")
  ax2.plot(epochs, val_accuracies, 'g--', label="Val Acc")
  ax2.legend(loc="upper right")

  plt.title("Training Metrics")
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path)
  else:
    plt.show()

  plt.close()