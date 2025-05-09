import sys, os
import numpy as np
from PIL import Image, ImageOps
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QColor
from PyQt5.QtCore import Qt, QPoint
from src.predict import predict_digit

class DrawingArea(QLabel):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setFixedSize(403, 300)
    self.setStyleSheet("background-color: transparent;")
    self.canvas = QImage(self.size(), QImage.Format_ARGB32)
    self.canvas.fill(Qt.transparent)
    self.last_point = QPoint()
    self.t, self.forward = 0.0, True
    self.color1, self.color2 = QColor("#B388EB"), QColor("#8EC5FC")

  def lerp_color(self, c1, c2, t):
    r = int(c1.red() * (1 - t) + c2.red() * t)
    g = int(c1.green() * (1 - t) + c2.green() * t)
    b = int(c1.blue() * (1 - t) + c2.blue() * t)
    return QColor(r, g, b)

  def mousePressEvent(self, event):
    if event.button() == Qt.LeftButton:
      self.last_point = event.pos()

  def mouseMoveEvent(self, event):
    if event.buttons() & Qt.LeftButton:
      painter = QPainter(self.canvas)
      pen = QPen(self.lerp_color(self.color1, self.color2, self.t), 16, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
      painter.setPen(pen)
      painter.drawLine(self.last_point, event.pos())
      self.last_point = event.pos()
      self.t += 0.01 if self.forward else -0.01
      if self.t >= 1.0: self.t, self.forward = 1.0, False
      elif self.t <= 0.0: self.t, self.forward = 0.0, True
      self.update()

  def paintEvent(self, event):
    QPainter(self).drawImage(0, 0, self.canvas)

  def clear(self):
    self.canvas.fill(Qt.transparent)
    self.update()

class App(QWidget):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("Handwritten Digit Recognition")
    self.setFixedSize(900, 600)
    self.set_background("ui/design.png")

    self.drawing_area = DrawingArea(self)
    self.drawing_area.move(80, 159)

    self.prediction_label = QLabel("", self)
    self.prediction_label.setStyleSheet("color: #5A4FCF; font-size: 40px; font-weight: bold; background: transparent;")
    self.prediction_label.setAlignment(Qt.AlignCenter)
    self.prediction_label.setFixedSize(100, 60)
    self.prediction_label.move(635, 400)

    self.confidence_label = QLabel("", self)
    self.confidence_label.setStyleSheet("color: #5A4FCF; font-size: 20px; font-weight: bold; background: transparent;")
    self.confidence_label.setAlignment(Qt.AlignCenter)
    self.confidence_label.setFixedSize(150, 30)
    self.confidence_label.move(745, 120)

    self.classify_btn = QPushButton("Classify", self)
    self.classify_btn.move(240, 500)
    self.classify_btn.clicked.connect(self.classify_digit)

    self.clear_btn = QPushButton("Clear", self)
    self.clear_btn.move(645, 507)
    self.clear_btn.clicked.connect(self.drawing_area.clear)

  def set_background(self, path):
    bg = QLabel(self)
    bg.setPixmap(QPixmap(path).scaled(self.width(), self.height(), Qt.KeepAspectRatioByExpanding))
    bg.resize(self.size())
    bg.lower()

  def save_input(self, qimg):
    os.makedirs("drawn_digits", exist_ok=True)
    gray = qimg.convertToFormat(QImage.Format_Grayscale8)
    ptr = gray.bits(); ptr.setsize(gray.byteCount())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((gray.height(), gray.bytesPerLine()))[:, :gray.width()]
    img = ImageOps.invert(Image.fromarray(255 - arr).convert("L")).resize((28, 28), Image.Resampling.LANCZOS)
    path = f"drawn_digits/digit_{len(os.listdir('drawn_digits')) + 1}.png"
    img.save(path)
    print(f"[INFO] Image saved to: {path}")
    return img

  def classify_digit(self):
    image = self.drawing_area.canvas.copy()
    pil_img = self.save_input(image)
    digit, confidence = self.predict(pil_img)
    self.prediction_label.setText(str(digit))
    self.confidence_label.setText(f"{round(confidence * 100)}%")

  def predict(self, img):
    try:
      return predict_digit(img)
    except ImportError:
      print("[ERROR] predict_digit not found.")
      return self.dummy_predict(img), 1.0
    except Exception as e:
      print("[ERROR] Prediction failed:", e)
      return "?", 0.0

  def dummy_predict(self, img):
    return np.random.randint(0, 10)

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = App()
  window.show()
  sys.exit(app.exec_())
