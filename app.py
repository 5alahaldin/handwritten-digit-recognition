import sys, os
import numpy as np
from PIL import Image, ImageOps
from PyQt5.QtWidgets import (
  QApplication, QWidget, QLabel, QPushButton,
  QGraphicsDropShadowEffect, QShortcut
)
from PyQt5.QtGui import (
  QPixmap, QPainter, QPen, QImage, QColor,
  QFontDatabase, QFont, QKeySequence
)
from PyQt5.QtCore import Qt, QPoint
from src.predict import predict_digit

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600
DRAWING_AREA_SIZE = (403, 300)
CLEAR_BTN_SIZE = (124, 60)
CLEAR_BTN_POSITION = (220, 480)
PREDICTION_LABEL_POSITION = (620, 380)
CONFIDENCE_LABEL_POSITION = (725, 115)
SHORTCUT_HINT_POSITION = (535, 510)
PRIMARY_COLOR = "#6C52E6"
SECONDARY_COLOR = "#B5A7F8"

class DrawingArea(QLabel):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setFixedSize(*DRAWING_AREA_SIZE)
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

  def mouseReleaseEvent(self, event):
    if event.button() == Qt.LeftButton:
      self.parent().classify_digit()

  def paintEvent(self, event):
    QPainter(self).drawImage(0, 0, self.canvas)

  def clear(self):
    self.canvas.fill(Qt.transparent)
    self.update()

class OutlinedLabel(QLabel):
  def __init__(self, text="", parent=None):
    super().__init__(text, parent)
    self.outline_color = QColor("#000000")
    self.text_color = QColor("#bdaecd")
    self.font_size = 120
    self.setAttribute(Qt.WA_TranslucentBackground)

  def setOutlineColor(self, color):
    self.outline_color = QColor(color)

  def setTextColor(self, color):
    self.text_color = QColor(color)

  def setFontSize(self, size):
    self.font_size = size
    self.setFont(QFont(self.font().family(), self.font_size, QFont.Bold))

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    font = self.font()
    painter.setFont(font)
    text = self.text()
    rect = self.rect()
    pen = QPen(self.outline_color, 4, Qt.SolidLine)
    painter.setPen(pen)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      painter.drawText(rect.translated(dx, dy), Qt.AlignCenter, text)
    painter.setPen(QPen(self.text_color))
    painter.drawText(rect, Qt.AlignCenter, text)

class App(QWidget):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("Handwritten Digit Recognition")
    self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    self.set_background("ui/design.png")
    font_id = QFontDatabase.addApplicationFont("ui/Perfect-DOS-VGA-437.ttf")
    if font_id != -1:
      font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
      QApplication.setFont(QFont(font_family))

    self.drawing_area = DrawingArea(self)
    self.drawing_area.move(80, 159)

    def create_shadow(widget):
      shadow = QGraphicsDropShadowEffect(self)
      shadow.setBlurRadius(15)
      shadow.setColor(QColor(0, 0, 0, 160))
      shadow.setOffset(0, 2)
      widget.setGraphicsEffect(shadow)

    self.prediction_label = OutlinedLabel("", self)
    self.prediction_label.setFontSize(100)
    self.prediction_label.setOutlineColor(PRIMARY_COLOR)
    self.prediction_label.setTextColor(SECONDARY_COLOR)
    self.prediction_label.setFixedSize(140, 120)
    self.prediction_label.move(*PREDICTION_LABEL_POSITION)

    self.confidence_label = OutlinedLabel("", self)
    self.confidence_label.setFontSize(30)
    self.confidence_label.setOutlineColor(PRIMARY_COLOR)
    self.confidence_label.setTextColor(SECONDARY_COLOR)
    self.confidence_label.setFixedSize(200, 50)
    self.confidence_label.move(*CONFIDENCE_LABEL_POSITION)

    self.shortcut_hint_label = OutlinedLabel("[C] Clear  [Q] Quit", self)
    self.shortcut_hint_label.setFont(QFont(self.font().family(), 19, QFont.Bold))
    self.shortcut_hint_label.setTextColor("#F5E9FD")
    self.shortcut_hint_label.setOutlineColor(PRIMARY_COLOR)
    self.shortcut_hint_label.setFixedSize(300, 30)
    self.shortcut_hint_label.move(*SHORTCUT_HINT_POSITION)
    create_shadow(self.shortcut_hint_label)

    self.clear_btn = QPushButton("", self)
    self.clear_btn.setCursor(Qt.PointingHandCursor)
    self.clear_btn.setFixedSize(*CLEAR_BTN_SIZE)
    self.clear_btn.move(*CLEAR_BTN_POSITION)
    self.clear_btn.setStyleSheet("""
      QPushButton {
        border: none;
        background-image: url(ui/clear_default.png);
        background-repeat: no-repeat;
        background-position: center;
      }
      QPushButton:hover {
        background-image: url(ui/clear_hover.png);
      }
      QPushButton:pressed {
        background-image: url(ui/clear_pressed.png);
      }
    """)
    self.clear_btn.clicked.connect(self.clear_all)
    create_shadow(self.clear_btn)

    clear_shortcut = QShortcut(QKeySequence("C"), self)
    clear_shortcut.activated.connect(self.clear_all)

    exit_shortcut = QShortcut(QKeySequence("Q"), self)
    exit_shortcut.activated.connect(self.close)

  def set_background(self, path):
    bg = QLabel(self)
    bg.setPixmap(QPixmap(path).scaled(self.width(), self.height(), Qt.KeepAspectRatioByExpanding))
    bg.resize(self.size())
    bg.lower()

  def clear_all(self):
    self.drawing_area.clear()
    self.prediction_label.setText("")
    self.confidence_label.setText("")

  def save_input(self, qimg):
    gray = qimg.convertToFormat(QImage.Format_Grayscale8)
    ptr = gray.bits(); ptr.setsize(gray.byteCount())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((gray.height(), gray.bytesPerLine()))[:, :gray.width()]
    img = ImageOps.invert(Image.fromarray(255 - arr).convert("L")).resize((28, 28), Image.Resampling.LANCZOS)

    # os.makedirs("drawn_digits", exist_ok=True)
    # path = f"drawn_digits/digit_{len(os.listdir('drawn_digits')) + 1}.png"
    # img.save(path)
    # print(f"[INFO] Image saved to: {path}")

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
