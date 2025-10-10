# file: hello_pyqt6.py
from PyQt6.QtWidgets import QApplication, QLabel

app = QApplication([])
label = QLabel("Hello, PyQt6!")
label.resize(240, 100)
label.show()
app.exec()
