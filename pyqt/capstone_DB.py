# file: capstone_DB.py

from PyQt6.QtWidgets import QApplication, QLabel

app = QApplication([])
label = QLabel("Hello, PyQt6!")
label.resize(2400, 1000)
label.show()
app.exec()
