import sys
from PyQt5 import QtWidgets, uic

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("untitled.ui", self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyApp()
    mainWindow.show()
    sys.exit(app.exec_())