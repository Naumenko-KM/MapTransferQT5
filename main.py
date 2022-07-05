import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from MainWindow import MainWindow
from StartWindow import Ui_StartWindow
from stub import get_image


class Ui_StartWindow(QtWidgets.QMainWindow):
    def setupUi(self, StartWindow):
        StartWindow.setObjectName("StartWindow")
        StartWindow.resize(452, 300)
        self.pushButton = QtWidgets.QPushButton(StartWindow)
        self.pushButton.setGeometry(QtCore.QRect(40, 250, 80, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(StartWindow)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 250, 80, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(StartWindow.close)
        self.pushButton_3 = QtWidgets.QPushButton(StartWindow)
        self.pushButton_3.setGeometry(QtCore.QRect(360, 60, 41, 21))
        self.pushButton_3.setObjectName("pushButton_3")
        self.widget = QtWidgets.QWidget(StartWindow)
        self.widget.setGeometry(QtCore.QRect(30, 30, 311, 141))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_2.addWidget(self.lineEdit)
        self.comboBox = QtWidgets.QComboBox(self.widget)
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout_2.addWidget(self.comboBox)
        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(StartWindow)
        QtCore.QMetaObject.connectSlotsByName(StartWindow)

    def retranslateUi(self, StartWindow):
        _translate = QtCore.QCoreApplication.translate
        StartWindow.setWindowTitle(_translate("StartWindow",
                                              "Выберите изображение"))
        self.pushButton.setText(_translate("StartWindow", "Выход"))
        self.pushButton_2.setText(_translate("StartWindow", "Запуск"))
        self.pushButton_3.setText(_translate("StartWindow", "..."))
        self.label.setText(_translate("StartWindow", "Путь:"))
        self.label_2.setText(_translate("StartWindow", "Тип:"))

    def next_window(self):
        self.new_window = MainWindow()
        self.new_window.show()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.setObjectName("MainWindow")
        self.resize(800, 524)
        self.img_size = 512
        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setObjectName("centralwidget")
        self.setWindowTitle("MainWindow")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(660, 60, 89, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("New image")

        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 110, 731, 341))
        self.layoutWidget.setObjectName("layoutWidget")
        self.pushButton_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Quit")
        self.pushButton_2.clicked.connect(QtWidgets.qApp.quit)
        self.pushButton_3 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("Save")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 60, 281, 17))
        self.label.setObjectName("label")
        self.label.setText("Filename: Example_1.png")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        file_name_0 = '20.0_49.0_vis.jpeg'
        image_0 = get_image()
        image_0 = QtGui.QPixmap.fromImage(image_0)
        image_0 = image_0.scaled(self.img_size, self.img_size,
                                 QtCore.Qt.KeepAspectRatio)
        self.label_img_0 = QtWidgets.QLabel(self.layoutWidget)
        self.label_img_0.setObjectName("label_image_0")
        self.label_img_0.setPixmap(image_0)
        self.label_img_0.setAlignment(QtCore.Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label_img_0)

        file_name_1 = '20.0_49.0_inf.jpeg'
        image_1 = get_image()
        image_1 = QtGui.QPixmap.fromImage(image_1)
        image_1 = image_1.scaled(self.img_size, self.img_size,
                                 QtCore.Qt.KeepAspectRatio)
        self.label_img_1 = QtWidgets.QLabel(self.layoutWidget)
        self.label_img_1.setObjectName("label_image_1")
        self.label_img_1.setPixmap(image_1)
        self.label_img_1.setAlignment(QtCore.Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label_img_1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(500)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.setCentralWidget(self.centralwidget)

        self.menuBar = QtWidgets.QMenuBar()
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menuBar.setObjectName("menuBar")
        self.menuCreate = QtWidgets.QMenu(self.menuBar)
        self.menuCreate.setObjectName("menuCreate")
        self.menuCreate.setTitle("Change mode")
        self.setMenuBar(self.menuBar)
        self.menuCreate.addSeparator()
        self.menuBar.addAction(self.menuCreate.menuAction())

        QtCore.QMetaObject.connectSlotsByName(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_StartWindow()
    ui.setupUi(window)
    ui.pushButton.clicked.connect(QtWidgets.qApp.quit)
    ui.pushButton_2.clicked.connect(ui.next_window)
    window.show()
    sys.exit(app.exec_())

# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     window = Ui_StartWindow()
#     # window.pushButton.clicked.connect(window.close()
#     window.pushButton.clicked.connect(QtWidgets.qApp.quit)
#     window.show()

#     sys.exit(app.exec_())

    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_StartWindow()
    ui.setupUi(window)
    # ui.resize(800, 524)
    ui.pushButton.clicked.connect(QtWidgets.qApp.quit)
    window.show()
    sys.exit(app.exec_())
