import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageQt
import numpy as np

from model import generate_image


class StartWindow(QtWidgets.QMainWindow):
    def setupUi(self, StartWindow):
        StartWindow.setObjectName("StartWindow")
        StartWindow.resize(452, 300)
        StartWindow.setWindowTitle("Выберите изображение")

        self.pushButton = QtWidgets.QPushButton(StartWindow)
        self.pushButton.setGeometry(QtCore.QRect(40, 250, 80, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Выход")
        self.pushButton.clicked.connect(QtWidgets.qApp.quit)

        self.pushButton_2 = QtWidgets.QPushButton(StartWindow)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 250, 80, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Запуск")
        self.pushButton_2.clicked.connect(self.on_clicled)
        self.pushButton_2.clicked.connect(StartWindow.close)

        self.pushButton_3 = QtWidgets.QPushButton(StartWindow)
        self.pushButton_3.setGeometry(QtCore.QRect(360, 60, 41, 21))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("...")

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
        self.label.setText("Путь:")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.label_2.setText("Тип:")

        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.comboBox = QtWidgets.QComboBox(self.widget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Видимый-инфракрасный")
        self.comboBox.addItem("Зима-лето")
        self.comboBox.addItem("Лето-Зима")

        self.verticalLayout.addWidget(self.label)
        self.verticalLayout.addWidget(self.label_2)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_2.addWidget(self.lineEdit)
        self.verticalLayout_2.addWidget(self.comboBox)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout.addLayout(self.verticalLayout_2)

        QtCore.QMetaObject.connectSlotsByName(StartWindow)

    def on_clicled(self):
        path = self.lineEdit.text()
        mode = self.comboBox.currentIndex()
        self.new_window = MainWindow(path, mode)
        self.new_window.show()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path, mode):
        QtWidgets.QMainWindow.__init__(self)

        self.path_to_image_0 = path
        self.mode = mode
        self.setObjectName("MainWindow")
        self.resize(800, 524)
        self.img_size = 512
        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setObjectName("centralwidget")
        self.setWindowTitle("Результаты работы")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(622, 60, 150, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Новое изображение")
        self.pushButton.clicked.connect(self.on_clicled_new_image)
        self.pushButton.clicked.connect(self.close)

        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 110, 731, 341))
        self.layoutWidget.setObjectName("layoutWidget")

        self.pushButton_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Выход")
        self.pushButton_2.clicked.connect(QtWidgets.qApp.quit)

        self.pushButton_3 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("Сохранить")
        self.pushButton_3.clicked.connect(self.on_clicked_save_image)

        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        image_0 = self.get_original_image()
        # image_0 = QtGui.QPixmap.fromImage(image_0)
        image_0 = QtGui.QPixmap(self.path_to_image_0)
        image_0 = image_0.copy(0, 0, self.img_shape, self.img_shape)
        image_0 = image_0.scaled(self.img_size, self.img_size,
                                 QtCore.Qt.KeepAspectRatio)
        self.label_img_0 = QtWidgets.QLabel(self.layoutWidget)
        self.label_img_0.setObjectName("label_image_0")
        self.label_img_0.setPixmap(image_0)
        self.label_img_0.setAlignment(QtCore.Qt.AlignCenter)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 60, 281, 17))
        self.label.setObjectName("label")
        # self.label.setText(f"Файл: {self.path_to_image_0}")
        self.label.setText("Файл: {}".format(self.path_to_image_0))

        image_1 = self.get_generated_image()
        image_1 = QtGui.QPixmap.fromImage(image_1)
        image_1 = image_1.scaled(self.img_size, self.img_size,
                                 QtCore.Qt.KeepAspectRatio)

        self.label_img_1 = QtWidgets.QLabel(self.layoutWidget)
        self.label_img_1.setObjectName("label_image_1")
        self.label_img_1.setPixmap(image_1)
        self.label_img_1.setAlignment(QtCore.Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label_img_0)
        self.horizontalLayout.addWidget(self.label_img_1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(500)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.setCentralWidget(self.centralwidget)

        QtCore.QMetaObject.connectSlotsByName(self)

    def get_generated_image(self):
        x = generate_image(self.path_to_image_0, self.mode)
        max_shape = min(x.shape[:2])
        x = x[:max_shape, :max_shape, :]
        img = Image.fromarray(x, mode="RGB")
        img_shape = min(img.size)
        img = img.crop((0, 0, img_shape, img_shape))
        img = img.resize((self.img_size, self.img_size))
        self.image_1 = img
        print(img.size, type(img))
        img = ImageQt.ImageQt(img)
        return img

    def get_original_image(self):
        # 20.0_49.0_vis.jpeg
        # with open('temp.txt') as f:
        #     self.path_to_image_0 = f.readlines()[-1]
        img = Image.open(self.path_to_image_0)
        img.load()
        self.img_shape = min(img.size)
        img = img.crop((0, 0, self.img_shape, self.img_shape))
        img = img.resize((self.img_size, self.img_size))
        print(img.size, type(img), img.mode)
        # img.show()
        img = ImageQt.ImageQt(img)
        return img

    def on_clicked_save_image(self):
        # self.image_1.save(f'{self.path_to_image_0[:-9]}_inf_generated.jpeg')
        self.image_1.save('{}_inf_generated.jpeg'.format(
            self.path_to_image_0[:-9]
            ))

    def on_clicled_new_image(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = StartWindow()
        self.ui.setupUi(self.window)
        self.window.show()


if __name__ == '__main__':
    print(12345)
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = StartWindow()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())
