# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.setObjectName("MainWindow")
        self.resize(800, 524)
        self.img_size = 512
        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setObjectName("centralwidget")

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
        image_0  = QtGui.QPixmap(file_name_0)
        image_0 = image_0.scaled(self.img_size, self.img_size, QtCore.Qt.KeepAspectRatio)
        self.label_img_0 = QtWidgets.QLabel(self.layoutWidget)
        self.label_img_0.setObjectName("label_image_0")
        self.label_img_0.setPixmap(image_0)
        self.label_img_0.setAlignment(QtCore.Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label_img_0)

        file_name_1 = '20.0_49.0_inf.jpeg'
        image_1  = QtGui.QPixmap(file_name_1)
        image_1 = image_1.scaled(self.img_size, self.img_size, QtCore.Qt.KeepAspectRatio)
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
        


app = QtWidgets.QApplication(sys.argv)
# window = QtWidgets.QMainWindow()
ui = MainWindow()
# ui.resize(800, 524)
ui.pushButton_2.clicked.connect(QtWidgets.qApp.quit)
ui.show()
sys.exit(app.exec_())