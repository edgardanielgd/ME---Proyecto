# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainInterface.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QSizePolicy,
    QStatusBar, QTextEdit, QWidget)
import Images_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(475, 348)
        MainWindow.setStyleSheet(u"background-image: url(:/Background/Background.jpg);\n"
"background-repeat: repeat;\n"
"color: white;")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.lblWindowTitle = QLabel(self.centralwidget)
        self.lblWindowTitle.setObjectName(u"lblWindowTitle")
        self.lblWindowTitle.setGeometry(QRect(100, 20, 291, 41))
        font = QFont()
        font.setPointSize(15)
        self.lblWindowTitle.setFont(font)
        self.lblWindowTitle.setAutoFillBackground(False)
        self.lblWindowTitle.setStyleSheet(u"background: none;\n"
"background-color: black;\n"
"color: white;\n"
"border-color: white;\n"
"border-style:solid;\n"
"border-width: 2px;")
        self.lblWindowTitle.setTextFormat(Qt.RichText)
        self.lblWindowTitle.setAlignment(Qt.AlignCenter)
        self.btnErase = QPushButton(self.centralwidget)
        self.btnErase.setObjectName(u"btnErase")
        self.btnErase.setGeometry(QRect(80, 70, 111, 31))
        font1 = QFont()
        font1.setPointSize(12)
        self.btnErase.setFont(font1)
        self.btnErase.setStyleSheet(u"QPushButton {\n"
"	background: none;\n"
"	background-color: purple;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	background-color: blue;\n"
"}")
        self.txtText = QTextEdit(self.centralwidget)
        self.txtText.setObjectName(u"txtText")
        self.txtText.setGeometry(QRect(20, 140, 201, 151))
        self.txtText.setStyleSheet(u"background: none;\n"
"border-radius: 10px;\n"
"border-style: solid;\n"
"border-color: black;\n"
"border-width: 2px;\n"
"color: black;\n"
"")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 110, 201, 21))
        font2 = QFont()
        font2.setPointSize(8)
        self.label.setFont(font2)
        self.label.setStyleSheet(u"background: none;\n"
"background-color: black;\n"
"color: white;\n"
"border-width: 1px;\n"
"border-color: white;\n"
"border-style: solid;")
        self.label.setAlignment(Qt.AlignCenter)
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(50, 300, 121, 21))
        self.label_2.setFont(font2)
        self.label_2.setStyleSheet(u"background: none;\n"
"background-color: black;\n"
"border-color: white;\n"
"border-width: 1px;\n"
"border-style: solid;")
        self.label_2.setAlignment(Qt.AlignCenter)
        self.cmbModel = QComboBox(self.centralwidget)
        self.cmbModel.setObjectName(u"cmbModel")
        self.cmbModel.setGeometry(QRect(180, 300, 181, 22))
        self.cmbModel.setStyleSheet(u"background: none;\n"
"color: black;")
        self.cmbModel.setEditable(False)
        self.lstPredictions = QListWidget(self.centralwidget)
        self.lstPredictions.setObjectName(u"lstPredictions")
        self.lstPredictions.setGeometry(QRect(230, 140, 221, 151))
        self.lstPredictions.setStyleSheet(u"background: none;\n"
"border-radius: 10px;\n"
"border-style: solid;\n"
"border-color: black;\n"
"border-width: 2px;\n"
"color: black;")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(240, 110, 201, 21))
        self.label_3.setFont(font2)
        self.label_3.setStyleSheet(u"background: none;\n"
"background-color: black;\n"
"color: white;\n"
"border-width: 1px;\n"
"border-color: white;\n"
"border-style: solid;")
        self.label_3.setAlignment(Qt.AlignCenter)
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(400, 20, 61, 51))
        font3 = QFont()
        font3.setPointSize(30)
        self.label_4.setFont(font3)
        self.label_4.setStyleSheet(u"background: none;\n"
"color: red;\n"
"background-color: rgba(255,255,0,0.5);\n"
"border-style: solid;")
        self.label_4.setAlignment(Qt.AlignCenter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Next Word Predictor", None))
        self.lblWindowTitle.setText(QCoreApplication.translate("MainWindow", u"Next word predictor on rails!", None))
        self.btnErase.setText(QCoreApplication.translate("MainWindow", u"Reset Text", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Type a phrase you want us to complete", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Choose a model to use", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Predicted words", None))
#if QT_CONFIG(tooltip)
        self.label_4.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Datasets used for training the available models contain a limited number of words, if no one could be chosen then you will see that predicted words list won't change until the sequence (depending on the model) is found or trainable in any way</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u26a0\ufe0f", None))
    # retranslateUi

