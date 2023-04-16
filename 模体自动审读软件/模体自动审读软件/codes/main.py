# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, './authorization')
from register import *

from app import run_app

import os
import base64

import  glob
#import sys
from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class LoginDialog(QtGui.QDialog):
 def __init__(self, parent=None):
    QtGui.QDialog.__init__(self, parent)


    self.setWindowTitle(u'CT设备成像质量检测系统--注册')
    self.resize(400, 150)

    self.leName = QtGui.QLineEdit(self)
    self.leName.setPlaceholderText(u'用户名')
    self.leName.setReadOnly(1)
    self.plainTextEdit = QtGui.QPlainTextEdit(self)
    self.plainTextEdit.setPlainText(u'说明：以上是您的机器码，请将此机器码提交给软件提供方，以获取授权注册码并输入')
    self.plainTextEdit.setReadOnly(1)
    self.lePassword = QtGui.QLineEdit(self)
    #self.lePassword.setEchoMode(QtGui.QLineEdit.Password)
    self.lePassword.setPlaceholderText(u'注册码')

    self.pbLogin = QtGui.QPushButton(u'注册', self)
    self.pbCancel = QtGui.QPushButton(u'取消', self)

    self.pbLogin.clicked.connect(self.login)
    self.pbCancel.clicked.connect(self.reject)
    self.regist=register()

    layout = QtGui.QVBoxLayout()
    layout.addWidget(self.leName)
    layout.addWidget(self.plainTextEdit)
    layout.addWidget(self.lePassword)

    # 放一个间隔对象美化布局
    spacerItem = QtGui.QSpacerItem(20, 48, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
    layout.addItem(spacerItem)

    # 按钮布局
    buttonLayout = QtGui.QHBoxLayout()
    # 左侧放一个间隔
    spancerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
    buttonLayout.addItem(spancerItem2)
    buttonLayout.addWidget(self.pbLogin)
    buttonLayout.addWidget(self.pbCancel)

    layout.addLayout(buttonLayout)

    self.setLayout(layout)
    self.leName.setText(str(self.regist.getCVolumeSerialNumber()))

 def login(self):
    key=self.lePassword.text()


    if self.regist.regist(key):
        self.accept()  # 关闭对话框并返回1
    else:
        QtGui.QMessageBox.critical(self, u'错误', u'注册码错误')






if __name__ == "__main__":
    reg=register()
    if reg.checkAuthored() == 1:
        run_app()
    else:
        app = QtGui.QApplication(sys.argv)
        dialog = LoginDialog()
        if dialog.exec_():
            run_app()

