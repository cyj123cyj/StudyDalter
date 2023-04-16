# -*- coding: utf-8 -*-
import sys
from PyQt4 import QtGui,QtCore
from encrypter import encrypter
class LoginDialog(QtGui.QDialog):
 def __init__(self, parent=None):
    QtGui.QDialog.__init__(self, parent)


    self.setWindowTitle(u'CT设备成像质量检测系统--注册机')
    self.resize(400, 150)

    self.leName = QtGui.QLineEdit(self)
    self.leName.setPlaceholderText(u'用户名')
    self.plainTextEdit = QtGui.QPlainTextEdit(self)
    self.plainTextEdit.setPlainText(u'说明：输入您的机器码，点击确定按钮以获取授权注册码')
    self.plainTextEdit.setEnabled(0)
    self.lePassword = QtGui.QLineEdit(self)

    self.lePassword.setPlaceholderText(u'注册码')
    self.lePassword.setReadOnly(1)
    self.pbLogin = QtGui.QPushButton(u'确定', self)
    self.pbCancel = QtGui.QPushButton(u'取消', self)

    self.pbLogin.clicked.connect(self.login)
    self.pbCancel.clicked.connect(self.reject)


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

 def login(self):
    code=str(self.leName.text())
    enc=encrypter()
    self.lePassword.setText(str(enc.DesEncrypt(code)))


if __name__ == "__main__":

        app = QtGui.QApplication(sys.argv)
        dialog = LoginDialog()
        dialog.show()
        sys.exit(app.exec_())

