import sys

from PyQt5.Qt import *
from viewers.register import *
from register_code.encrypter import Encrypter


class LoginDialog(QDialog):
    def __init__(self, title,username, plain_text, logincode,pblogin,cancel, parent=None):
        QDialog.__init__(self, parent)
        self.same(title, username,plain_text,logincode,pblogin,cancel)

    def same(self, title,username, plain_text,logincode, pblogin,cancel):
        self.setWindowTitle(title)
        self.resize(400, 150)
        self.leName = QLineEdit(self)
        self.leName.setPlaceholderText(username)
        self.plainTextEdit = QPlainTextEdit(self)
        self.plainTextEdit.setPlainText(plain_text)
        self.lePassword = QLineEdit(self)
        self.lePassword.setPlaceholderText(logincode)
        self.pbLogin = QPushButton(pblogin, self)
        self.pbCancel = QPushButton(cancel, self)
        self.pbCancel.clicked.connect(self.reject)
        layout = QVBoxLayout()
        layout.addWidget(self.leName)
        layout.addWidget(self.plainTextEdit)
        layout.addWidget(self.lePassword)
        spacerItem = QSpacerItem(20, 48, QSizePolicy.Minimum, QSizePolicy.Expanding)  # 放一个间隔对象美化布局
        layout.addItem(spacerItem)
        buttonLayout = QHBoxLayout()  # 按钮布局
        spancerItem2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        buttonLayout.addItem(spancerItem2)
        buttonLayout.addWidget(self.pbLogin)
        buttonLayout.addWidget(self.pbCancel)
        layout.addLayout(buttonLayout)
        self.setLayout(layout)

    def different_main(self):
        self.leName.setReadOnly(True)
        self.plainTextEdit.setEnabled(False)
        self.pbLogin.clicked.connect(self.login_01)
        self.regist = Register()
        self.leName.setText(str(self.regist.getCVolumeSerialNumber()))

    def different_encrypter(self):
        self.plainTextEdit.setReadOnly(True)
        self.pbLogin.clicked.connect(self.login_02)
        self.lePassword.setReadOnly(True)

    def login_01(self):
        self.regist = Register()
        key = self.lePassword.text()
        if self.regist.regist(key):
            self.accept()  # 关闭对话框并返回1
        else:
            QMessageBox.critical(self, '错误', '注册码错误')

    def login_02(self):
        code = str(self.leName.text())
        enc = Encrypter()
        code1 = enc.DesEncrypt(code)
        code1 = code1.decode("ascii")
        self.lePassword.setText(code1)
