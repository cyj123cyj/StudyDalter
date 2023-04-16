import sys

from PyQt5.Qt import *
from pyDes import des
from viewers.register import *
from register_code.encrypter import Encrypter


def test_des():#加密之后转base64编码返回
    a = '123!@#ab'  # 加密字符串必须是8位以上
    key = des(a, CBC,b"\x22\x33\x35\x81\xBC\x38\x5A\xE7",pad=None, padmode=PAD_PKCS5)
    print(base64.b64encode(key.encrypt('abcd123!@#')))


class LoginDialog(QDialog):
    def __init__(self, title, plain_text, pblogin, parent=None):
        QDialog.__init__(self, parent)
        self.same(title, plain_text, pblogin)

    def same(self, title, plain_text, pblogin):
        self.setWindowTitle(title)
        self.resize(400, 150)
        self.leName = QLineEdit(self)
        self.leName.setPlaceholderText('用户名')
        self.plainTextEdit = QPlainTextEdit(self)
        self.plainTextEdit.setPlainText(plain_text)
        self.lePassword = QLineEdit(self)
        self.lePassword.setPlaceholderText('注册码')
        self.pbLogin = QPushButton(pblogin, self)
        self.pbCancel = QPushButton('取消', self)
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
            print("ok")
            self.accept()  # 关闭对话框并返回1
        else:
            QMessageBox.critical(self, '错误', '注册码错误')

    def login_02(self):
        code = str(self.leName.text())
        enc = Encrypter()
        code1 = enc.DesEncrypt(code)
        code1 = code1.decode("ascii")
        self.lePassword.setText(code1)


import pyDes,base64


def test_encrypt(data, key):
    des = pyDes.des(key, pyDes.CBC, b"\0\0\0\0\0\0\0\0", pad=None, padmode=pyDes.PAD_PKCS5)
    ecryptdata = des.encrypt(data)
    return bytes.decode(base64.b64encode(ecryptdata))  # base64 encoding bytes


def test_decrypt(ecryptdata, key):
    des = pyDes.des(key, pyDes.CBC, b"\0\0\0\0\0\0\0\0", pad=None, padmode=pyDes.PAD_PKCS5)
    data = des.decrypt(base64.b64decode(ecryptdata))  # base64 decoding bytes
    return bytes.decode(data)
class Qtest(QWidget):
    def ptest(self):
        QMessageBox.information(self,'出错','图像无法识别',QMessageBox.Ok)


if __name__ == '__main__':
    # a = '5678943'
    # from PyQt5.QtWidgets import QMessageBox
    app = QApplication(sys.argv)
    QMessageBox.information(QWidget(),
                            '出错',
                            '当前图像文件无法自动测量水模体相关指标！',
                            QMessageBox.Ok)
