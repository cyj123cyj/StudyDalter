# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import *
from viewers.login_dialog import LoginDialog


def login_encrypter():
    app = QApplication(sys.argv)
    dialog = LoginDialog('CT设备成像质量检测系统--注册机',
                         '用户名',
                         '说明：输入您的机器码，点击确定按钮以获取授权注册码',
                         '注册码',
                         '确定',
                         '取消')
    dialog.different_encrypter()
    dialog.show()
    sys.exit(dialog.exec_())


if __name__ == "__main__":
    login_encrypter()
