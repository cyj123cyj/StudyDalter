# -*- coding: utf-8 -*-
import sys

sys.path.insert(0, './authorization')

from viewers.register import *
from viewers.app import run_app
from PyQt5.QtWidgets import *
from viewers.login_dialog import LoginDialog


def dialog_main():
    app = QApplication(sys.argv)
    dialog = LoginDialog('CT设备成像质量检测系统--注册',
                         '用户名',
                         '说明：以上是您的机器码，请将此机器码提交给软件提供方，以获取授权注册码并输入',
                         '注册码',
                         '登录',
                         '取消')
    dialog.different_main()
    if dialog.exec_():
        run_app()


if __name__ == "__main__":
    reg = Register()
    if reg.checkAuthored() == 1:
        run_app()
    else:
        dialog_main()
