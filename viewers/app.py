import sys
import PyQt5.QtCore as QtCore

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from viewers.viewer import Viewer
from viewers.languagecheck import LanguageOk


def run_app():  # 主页面显示
    if len(sys.argv) < 2:  # 根据命令行参数断定path是什么
        path = ".."  # 当前所在目录的上一级目录
    else:
        path = sys.argv[1]
    app = QApplication(sys.argv)
    lan = LanguageOk()
    if lan.check_language() == 1:
        trans = QtCore.QTranslator()
        trans.load("ctdicom_en")
        app.installTranslator(trans)
    else:
        pass
    scaleRate = app.screens()[0].logicalDotsPerInch() / 96
    font = QFont()
    font.setPixelSize(int(14 * scaleRate))  # *scaleRate
    app.setFont(font)  # 根据dpi调整字体大小
    viewer = Viewer(path)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
