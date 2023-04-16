from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 图片背景
BACKGROUNDBRUSH = QColor(30, 30, 30)
# 滚轮速度，向上为增速
FACTOR = 1
# 窗宽调速
FACTOR0 = 1
# 窗位调速
FACTOR1 = 1
# 标记画笔
PEN = QPen(Qt.red, 3, Qt.SolidLine)

if __name__ == "__main__":
    print(BACKGROUNDBRUSH)
    print(PEN)
