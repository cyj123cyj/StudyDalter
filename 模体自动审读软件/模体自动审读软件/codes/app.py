#!/usr/bin/env python2
import sys
from languagecheck import languageok
import os
import base64

# Try imports and report missing packages.
error = False

# Just to check presence of essential libraries
#import imports

import qtpy
from qtpy import QtCore, QtWidgets
from PyQt4 import QtGui
from viewer import Viewer

def run_app():

        if len(sys.argv) < 2:
            path = "."
        else:
            path = sys.argv[1]
        app = QtWidgets.QApplication(sys.argv)
        lan=languageok()
        print lan.check_language()
        if lan.check_language()==1:
                trans=QtCore.QTranslator()
                trans.load("ctdicom_en")
                app.installTranslator(trans)
                print 1
        elif lan.check_language()==0:
                print 2

        elif lan.check_language()==-6:
                print 3
        elif lan.check_language()==-10:
                print 4
        QtCore.QCoreApplication.setApplicationName("dicom viewer")
        QtCore.QCoreApplication.setOrganizationName("Gui")
        font = QtGui.QFont()
        pointsize = font.pointSize()
        font.setPixelSize(pointsize*1.2)
        app.setFont(font)
        viewer = Viewer(path)
        viewer.show()
        
        #viewer.showMaximized()

        sys.exit(app.exec_())

if __name__ == "__main__":
        run_app()
