# -*- coding: utf-8 -*-
#导入一些模块
from __future__ import division
import os, scipy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys, pylab;
import linear
import xlwt
import xlrd
from xlutils.copy import copy;
from qtpy import QtWidgets, QtCore
from languagecheck import languageok
import pydicom as dicom
import numpy as np
import os.path
import win32api, win32con
from util import *
from dicom_widget_data import DicomData,DicomWidget
from resolution import *
from thickness import SpiralBeads, CT_phantom
from scipy import stats
import matplotlib
matplotlib.use('Qt4Agg', warn=False)
from PyQt4 import QtCore, QtGui
from math import sqrt
from PyQt4.QtCore import SIGNAL
from PyQt4.QtGui import QFileDialog, QDialog, QApplication, QWidget, \
    QPushButton, QLabel, QLineEdit, QHBoxLayout, QFormLayout,QStringListModel,QMessageBox,QVBoxLayout
from PyQt4.QtCore import QDir,QTranslator
from water import Water_Phantom,DICOM,InvalidDicomError
from linear import*
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
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, path = None,min1 = 0,max1 = 0,resolution = 0,av = 0,lcd = 0,noise = 0,homogeneity = 0,mean1 = 0):
        super(Viewer, self).__init__()
        if (win32api.GetSystemMetrics(win32con.SM_CXSCREEN) < 1920):    # 获取系统分辨率
            scale_factor = 1.0    # 控件布局缩放比例以适应Windows设置中的缩放比例
        else: scale_factor = 2.0
        #初始化一些值
        if (scale_factor == 1.0):
            self.setMinimumSize(1350, 750)#分辨率小于1920的情况下默认窗口大小
            self.move(0, 0)
        else:
            self.setMinimumSize(1900,1000)
            self.move(0, 0)
        self.cb = 1
        
        self.mgy = 0
        self.mgy1 = 0
        self.slice8_biaocheng = 0
        self.slice8_shice = 0
        self.slice8_err = 0
        self.slice28_biaocheng = 0
        self.slice28_shice = 0
        self.slice28_err = 0
        self.slice2_biaocheng = 0
        self.slice2_shice = 0
        self.slice2_err = 0
        self.place=0
        self.setWindowTitle(_translate("","医用CT成像设备质量检测系统",None))
        self.file = None
        self.array3 = [0,0,0,0,0,0,0,0]
        self.high_hu = 2000
        self.low_hu = -1024
        self.noise = noise
        self.min1 = min1
        self.max1 = max1
        self.mean1 = mean1
        self.lcd = lcd
        self.resolution = resolution
        self.homogeneity = homogeneity
        self.av = av
        self.av2 = 0
        self.av3 = 0
        self.av5 = 0
        self.av7 = 0
        self.a0 = 0
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = 0
        self.a6 = 0
        self.a7 = 0
        self.a8 = 0
        self.a9 = 0
        self.array1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0]
        self.dcmKVP = 0
        self.dcmXRayTubeCurrent = 0
        self.dcmExposureTime = 0
        self.ctLilunDef = [-1000.0, -630.0, -100.0, 120.0, 365.0, 550.0, 1000.0, 1280.0]
        self.pix_label = DicomWidget(parent = self, kwargs = None)  #pix_label为Dicom图片显示部件
        self.resolution10 = 0
        self.resolution50 = 0
        self.resolution10x = 0
        self.resolution50x = 0
        self.mid = 0
        self.length = 255
        self.diam = 166.3 #钨珠直径
        self.pitch = 90 #钨珠螺距
        self.beadsnum = 180 #钨珠数量
        self.CT_err = 0
        
        scroll_area = QtWidgets.QScrollArea()   #滚动窗口，包含大部分窗口部件，目前未实现滚动
        scroll_area.setWidgetResizable(True)    #设置窗口内部件可跟随窗口大小变化


        self.widget = QtGui.QWidget(scroll_area)    #wiget部分包括各指标计算部件
        
        if (scale_factor == 1.0):   #根据显示比例确定各指标计算区域的位置、大小
            self.widget.setGeometry(QtCore.QRect(520, 0, 580,700))
            self.widget.setFixedSize(580,700)#setMinimumSize(580,640)
        else:
            self.widget.setGeometry(QtCore.QRect(520, 0, 800, 800))
            self.widget.setFixedSize(800,800)#setMinimumSize(600,600)
        
        #对比度滑块设定
        self.horizontalSlider = QtGui.QSlider(scroll_area)
        self.horizontalSlider.setGeometry(QtCore.QRect(0, 514, 512, 19))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider1 = QtGui.QSlider(scroll_area)
        self.horizontalSlider1.setGeometry(QtCore.QRect(0, 542, 512, 19))
        self.horizontalSlider1.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider1.setObjectName(_fromUtf8("horizontalSlider1"))
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(255)
        self.horizontalSlider1.setMinimum(0)
        self.horizontalSlider1.setMaximum(255)
        self.horizontalSlider1.setSliderPosition(self.horizontalSlider1.maximum())
        
        
        self.groupBox_duibidu = QtGui.QGroupBox(scroll_area) #对比度调节组合框
        self.groupBox_duibidu.setMinimumSize(200, 100)
        self.gridLayout_duibidu = QtGui.QGridLayout(self.groupBox_duibidu)
        self.groupBox_duibidu.setTitle(_translate("Viewer", "对比度调节", None))
        self.gridLayout_duibidu.addWidget(self.horizontalSlider,0,0)
        self.gridLayout_duibidu.addWidget(self.horizontalSlider1,2,0)
        
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout_all = QtGui.QGridLayout(self.widget)    #为wiget设置网格布局
        self.gridLayout_all.setObjectName(_fromUtf8("gridLayout_all"))
        self.gridLayout_all.setColumnStretch(0, 1)  #设置布局内每一列伸展的权重
        self.gridLayout_all.setColumnStretch(1, 5)
        
        self.gridLayout_scroll = QtGui.QGridLayout(scroll_area) #scroll_area内添加网格布局，包括图片显示区、对比度调节区、指标计算
        self.gridLayout_scroll.addWidget(self.pix_label,0,0)    #将图片显示加入gridLayout_scroll布局
        self.gridLayout_scroll.addWidget(self.widget,0,1)   #将指标计算部分加入gridLayout_scroll布局
        self.gridLayout_scroll.addWidget(self.groupBox_duibidu,1,0) #将对比度调节部分加入gridLayout_scroll布局

        #各指标计算控件设置
        #1、机架倾角
        self.groupBox_qingjiao = QtGui.QGroupBox(self.widget)
        self.gridLayout_qingjiao= QtGui.QGridLayout(self.groupBox_qingjiao)
        self.lineEdit_angle = QtGui.QLineEdit(self.groupBox_qingjiao)
        self.lineEdit_angle.setEnabled(0)
        self.pushButton_angle = QtGui.QPushButton(self.groupBox_qingjiao)
        self.pushButton_angle.setText(_translate("", "计算倾角", None))
        self.gridLayout_qingjiao.addWidget(self.pushButton_angle,0,0,1,1)
        self.gridLayout_qingjiao.addWidget(self.lineEdit_angle,0,1,1,1)
        
        
        #2、水模体指标计算
        self.groupBox_water = QtGui.QGroupBox(self.widget)
        self.gridLayout_water = QtGui.QGridLayout(self.groupBox_water)
        self.water_pushButton = QtGui.QPushButton(self.groupBox_water)
        self.water_label_CT = QtGui.QLabel(self.groupBox_water)
        self.water_lineEdit_CT = QtGui.QLineEdit(self.groupBox_water)
        self.water_lineEdit_CT.setReadOnly(True)
        self.water_label_noise = QtGui.QLabel(self.groupBox_water)
        self.water_lineEdit_noise = QtGui.QLineEdit(self.groupBox_water)
        self.water_label_junyun = QtGui.QLabel(self.groupBox_water)
        self.water_lineEdit_junyun = QtGui.QLineEdit(self.groupBox_water)
        self.water_label_lcd = QtGui.QLabel(self.groupBox_water)
        self.water_label_lcd_2 = QtGui.QLabel(self.groupBox_water)
        self.water_label_lcd_3 = QtGui.QLabel(self.groupBox_water)
        self.water_label_lcd_5 = QtGui.QLabel(self.groupBox_water)
        self.water_label_lcd_7 = QtGui.QLabel(self.groupBox_water)
        self.water_lineEdit_lcd_2 = QtGui.QLineEdit(self.groupBox_water)
        self.water_lineEdit_lcd_3 = QtGui.QLineEdit(self.groupBox_water)
        self.water_lineEdit_lcd_5= QtGui.QLineEdit(self.groupBox_water)
        self.water_lineEdit_lcd_7 = QtGui.QLineEdit(self.groupBox_water)
        self.groupBox_water.setObjectName(_fromUtf8("groupBox_water"))
        self.gridLayout_water.setObjectName(_fromUtf8("gridLayout_water"))
        self.water_pushButton.setObjectName(_fromUtf8("water_pushButton"))
        self.water_label_CT.setObjectName(_fromUtf8("water_label_CT"))
        self.water_lineEdit_CT.setObjectName(_fromUtf8("water_lineEdit_CT"))
        self.water_label_noise.setObjectName(_fromUtf8("water_label_noise"))
        self.water_lineEdit_noise.setObjectName(_fromUtf8("water_lineEdit_noise"))
        self.water_label_junyun.setObjectName(_fromUtf8("water_label_junyun"))
        self.water_lineEdit_junyun.setObjectName(_fromUtf8("water_lineEdit_junyun"))
        self.water_label_lcd.setObjectName(_fromUtf8("water_label_lcd"))
        self.water_checkbox1 = QtGui.QCheckBox(self.groupBox_water)
        self.water_checkbox2 = QtGui.QCheckBox(self.groupBox_water)
        self.water_checkbox3 = QtGui.QCheckBox(self.groupBox_water)
        self.water_checkbox4 = QtGui.QCheckBox(self.groupBox_water)
        self.water_label_lcd_2.setObjectName(_fromUtf8("2"))
        self.water_label_lcd_3.setObjectName(_fromUtf8("3"))
        self.water_label_lcd_5.setObjectName(_fromUtf8("5"))
        self.water_label_lcd_7.setObjectName(_fromUtf8("7"))
        self.water_lineEdit_lcd_2.setObjectName(_fromUtf8("water_lineEdit_lcd_2"))
        self.water_lineEdit_lcd_3.setObjectName(_fromUtf8("water_lineEdit_lcd_3"))
        self.water_lineEdit_lcd_5.setObjectName(_fromUtf8("water_lineEdit_lcd_5"))
        self.water_lineEdit_lcd_7.setObjectName(_fromUtf8("water_lineEdit_lcd_7"))
        self.gridLayout_water.addWidget(self.water_pushButton, 0, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_label_CT, 1, 0, 1, 4)
        self.gridLayout_water.addWidget(self.water_lineEdit_CT, 1, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_label_noise, 2, 0, 1, 4)
        self.gridLayout_water.addWidget(self.water_lineEdit_noise, 2, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_label_junyun, 3, 0, 1, 4)
        self.gridLayout_water.addWidget(self.water_lineEdit_junyun, 3, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_label_lcd, 4, 0, 1, 4)
        vbox=QVBoxLayout()#纵向布局
        hbox1=QHBoxLayout()#横向布局
        hbox2=QHBoxLayout()#横向布局

        hbox1.addWidget(self.water_checkbox1)#, 0, 0, 1, 1)
        hbox1.addWidget(self.water_label_lcd_2)#, 0, 1, 1, 1)
        hbox1.addWidget(self.water_lineEdit_lcd_2)#, 0, 2, 1, 3)
        
        hbox1.addWidget(self.water_checkbox2)#, 0, 5, 1, 1)
        hbox1.addWidget(self.water_label_lcd_3)#, 0, 6, 1, 1)
        hbox1.addWidget(self.water_lineEdit_lcd_3)#, 0, 7, 1, 3)
        
        hbox2.addWidget(self.water_checkbox3)#, 1, 0, 1, 1)
        hbox2.addWidget(self.water_label_lcd_5)#, 1, 1, 1, 1)
        hbox2.addWidget(self.water_lineEdit_lcd_5)#, 1, 2, 1, 3)
        
        hbox2.addWidget(self.water_checkbox4)#, 1, 5, 1, 1)
        hbox2.addWidget(self.water_label_lcd_7)#, 1, 6, 1, 1)
        hbox2.addWidget(self.water_lineEdit_lcd_7)#, 1, 7, 1, 3)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        self.gridLayout_water.addLayout(vbox,5,0,2,8)
        self.water_pushButton_all = QtGui.QPushButton(self.groupBox_water)
        self.water_pushButton_all.setObjectName(_fromUtf8("water_pushButton_all"))

        self.water_checkbox1.setObjectName(_fromUtf8("water_checkbox1"))
        self.water_checkbox2.setObjectName(_fromUtf8("water_checkbox2"))
        self.water_checkbox3.setObjectName(_fromUtf8("water_checkbox3"))
        self.water_checkbox4.setObjectName(_fromUtf8("water_checkbox4"))
        self.water_pushButton_jz = QtGui.QPushButton(self.groupBox_water)
        self.water_pushButton_jz.setObjectName(_fromUtf8("LCD值"))
        self.gridLayout_water.addWidget(self.water_pushButton_jz, 7, 1, 1, 1)
        self.water_lineEdit_jz = QtGui.QLineEdit(self.groupBox_water)
        self.water_lineEdit_jz.setObjectName(_fromUtf8("water_lineEdit_jz"))
        self.gridLayout_water.addWidget(self.water_lineEdit_jz, 7, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_pushButton_all, 8, 0, 1, 8)
        
        
        #3、CT线性值指标计算
        self.groupBox_linear = QtGui.QGroupBox(self.widget)
        self.gridLayout_linear = QtGui.QGridLayout(self.groupBox_linear)
        self.linear_pushButton = QtGui.QPushButton(self.groupBox_linear)
        self.linear_label_celiang = QtGui.QLabel(self.groupBox_linear)
        self.linear_lineEdit_ctceliang1 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang2 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang3 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang4 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang5 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang6 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang7 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang8 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_pushButton_all = QtGui.QPushButton(self.groupBox_linear)
        self.linear_pushButton_all.setObjectName(_fromUtf8("linear_pushButton_all"))
        self.linear_label_lilun = QtGui.QLabel(self.groupBox_linear)
        self.checkbox1 = QtGui.QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun1 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun1.setEnabled(False)
        self.checkbox2 = QtGui.QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun2 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun2.setEnabled(False)
        self.checkbox3 = QtGui.QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun3 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun3.setEnabled(False)
        self.checkbox4 = QtGui.QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun4 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun4.setEnabled(False)
        self.checkbox5 = QtGui.QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun5 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun5.setEnabled(False)
        self.checkbox6 = QtGui.QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun6 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun6.setEnabled(False)
        self.checkbox7 = QtGui.QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun7 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun7.setEnabled(False)
        self.checkbox8 = QtGui.QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun8 = QtGui.QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun8.setEnabled(False)
        self.pushButton_ctxianxing = QtGui.QPushButton(self.groupBox_linear)
        self.linear_label_CT = QtGui.QLabel(self.groupBox_linear)
        self.linear_lineEdit_CT = QtGui.QLineEdit(self.groupBox_linear)
        self.groupBox_linear.setObjectName(_fromUtf8("groupBox_linear"))
        self.gridLayout_linear.setObjectName(_fromUtf8("gridLayout_linear"))
        self.linear_pushButton.setObjectName(_fromUtf8("linear_pushButton"))
        self.linear_label_celiang.setObjectName(_fromUtf8("linear_label_celiang"))
        self.linear_lineEdit_ctceliang1.setObjectName(_fromUtf8("linear_lineEdit_ctceliang1"))
        self.linear_lineEdit_ctceliang2.setObjectName(_fromUtf8("linear_lineEdit_ctceliang2"))
        self.linear_lineEdit_ctceliang3.setObjectName(_fromUtf8("linear_lineEdit_ctceliang3"))
        self.linear_lineEdit_ctceliang4.setObjectName(_fromUtf8("linear_lineEdit_ctceliang4"))
        self.linear_lineEdit_ctceliang5.setObjectName(_fromUtf8("linear_lineEdit_ctceliang5"))
        self.linear_lineEdit_ctceliang6.setObjectName(_fromUtf8("linear_lineEdit_ctceliang6"))
        self.linear_lineEdit_ctceliang7.setObjectName(_fromUtf8("linear_lineEdit_ctceliang7"))
        self.linear_lineEdit_ctceliang8.setObjectName(_fromUtf8("linear_lineEdit_ctceliang8"))
        self.linear_label_lilun.setObjectName(_fromUtf8("linear_label_lilun"))
        self.checkbox1.setObjectName(_fromUtf8("checkbox1"))
        self.linear_lineEdit_ctlilun1.setObjectName(_fromUtf8("linear_lineEdit_ctlilun1"))
        self.checkbox2.setObjectName(_fromUtf8("checkbox2"))
        self.linear_lineEdit_ctlilun2.setObjectName(_fromUtf8("linear_lineEdit_ctlilun2"))
        self.checkbox3.setObjectName(_fromUtf8("checkbox3"))
        self.linear_lineEdit_ctlilun3.setObjectName(_fromUtf8("linear_lineEdit_ctlilun3"))
        self.checkbox4.setObjectName(_fromUtf8("checkbox4"))
        self.linear_lineEdit_ctlilun4.setObjectName(_fromUtf8("linear_lineEdit_ctlilun4"))
        self.linear_lineEdit_ctlilun5.setObjectName(_fromUtf8("linear_lineEdit_ctlilun5"))
        self.checkbox5.setObjectName(_fromUtf8("checkbox5"))
        self.checkbox6.setObjectName(_fromUtf8("checkbox6"))
        self.linear_lineEdit_ctlilun6.setObjectName(_fromUtf8("linear_lineEdit_ctlilun6"))
        self.checkbox7.setObjectName(_fromUtf8("checkbox7"))
        self.linear_lineEdit_ctlilun7.setObjectName(_fromUtf8("linear_lineEdit_ctlilun7"))
        self.checkbox8.setObjectName(_fromUtf8("checkbox8"))
        self.linear_lineEdit_ctlilun8.setObjectName(_fromUtf8("linear_lineEdit_ctlilun8"))
        self.pushButton_ctxianxing.setObjectName(_fromUtf8("pushButton_ctxianxing"))
        self.linear_label_CT.setObjectName(_fromUtf8("linear_label_CT"))
        self.linear_lineEdit_CT.setObjectName(_fromUtf8("linear_lineEdit_CT"))
        self.gridLayout_linear.addWidget(self.linear_pushButton, 1, 6, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_label_celiang, 1, 0, 1, 6)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang1, 2, 0, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang2, 2, 2, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang3, 2, 4, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang4, 2, 6, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang5, 3, 0, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang6, 3, 2, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang7, 3, 4, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang8, 3, 6, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_pushButton_all, 4, 0, 1, 8)
        self.gridLayout_linear.addWidget(self.linear_label_lilun, 5, 0, 1, 6)
        self.gridLayout_linear.addWidget(self.checkbox1, 6, 0, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun1, 6, 1, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox2, 6, 2, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun2, 6, 3, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox3, 6, 4, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun3, 6, 5, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox4, 6, 6, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun4, 6, 7, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox5, 7, 0, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun5, 7, 1, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox6, 7, 2, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun6, 7, 3, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox7, 7, 4, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun7, 7, 5, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox8, 7, 6, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun8, 7, 7, 1, 1)
        self.gridLayout_linear.addWidget(self.pushButton_ctxianxing, 9, 6, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_label_CT, 9, 0, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_CT, 9, 2, 1, 2)
        
        self.linear_lineEdit_ctlilun1.setText("%.1f"%(self.ctLilunDef[0]))
        self.linear_lineEdit_ctlilun2.setText("%.1f"%(self.ctLilunDef[1]))
        self.linear_lineEdit_ctlilun3.setText("%.1f"%(self.ctLilunDef[2]))
        self.linear_lineEdit_ctlilun4.setText("%.1f"%(self.ctLilunDef[3]))
        self.linear_lineEdit_ctlilun5.setText("%.1f"%(self.ctLilunDef[4]))
        self.linear_lineEdit_ctlilun6.setText("%.1f"%(self.ctLilunDef[5]))
        self.linear_lineEdit_ctlilun7.setText("%.1f"%(self.ctLilunDef[6]))
        self.linear_lineEdit_ctlilun8.setText("%.1f"%(self.ctLilunDef[7]))
        #4、层厚值指标计算
        self.groupBox_sliceThickness = QtGui.QGroupBox(self.widget)
        self.gridLayout_sliceThickness = QtGui.QGridLayout(self.groupBox_sliceThickness)
        self.label_cenghou8 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_pushButton_8 = QtGui.QPushButton(self.groupBox_sliceThickness)
        self.sliceThickness_label_bc8 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_bc8 = QtGui.QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_shice8 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_shice8 = QtGui.QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_error8 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_error8 = QtGui.QLineEdit(self.groupBox_sliceThickness)
        self.label_cenghou28 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_pushButton_28 = QtGui.QPushButton(self.groupBox_sliceThickness)
        self.sliceThickness_label_bc28 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_bc28 = QtGui.QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_shice28 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_shice28 = QtGui.QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_error28 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_error28 = QtGui.QLineEdit(self.groupBox_sliceThickness)
        self.label_cenghou2 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_pushButton_2 = QtGui.QPushButton(self.groupBox_sliceThickness)
        self.sliceThickness_label_bc2 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_bc2 = QtGui.QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_shice2 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_shice2 = QtGui.QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_error2 = QtGui.QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_error2 = QtGui.QLineEdit(self.groupBox_sliceThickness)
        self.groupBox_sliceThickness.setObjectName(_fromUtf8("groupBox_sliceThickness"))
        self.gridLayout_sliceThickness.setObjectName(_fromUtf8("gridLayout_sliceThickness"))
        self.sliceThickness_pushButton_8.setObjectName(_fromUtf8("sliceThickness_pushButton_8"))
        self.sliceThickness_label_bc8.setObjectName(_fromUtf8("sliceThickness_label_bc8"))
        self.sliceThickness_lineEdit_bc8.setObjectName(_fromUtf8("sliceThickness_lineEdit_bc8"))
        self.sliceThickness_label_shice8.setObjectName(_fromUtf8("sliceThickness_label_shice8"))
        self.sliceThickness_lineEdit_shice8.setObjectName(_fromUtf8("sliceThickness_lineEdit_shice8"))
        self.sliceThickness_label_error8.setObjectName(_fromUtf8("sliceThickness_label_error8"))
        self.sliceThickness_lineEdit_error8.setObjectName(_fromUtf8("sliceThickness_lineEdit_error8"))
        self.sliceThickness_pushButton_28.setObjectName(_fromUtf8("sliceThickness_pushButton_28"))
        self.sliceThickness_label_bc28.setObjectName(_fromUtf8("sliceThickness_label_bc28"))
        self.sliceThickness_lineEdit_bc28.setObjectName(_fromUtf8("sliceThickness_lineEdit_bc28"))
        self.sliceThickness_label_shice28.setObjectName(_fromUtf8("sliceThickness_label_shice28"))
        self.sliceThickness_lineEdit_shice28.setObjectName(_fromUtf8("sliceThickness_lineEdit_shice28"))
        self.sliceThickness_label_error28.setObjectName(_fromUtf8("sliceThickness_label_error28"))
        self.sliceThickness_lineEdit_error28.setObjectName(_fromUtf8("sliceThickness_lineEdit_error28"))
        self.sliceThickness_pushButton_2.setObjectName(_fromUtf8("sliceThickness_pushButton_2"))
        self.sliceThickness_label_bc2.setObjectName(_fromUtf8("sliceThickness_label_bc2"))
        self.sliceThickness_lineEdit_bc2.setObjectName(_fromUtf8("sliceThickness_lineEdit_bc2"))
        self.sliceThickness_label_shice2.setObjectName(_fromUtf8("sliceThickness_label_shice2"))
        self.sliceThickness_lineEdit_shice2.setObjectName(_fromUtf8("sliceThickness_lineEdit_shice2"))
        self.sliceThickness_label_error2.setObjectName(_fromUtf8("sliceThickness_label_error2"))
        self.sliceThickness_lineEdit_error2.setObjectName(_fromUtf8("sliceThickness_lineEdit_error2"))

        self.sliceThickness_pushButton_all = QtGui.QPushButton(self.groupBox_sliceThickness)
        self.sliceThickness_pushButton_all.setObjectName(_fromUtf8("sliceThickness_pushButton_all"))
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_pushButton_8, 0, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_bc8, 1, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_bc8, 1, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_shice8, 2, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_shice8, 2, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_error8, 3, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_error8, 3, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_pushButton_28, 4, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_bc28, 5, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_bc28, 5, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_shice28, 6, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_shice28, 6, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_error28, 7, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_error28, 7, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_pushButton_2, 8, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_bc2, 9, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_bc2, 9, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_shice2, 10, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_shice2, 10, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_error2, 11, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_error2, 11, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_pushButton_all, 12, 0, 1, 4)
        #5、空间分辨率指标计算
        self.groupBox_resolution = QtGui.QGroupBox(self.widget)
        self.gridLayout_resolution = QtGui.QGridLayout(self.groupBox_resolution)
        self.resolution_label_x = QtGui.QLabel(self.groupBox_resolution)
        self.resolution_label_changgui = QtGui.QLabel(self.groupBox_resolution)
        self.resolution_label_gaofenbian = QtGui.QLabel(self.groupBox_resolution)
        self.resolution_label_y = QtGui.QLabel(self.groupBox_resolution)
        self.resolution_lineEdit_y = QtGui.QLineEdit(self.groupBox_resolution)
        self.resolution_lineEdit_x = QtGui.QLineEdit(self.groupBox_resolution)
        self.resolution_pushButton_1 = QtGui.QPushButton(self.groupBox_resolution)
        self.resolution_pushButton_2 = QtGui.QPushButton(self.groupBox_resolution)
        
        self.resolution_checkbox1 = QtGui.QCheckBox(self.groupBox_resolution)
        self.resolution_label_show1 = QtGui.QLabel(self.groupBox_resolution)
        self.resolution_checkbox2 = QtGui.QCheckBox(self.groupBox_resolution)
        self.resolution_label_show2 = QtGui.QLabel(self.groupBox_resolution)
        
        self.resolution_lineEdit_C_10 = QtGui.QLineEdit(self.groupBox_resolution)
        self.resolution_lineEdit_G_10 = QtGui.QLineEdit(self.groupBox_resolution)
        self.resolution_label_C_10 = QtGui.QLabel(self.groupBox_resolution)
        self.resolution_label_G_10 = QtGui.QLabel(self.groupBox_resolution)
        self.resolution_label_G_50 = QtGui.QLabel(self.groupBox_resolution)
        self.resolution_label_C_50 = QtGui.QLabel(self.groupBox_resolution)
        self.resolution_lineEdit_C_50 = QtGui.QLineEdit(self.groupBox_resolution)
        self.resolution_lineEdit_G_50 = QtGui.QLineEdit(self.groupBox_resolution)
        self.groupBox_resolution.setObjectName(_fromUtf8("groupBox_resolution"))
        self.gridLayout_resolution.setObjectName(_fromUtf8("gridLayout_resolution"))
        self.resolution_label_x.setObjectName(_fromUtf8("resolution_label_x"))
        self.resolution_label_changgui.setObjectName(_fromUtf8("resolution_label_changgui"))
        self.resolution_label_gaofenbian.setObjectName(_fromUtf8("resolution_label_gaofenbian"))
        self.resolution_label_y.setObjectName(_fromUtf8("resolution_label_y"))
        self.resolution_lineEdit_y.setObjectName(_fromUtf8("resolution_lineEdit_y"))
        self.resolution_lineEdit_x.setObjectName(_fromUtf8("resolution_lineEdit_x"))
        self.resolution_pushButton_1.setObjectName(_fromUtf8("resolution_pushButton_1"))
        self.resolution_pushButton_2.setObjectName(_fromUtf8("resolution_pushButton_2"))
        self.resolution_lineEdit_C_10.setObjectName(_fromUtf8("resolution_lineEdit_C_10"))
        self.resolution_lineEdit_G_10.setObjectName(_fromUtf8("resolution_lineEdit_G_10"))
        self.resolution_label_C_10.setObjectName(_fromUtf8("resolution_label_C_10"))
        self.resolution_label_G_10.setObjectName(_fromUtf8("resolution_label_G_10"))
        self.resolution_label_G_50.setObjectName(_fromUtf8("resolution_label_G_50"))
        self.resolution_label_C_50.setObjectName(_fromUtf8("resolution_label_C_50"))
        self.resolution_lineEdit_C_50.setObjectName(_fromUtf8("resolution_lineEdit_C_50"))
        self.resolution_lineEdit_G_50.setObjectName(_fromUtf8("resolution_lineEdit_G_50"))
        self.resolution_pushButton_all = QtGui.QPushButton(self.groupBox_resolution)
        self.resolution_pushButton_all.setObjectName(_fromUtf8("resolution_pushButton_all"))
        self.gridLayout_resolution.addWidget(self.resolution_label_x, 0, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_changgui, 1, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_gaofenbian, 1, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_y, 0, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_y, 0, 3, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_x, 0, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_pushButton_1, 1, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_pushButton_2, 1, 3, 1, 1)


        self.gridLayout_resolution.addWidget(self.resolution_checkbox1, 2, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_show1, 2, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_checkbox2, 2, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_show2, 2, 3, 1, 1)
        
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_C_10, 3, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_G_10, 3, 3, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_C_10, 3, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_G_10, 3, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_G_50, 4, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_C_50, 4, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_C_50, 4, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_G_50, 4, 3, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_pushButton_all, 5, 0, 1, 4)

        
        #6、剂量指数
        self.groupBox_5 = QtGui.QGroupBox(self.widget)
        self.gridLayout_7 = QtGui.QGridLayout(self.groupBox_5)
        self.pushButton_head = QtGui.QPushButton(self.groupBox_5)
        self.label_head_ceter = QtGui.QLabel(self.groupBox_5)
        self.label_head = QtGui.QLabel(self.groupBox_5)
        self.label_body = QtGui.QLabel(self.groupBox_5)
        self.label_head_0 = QtGui.QLabel(self.groupBox_5)
        self.label_head_3 = QtGui.QLabel(self.groupBox_5)
        self.label_head_6 = QtGui.QLabel(self.groupBox_5)
        self.label_head_9 = QtGui.QLabel(self.groupBox_5)
        self.label_head_CTDIw = QtGui.QLabel(self.groupBox_5)
        self.lineEdit_head_center = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_head_0 = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_head_3 = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_head_6 = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_head_9 = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_head_CTDIw = QtGui.QLineEdit(self.groupBox_5)
        self.pushButton_body = QtGui.QPushButton(self.groupBox_5)
        self.label_body_ceter = QtGui.QLabel(self.groupBox_5)
        self.label_body_0 = QtGui.QLabel(self.groupBox_5)
        self.label_body_3 = QtGui.QLabel(self.groupBox_5)
        self.label_body_6 = QtGui.QLabel(self.groupBox_5)
        self.label_body_9 = QtGui.QLabel(self.groupBox_5)
        self.label_body_CTDIw = QtGui.QLabel(self.groupBox_5)
        self.lineEdit_body_center = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_body_0 = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_body_3 = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_body_6 = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_body_9 = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_body_CTDIw = QtGui.QLineEdit(self.groupBox_5)
        self.groupBox_5.setObjectName(_fromUtf8("groupBox_5"))
        self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
        self.pushButton_head.setObjectName(_fromUtf8("pushButton_head"))
        self.label_head_ceter.setObjectName(_fromUtf8("label_head_ceter"))
        self.label_head.setObjectName(_fromUtf8("label_head"))
        self.label_body.setObjectName(_fromUtf8("label_body"))
        self.label_head_0.setObjectName(_fromUtf8("label_head_0"))
        self.label_head_3.setObjectName(_fromUtf8("label_head_3"))
        self.label_head_6.setObjectName(_fromUtf8("label_head_6"))
        self.label_head_9.setObjectName(_fromUtf8("label_head_9"))
        self.label_head_CTDIw.setObjectName(_fromUtf8("label_head_CTDIw"))
        self.lineEdit_head_center.setObjectName(_fromUtf8("lineEdit_head_center"))
        self.lineEdit_head_0.setObjectName(_fromUtf8("lineEdit_head_0"))
        self.lineEdit_head_3.setObjectName(_fromUtf8("lineEdit_head_3"))
        self.lineEdit_head_6.setObjectName(_fromUtf8("lineEdit_head_6"))
        self.lineEdit_head_9.setObjectName(_fromUtf8("lineEdit_head_9"))
        self.lineEdit_head_CTDIw.setObjectName(_fromUtf8("lineEdit_head_CTDIw"))
        self.pushButton_body.setObjectName(_fromUtf8("pushButton_body"))
        self.label_body_ceter.setObjectName(_fromUtf8("label_body_ceter"))
        self.label_body_0.setObjectName(_fromUtf8("label_body_0"))
        self.label_body_3.setObjectName(_fromUtf8("label_body_3"))
        self.label_body_6.setObjectName(_fromUtf8("label_body_6"))
        self.label_body_9.setObjectName(_fromUtf8("label_body_9"))
        self.label_body_CTDIw.setObjectName(_fromUtf8("label_body_CTDIw"))
        self.lineEdit_body_center.setObjectName(_fromUtf8("lineEdit_body_center"))
        self.lineEdit_body_0.setObjectName(_fromUtf8("lineEdit_body_0"))
        self.lineEdit_body_3.setObjectName(_fromUtf8("lineEdit_body_3"))
        self.lineEdit_body_6.setObjectName(_fromUtf8("lineEdit_body_6"))
        self.lineEdit_body_9.setObjectName(_fromUtf8("lineEdit_body_9"))
        self.lineEdit_body_CTDIw.setObjectName(_fromUtf8("lineEdit_head_CTDIw"))
        self.gridLayout_7.addWidget(self.pushButton_head, 0, 1, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_ceter, 1, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head, 0, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_body, 0, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_0, 2, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_3, 3, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_6, 4, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_9, 5, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_CTDIw, 6, 0, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_center, 1, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_0, 2, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_3, 3, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_6, 4, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_9, 5, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_CTDIw, 6, 1, 1, 1)
        self.gridLayout_7.addWidget(self.pushButton_body, 0, 3, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_ceter, 1, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_0, 2, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_3, 3, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_6, 4, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_9, 5, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_CTDIw, 6, 2, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_center, 1, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_0, 2, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_3, 3, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_6, 4, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_9, 5, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_CTDIw, 6, 3, 1, 1)
        
        self.gridLayout_all.addWidget(self.groupBox_water, 0, 0, 36, 3)
        self.gridLayout_all.addWidget(self.groupBox_linear, 0, 3, 35, 9)
        self.gridLayout_all.addWidget(self.groupBox_resolution, 35,3,27,9)
        self.gridLayout_all.addWidget(self.groupBox_sliceThickness, 36, 0, 54, 3)
        self.gridLayout_all.addWidget(self.groupBox_5, 62,3,29,9)
        self.gridLayout_all.addWidget(self.groupBox_qingjiao,89,0,4,3)


        
        self.setCentralWidget(scroll_area)
        self.file_dock = QtWidgets.QDockWidget("Files", self)
        
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.file_dock)#左侧文件停靠
##        self.file_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)##
        self.file_list = QtWidgets.QListWidget()
        self.file_list.itemSelectionChanged.connect(self.on_file_item_change)
        self.file_dock.setWidget(self.file_list)
        self.hu_label = QtWidgets.QLabel("No image")
        self.c_label = QtWidgets.QLabel("")
        self.cw_label = QtWidgets.QLabel("")
        self.x_label = QtWidgets.QLabel("")
        self.y_label = QtWidgets.QLabel("")
        self.z_label = QtWidgets.QLabel("")
        self.use_fractional_coordinates = True
        self.ij_label = QtWidgets.QLabel("")
        self._zoom_level = 1
        self.mouse_x = -1
        self.mouse_y = -1
        self.water_pushButton_jz.setText(_translate("groupBox_water", "LCD值", None))
        self.resolution_label_changgui.setText(_translate("groupBox_sliceThickness", "常规算法", None))
        self.resolution_label_gaofenbian.setText(_translate("groupBox_sliceThickness", "高分辨算法", None))
        self.resolution_label_y.setText(_translate("groupBox_sliceThickness", "钨珠中心(y)", None))
        self.resolution_pushButton_2.setText(_translate("groupBox_sliceThickness", "计算", None))
        self.resolution_label_G_50.setText(_translate("groupBox_sliceThickness", "50%", None))
        self.resolution_label_C_50.setText(_translate("groupBox_sliceThickness", "50%", None))
        self.label_head.setText(_translate("groupBox_5", "头部模体", None))
        self.label_body.setText(_translate("groupBox_5", "体部模体", None))
        self.groupBox_water.setTitle(_translate("Viewer", "水模体指标", None))
        self.water_pushButton.setText(_translate("Viewer", "计算", None))
        self.water_label_CT.setText(_translate("Viewer", "CT值(水)", None))
        self.water_label_noise.setText(_translate("Viewer", "噪声(%)", None))
        self.water_label_junyun.setText(_translate("Viewer", "均匀性(HU)", None))
        self.water_label_lcd.setText(_translate("Viewer", "低对比可探测能力", None))
        self.water_label_lcd_2.setText(_translate("Viewer", "2", None))
        self.water_label_lcd_3.setText(_translate("Viewer", "3", None))
        self.water_label_lcd_5.setText(_translate("Viewer", "5", None))
        self.water_label_lcd_7.setText(_translate("Viewer", "7", None))
        self.water_pushButton_all.setText(_translate("Viewer", "统计列表中所有水模体指标", None))
        self.groupBox_linear.setTitle(_translate("Viewer", "CT线性值", None))
        self.linear_pushButton.setText(_translate("Viewer", "计算", None))
        self.linear_pushButton_all.setText(_translate("Viewer", "统计列表中所有测量CT值", None))
        self.linear_label_celiang.setText(_translate("Viewer", "八种材料的测量CT值", None))
        self.linear_label_lilun.setText(_translate("Viewer", "八种材料的理论CT值(用户输入)", None))
        self.pushButton_ctxianxing.setText(_translate("Viewer", "计算CT线性值", None))
        self.linear_label_CT.setText(_translate("Viewer", "CT线性值", None))
        self.groupBox_sliceThickness.setTitle(_translate("Viewer", "层厚偏差(%)", None))
        self.sliceThickness_pushButton_8.setText(_translate("Viewer", "计算", None))
        self.sliceThickness_pushButton_all.setText(_translate("Viewer", "统计列表中所有层厚", None))
        self.sliceThickness_label_bc8.setText(_translate("Viewer", "层厚标称", None))
        self.sliceThickness_label_shice8.setText(_translate("Viewer", "层厚实测值", None))
        self.sliceThickness_label_error8.setText(_translate("Viewer", "相对误差(%)", None))
        self.sliceThickness_pushButton_28.setText(_translate("Viewer", "计算", None))
        self.sliceThickness_label_bc28.setText(_translate("Viewer", "层厚标称", None))
        self.sliceThickness_label_shice28.setText(_translate("Viewer", "层厚实测值", None))
        self.sliceThickness_label_error28.setText(_translate("Viewer", "相对误差(%)", None))
        self.sliceThickness_pushButton_2.setText(_translate("Viewer", "计算", None))
        self.sliceThickness_label_bc2.setText(_translate("Viewer", "层厚标称", None))
        self.sliceThickness_label_shice2.setText(_translate("Viewer", "层厚实测值", None))
        self.sliceThickness_label_error2.setText(_translate("Viewer", "相对误差(%)", None))
        self.groupBox_resolution.setTitle(_translate("Viewer", "高对比分辨力(LP/cm)", None))
        self.resolution_label_x.setText(_translate("Viewer", "钨珠中心(x)", None))
        self.resolution_pushButton_1.setText(_translate("Viewer", "计算", None))
        self.resolution_label_C_10.setText(_translate("Viewer", "10%", None))
        self.resolution_label_G_10.setText(_translate("Viewer", "10%", None))
        self.resolution_label_show1.setText(_translate("Viewer", "显示MTF曲线", None))
        self.resolution_label_show2.setText(_translate("Viewer", "显示MTF曲线", None))
        self.resolution_pushButton_all.setText(_translate("Viewer", "统计列表中所有高对比分辨力值", None))
        self.label_head_ceter.setText(_translate("Viewer", "中心", None))
        self.label_head_CTDIw.setText(_translate("Viewer", "CTDIw", None))
        self.label_head_0.setText(_translate("Viewer", "0", None))
        self.label_head_3.setText(_translate("Viewer", "3", None))
        self.label_head_6.setText(_translate("Viewer", "6", None))
        self.label_head_9.setText(_translate("Viewer", "9", None))
        self.pushButton_head.setText(_translate("Viewer", "计算", None))
        self.lineEdit_head_center.setPlaceholderText(u'(mGy)')
        self.lineEdit_head_CTDIw.setPlaceholderText(u'(mGy)')
        self.lineEdit_head_0.setPlaceholderText(u'(mGy)')
        self.lineEdit_head_3.setPlaceholderText(u'(mGy)')
        self.lineEdit_head_6.setPlaceholderText(u'(mGy)')
        self.lineEdit_head_9.setPlaceholderText(u'(mGy)')
        self.label_body_ceter.setText(_translate("Viewer", "中心", None))
        self.label_body_CTDIw.setText(_translate("Viewer", "CTDIw", None))
        self.label_body_0.setText(_translate("Viewer", "0", None))
        self.label_body_3.setText(_translate("Viewer", "3", None))
        self.label_body_6.setText(_translate("Viewer", "6", None))
        self.label_body_9.setText(_translate("Viewer", "9", None))
        self.pushButton_body.setText(_translate("Viewer", "计算", None))
        self.lineEdit_body_center.setPlaceholderText(u'(mGy)')
        self.lineEdit_body_CTDIw.setPlaceholderText(u'(mGy)')
        self.lineEdit_body_0.setPlaceholderText(u'(mGy)')
        self.lineEdit_body_3.setPlaceholderText(u'(mGy)')
        self.lineEdit_body_6.setPlaceholderText(u'(mGy)')
        self.lineEdit_body_9.setPlaceholderText(u'(mGy)')
        self.statusBar().addPermanentWidget(self.cw_label)
        self.statusBar().addPermanentWidget(self.ij_label)
        self.statusBar().addPermanentWidget(self.x_label)
        self.statusBar().addPermanentWidget(self.y_label)
        self.statusBar().addPermanentWidget(self.z_label)
        self.statusBar().addPermanentWidget(self.hu_label)
        self.water_lineEdit_CT.setEnabled(0)
        self.water_lineEdit_noise.setEnabled(0)
        self.water_lineEdit_junyun.setEnabled(0)
        self.water_lineEdit_jz.setEnabled(0)
        #self.water_lineEdit_g1.setEnabled(0)
        self.sliceThickness_lineEdit_bc8.setEnabled(0)
        self.sliceThickness_lineEdit_shice8.setEnabled(0)
        self.sliceThickness_lineEdit_error8.setEnabled(0)
        self.linear_lineEdit_ctceliang1.setEnabled(0)
        self.linear_lineEdit_ctceliang2.setEnabled(0)
        self.linear_lineEdit_ctceliang3.setEnabled(0)
        self.linear_lineEdit_ctceliang4.setEnabled(0)
        self.linear_lineEdit_ctceliang8.setEnabled(0)
        self.linear_lineEdit_ctceliang5.setEnabled(0)
        self.linear_lineEdit_ctceliang6.setEnabled(0)
        self.linear_lineEdit_ctceliang7.setEnabled(0)
        self.resolution_lineEdit_C_10.setEnabled(0)
        self.linear_lineEdit_CT.setEnabled(0)
        self.water_lineEdit_lcd_2.setEnabled(0)
        self.water_lineEdit_lcd_3.setEnabled(0)
        self.sliceThickness_lineEdit_shice28.setEnabled(0)
        self.sliceThickness_lineEdit_bc28.setEnabled(0)
        self.water_lineEdit_lcd_5.setEnabled(0)
        self.water_lineEdit_lcd_7.setEnabled(0)
        self.resolution_lineEdit_G_10.setEnabled(0)
        self.sliceThickness_lineEdit_error28.setEnabled(0)
        self.sliceThickness_lineEdit_bc2.setEnabled(0)
        self.sliceThickness_lineEdit_shice2.setEnabled(0)
        self.sliceThickness_lineEdit_error2.setEnabled(0)
        self.water_label_lcd_3.setAlignment(QtCore.Qt.AlignRight)
        self.water_label_lcd_7.setAlignment(QtCore.Qt.AlignRight)
        self.water_label_lcd_5.setAlignment(QtCore.Qt.AlignRight)
        self.water_label_lcd_2.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_label_G_10.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_label_C_10.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_label_G_50.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_label_C_50.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_lineEdit_C_50.setEnabled(0)
        self.resolution_lineEdit_G_50.setEnabled(0)
        self.resolution_label_y.setAlignment(QtCore.Qt.AlignRight)
        self.angle = 0
        QtCore.QObject.connect(self.water_pushButton, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_water)
        QtCore.QObject.connect(self.water_pushButton_jz, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_water_jz)
        QtCore.QObject.connect(self.linear_pushButton, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_linear)
        QtCore.QObject.connect(self.sliceThickness_pushButton_8, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_slice8)
        QtCore.QObject.connect(self.sliceThickness_pushButton_28, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_slice28)
        QtCore.QObject.connect(self.sliceThickness_pushButton_2, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_slice2)
        QtCore.QObject.connect(self.water_pushButton_all, QtCore.SIGNAL(_fromUtf8("clicked()")), self.progress_wa)
        QtCore.QObject.connect(self.sliceThickness_pushButton_all, QtCore.SIGNAL(_fromUtf8("clicked()")), self.progress_th)
        QtCore.QObject.connect(self.linear_pushButton_all, QtCore.SIGNAL(_fromUtf8("clicked()")), self.progress_li)
        QtCore.QObject.connect(self.resolution_pushButton_all, QtCore.SIGNAL(_fromUtf8("clicked()")), self.progress_MTF)
        QtCore.QObject.connect(self.pushButton_ctxianxing, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_ctxianxing)
        QtCore.QObject.connect(self.linear_lineEdit_ctlilun1, QtCore.SIGNAL('textChanged(QString)'), self.onChanged0)#CT理论线性值1编辑框内容改变时调用onChanged0
        QtCore.QObject.connect(self.linear_lineEdit_ctlilun2, QtCore.SIGNAL('textChanged(QString)'), self.onChanged1)#CT理论线性值2编辑框内容改变时调用onChanged1
        QtCore.QObject.connect(self.linear_lineEdit_ctlilun3, QtCore.SIGNAL('textChanged(QString)'), self.onChanged2)#CT理论线性值3编辑框内容改变时调用onChanged2
        QtCore.QObject.connect(self.linear_lineEdit_ctlilun4, QtCore.SIGNAL('textChanged(QString)'), self.onChanged3)#CT理论线性值4编辑框内容改变时调用onChanged3
        QtCore.QObject.connect(self.linear_lineEdit_ctlilun5, QtCore.SIGNAL('textChanged(QString)'), self.onChanged4)#CT理论线性值5编辑框内容改变时调用onChanged4
        QtCore.QObject.connect(self.linear_lineEdit_ctlilun6, QtCore.SIGNAL('textChanged(QString)'), self.onChanged5)#CT理论线性值6编辑框内容改变时调用onChanged5
        QtCore.QObject.connect(self.linear_lineEdit_ctlilun7, QtCore.SIGNAL('textChanged(QString)'), self.onChanged6)#CT理论线性值7编辑框内容改变时调用onChanged6
        QtCore.QObject.connect(self.linear_lineEdit_ctlilun8, QtCore.SIGNAL('textChanged(QString)'), self.onChanged7)#CT理论线性值8编辑框内容改变时调用onChanged7
        QtCore.QObject.connect(self.resolution_lineEdit_x, QtCore.SIGNAL('textChanged(QString)'), self.onChanged8)#高对比分辨力钨珠位置x坐标编辑框内容改变时调用onChanged8
        QtCore.QObject.connect(self.resolution_lineEdit_y, QtCore.SIGNAL('textChanged(QString)'), self.onChanged9)
        QtCore.QObject.connect(self.lineEdit_head_center, QtCore.SIGNAL('textChanged(QString)'), self.onChanged11)#头部模体中心编辑框内容改变时调用onChanged11
        QtCore.QObject.connect(self.lineEdit_head_0, QtCore.SIGNAL('textChanged(QString)'), self.onChanged12)#头部模体0编辑框内容改变时调用onChanged12
        QtCore.QObject.connect(self.lineEdit_head_3, QtCore.SIGNAL('textChanged(QString)'), self.onChanged13)#头部模体3编辑框内容改变时调用onChanged13
        QtCore.QObject.connect(self.lineEdit_head_6, QtCore.SIGNAL('textChanged(QString)'), self.onChanged14)#头部模体6编辑框内容改变时调用onChanged14
        QtCore.QObject.connect(self.lineEdit_head_9, QtCore.SIGNAL('textChanged(QString)'), self.onChanged15)#头部模体9编辑框内容改变时调用onChanged15
        QtCore.QObject.connect(self.lineEdit_body_center, QtCore.SIGNAL('textChanged(QString)'), self.onChanged17)#体部模体中心编辑框内容改变时调用onChanged17
        QtCore.QObject.connect(self.lineEdit_body_0, QtCore.SIGNAL('textChanged(QString)'), self.onChanged18)#体部模体0编辑框内容改变时调用onChanged18
        QtCore.QObject.connect(self.lineEdit_body_3, QtCore.SIGNAL('textChanged(QString)'), self.onChanged19)#体部模体3编辑框内容改变时调用onChanged19
        QtCore.QObject.connect(self.lineEdit_body_6, QtCore.SIGNAL('textChanged(QString)'), self.onChanged20)#体部模体6编辑框内容改变时调用onChanged20
        QtCore.QObject.connect(self.lineEdit_body_9, QtCore.SIGNAL('textChanged(QString)'), self.onChanged21)#体部模体9编辑框内容改变时调用onChanged21
        QtCore.QObject.connect(self.resolution_pushButton_1, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_resolution_1)
        QtCore.QObject.connect(self.pushButton_head, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click8)
        QtCore.QObject.connect(self.pushButton_body, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click11)
        QtCore.QObject.connect(self.resolution_pushButton_2, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_resolution_2)
        QtCore.QObject.connect(self.pushButton_angle, QtCore.SIGNAL(_fromUtf8("clicked()")), self.button_click_angle)
        
        QtCore.QObject.connect(self.horizontalSlider, QtCore.SIGNAL(_fromUtf8("sliderMoved(int)")), self.update_mid)
        QtCore.QObject.connect(self.horizontalSlider1, QtCore.SIGNAL(_fromUtf8("sliderMoved(int)")), self.update_length)
        QtCore.QObject.connect(self.horizontalSlider, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.update_mid)
        QtCore.QObject.connect(self.horizontalSlider1, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.update_length)
        QtCore.QObject.connect(self.checkbox1, QtCore.SIGNAL(_fromUtf8('clicked()')), self.toggleckbox1)#点击勾选框1，调用toggleckbox1
        QtCore.QObject.connect(self.checkbox2, QtCore.SIGNAL(_fromUtf8('clicked()')), self.toggleckbox2)#点击勾选框2，调用toggleckbox2
        QtCore.QObject.connect(self.checkbox3, QtCore.SIGNAL(_fromUtf8('clicked()')), self.toggleckbox3)#点击勾选框3，调用toggleckbox3
        QtCore.QObject.connect(self.checkbox4, QtCore.SIGNAL(_fromUtf8('clicked()')), self.toggleckbox4)#点击勾选框4，调用toggleckbox4
        QtCore.QObject.connect(self.checkbox5, QtCore.SIGNAL(_fromUtf8('clicked()')), self.toggleckbox5)#点击勾选框5，调用toggleckbox5
        QtCore.QObject.connect(self.checkbox6, QtCore.SIGNAL(_fromUtf8('clicked()')), self.toggleckbox6)#点击勾选框6，调用toggleckbox6
        QtCore.QObject.connect(self.checkbox7, QtCore.SIGNAL(_fromUtf8('clicked()')), self.toggleckbox7)#点击勾选框7，调用toggleckbox7
        QtCore.QObject.connect(self.checkbox8, QtCore.SIGNAL(_fromUtf8('clicked()')), self.toggleckbox8)#点击勾选框8，调用toggleckbox8
        self.data = np.ndarray((512, 512), np.int8)
        self.update_cw()
        if os.path.isfile(path):
            self.load_files([path])
        elif os.path.isdir(path):
            self.load_files(dicom_files_in_dir(path))
        self.build_menu()       
        
    def toggleckbox1(self):
        if (self.checkbox1.isChecked()):
            self.linear_lineEdit_ctlilun1.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun1.setEnabled(False)
    def toggleckbox2(self):
        if (self.checkbox2.isChecked()):
            self.linear_lineEdit_ctlilun2.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun2.setEnabled(False)
    def toggleckbox3(self):
        if (self.checkbox3.isChecked()):
            self.linear_lineEdit_ctlilun3.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun3.setEnabled(False)
    def toggleckbox4(self):
        if (self.checkbox4.isChecked()):
            self.linear_lineEdit_ctlilun4.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun4.setEnabled(False)
    def toggleckbox5(self):
        if (self.checkbox5.isChecked()):
            self.linear_lineEdit_ctlilun5.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun5.setEnabled(False)
    def toggleckbox6(self):
        if (self.checkbox6.isChecked()):
            self.linear_lineEdit_ctlilun6.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun6.setEnabled(False)
    def toggleckbox7(self):
        if (self.checkbox7.isChecked()):
            self.linear_lineEdit_ctlilun7.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun7.setEnabled(False)
    def toggleckbox8(self):
        if (self.checkbox8.isChecked()):
            self.linear_lineEdit_ctlilun8.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun8.setEnabled(False)

    def update_mid(self):
        self.mid=self.horizontalSlider.value()
        if self.mid<self.length:
            self.pix_label.update_image1()

    def update_length(self):
        self.length=self.horizontalSlider1.value()
        if self.mid<self.length:
            self.pix_label.update_image1()
    def onChanged0(self):
        self.a0=float(self.linear_lineEdit_ctlilun1.text())
        self.array3[0]=1
    def onChanged1(self):
        self.a1= float(self.linear_lineEdit_ctlilun2.text())
        self.array3[1] = 1
    def onChanged2(self):
        self.a2 =float (self.linear_lineEdit_ctlilun3.text())
        self.array3[2] = 1
    def onChanged3(self):
        self.a3 = float (self.linear_lineEdit_ctlilun4.text())
        self.array3[3] = 1
    def onChanged4(self):
        self.a4= float (self.linear_lineEdit_ctlilun5.text())
        self.array3[4] = 1
    def onChanged5(self):
        self.a5= float (self.linear_lineEdit_ctlilun6.text())
        self.array3[5] = 1
    def onChanged6(self):
        self.a6 = float (self.linear_lineEdit_ctlilun7.text())
        self.array3[6] = 1
    def onChanged7(self):
        self.a7 = float (self.linear_lineEdit_ctlilun8.text())
        self.array3[7] = 1
    def onChanged8(self):
        if len(self.resolution_lineEdit_x.text())>2:
            print len(self.resolution_lineEdit_x.text())
            self.a8 = int(self.resolution_lineEdit_x.text())
            print (u'a8此刻输入的内容是：%s' % self.a8)
        else:
            self.a8=0
    def onChanged9(self):
        if len(self.resolution_lineEdit_y.text())>2:
            self.a9 = int(self.resolution_lineEdit_y.text())
            print (u'a9此刻输入的内容是：%s' % self.a9)
        else:
            self.a9=0
    def onChanged11(self):
        self.a71 = int(self.lineEdit_head_center.text())
        print (u'a71此刻输入的内容是：%s' % self.a71)
    def onChanged12(self):
        self.a72 = int(self.lineEdit_head_0.text())
        print (u'a72此刻输入的内容是：%s' % self.a72)
    def onChanged13(self):
        self.a73 = int(self.lineEdit_head_3.text())
        print (u'a73此刻输入的内容是：%s' % self.a73)
    def onChanged14(self):
        self.a74 = int(self.lineEdit_head_6.text())
        print (u'a71此刻输入的内容是：%s' % self.a74)
    def onChanged15(self):
        self.a75 = int(self.lineEdit_head_9.text())
        print (u'a71此刻输入的内容是：%s' % self.a75)
    def onChanged17(self):
        self.a81 = int(self.lineEdit_body_center.text())
        print (u'a81此刻输入的内容是：%s' % self.a81)
    def onChanged18(self):
        self.a82 = int(self.lineEdit_body_0.text())
        print (u'a72此刻输入的内容是：%s' % self.a82)
    def onChanged19(self):
        self.a83 = int(self.lineEdit_body_3.text())
        print (u'a73此刻输入的内容是：%s' % self.a83)
    def onChanged20(self):
        self.a84 = int(self.lineEdit_body_6.text())
        print (u'a71此刻输入的内容是：%s' % self.a84)
    def onChanged21(self):
        self.a85 = int(self.lineEdit_body_9.text())
        print (u'a71此刻输入的内容是：%s' % self.a85)

    def english(self):
        lan=languageok()
        lan.langu(1)
        reply = QMessageBox.information(self,
                                        u'提示',
                                        u'您确定使用英文界面吗？',
                                        QMessageBox.Yes| QMessageBox.No)
    def chinese(self):
        lan=languageok()
        lan.langu(0)
        reply = QMessageBox.information(self,
                                        u'!',
                                        u'Quit and use Chinese？',
                                        QMessageBox.Yes| QMessageBox.No)
    def button_click_angle(self):
       fname = self.file_name
       self.lineEdit_angle.setText(self.strlen(estimate_tilt_angle(fname,IsFile=True)))
       self.angle=estimate_tilt_angle(fname,IsFile=True)
    def button_click_water(self):
        fname =self.file_name
        fname = fname.replace('/', '\\\\')
        ds = dicom.read_file(fname)
        try:
            test = Water_Phantom(fname)
            av, noise = test.water_roi()
            homogeneity = test.homogeneity()
            sz1 = 2
            sz = int(round(sz1 / ds.PixelSpacing[0]))
            a2 = test.calculate_lcd(sz)
            sz1 = 3
            sz = int(round(sz1 / ds.PixelSpacing[0]))
            a3 = test.calculate_lcd(sz)
            sz1 = 5
            sz = int(round(sz1 / ds.PixelSpacing[0]))
            a5 = test.calculate_lcd(sz)
            sz1 = 7
            sz = int(round(sz1 / ds.PixelSpacing[0]))
            a7 = test.calculate_lcd(sz)
            self.av=av
            self.noise=noise*100
            self.homogeneity=homogeneity
            self.lcd=a2
            self.water_lineEdit_CT.setText(self.strlen(av));
            self.water_lineEdit_noise.setText(self.strlen(noise * 100));
            self.water_lineEdit_junyun.setText(self.strlen(homogeneity));
            self.av2 = a2
            self.av3 = a3
            self.av5 = a5
            self.av7 = a7
            self.water_lineEdit_lcd_2.setText(self.strlen(a2));
            self.water_lineEdit_lcd_3.setText(self.strlen(a3));
            self.water_lineEdit_lcd_5.setText(self.strlen(a5));
            self.water_lineEdit_lcd_7.setText(self.strlen(a7));
            self.water_lineEdit_jz.setText('')
            print ("the CT value of water phantom  is : ", av)
            print ("the noise of water phantom is : %s %%" % (noise * 100))
            print ("the homogeneity of water phantom is :", homogeneity)
            print ("the lcd is:", a2)
        except ValueError:
            QMessageBox.information(self,
                u'出错',
                u'当前图像文件无法自动测量水模体相关指标！',
                QMessageBox.Ok)
            return
            
        
    def button_click_water_jz(self):

        #均值和归一化
        checkn = [0,0,0,0]
        if self.water_checkbox1.isChecked():
            checkn[0]=1
        if self.water_checkbox2.isChecked():
            checkn[1]=1
        if self.water_checkbox3.isChecked():
            checkn[2]=1
        if self.water_checkbox4.isChecked():
            checkn[3]=1
        s1=self.av2 * 2 / 10
        s2=self.av3 * 3 / 10
        s3=self.av5 * 5 / 10
        s4=self.av7 * 7 / 10
##        self.av2 = s1
##        self.av3 = s2
##        self.av5 = s3
##        self.av7 = s4
        s5=(s1*checkn[0]+s2*checkn[1]+s3*checkn[2]+s4*checkn[3])/(sum(checkn))
        self.water_lineEdit_jz.setText(self.strlen(s5))
        

        
    def button_click_linear(self):
        fname = self.file_name;
        fname = fname.replace('/', '\\\\')
        test = Linearity_Phantom(fname)
        x = test.get_material_CT_values()
        self.array1 = sorted(x)
##        print(self.array1[0])
        self.linear_lineEdit_ctceliang1.setText("%.1f"%(self.array1[0]));
        self.linear_lineEdit_ctceliang2.setText("%.1f"%(self.array1[1]));
        self.linear_lineEdit_ctceliang3.setText("%.1f"%(self.array1[2]));
        self.linear_lineEdit_ctceliang4.setText("%.1f"%(self.array1[3]));
        self.linear_lineEdit_ctceliang5.setText("%.1f"%(self.array1[4]));
        self.linear_lineEdit_ctceliang6.setText("%.1f"%(self.array1[5]));
        self.linear_lineEdit_ctceliang7.setText("%.1f"%(self.array1[6]));
        self.linear_lineEdit_ctceliang8.setText("%.1f"%(self.array1[7]));
        self.linear_lineEdit_CT.setText('')
        self.CT_err=0
    def button_click_slice8(self):
        fname = self.file_name;
        fname = fname.replace('/', '\\\\')
        dcm= dicom.read_file(fname)                  
        try:
            phantom = CT_phantom(dcm)
            spiralbeads = SpiralBeads(phantom,diameter = 75, pitch = self.pitch,number_beads = self.beadsnum)
        except ValueError:
            QMessageBox.information(self,
                u'出错',
                u'当前图像文件无法自动测量层厚！',
                QMessageBox.Ok)
            return
        except Exception, e:
            print e
            QMessageBox.information(self,
                u'出错',
                u'错误：' + str(e),
                QMessageBox.Ok)
            return
        profile = spiralbeads.get_profile(displayImage=False)
        thickness = spiralbeads.get_lthickness(profile,bc = dcm.SliceThickness)
        if (thickness == None):
            QMessageBox.information(self,
                u'出错',
                u'当前图像未检测到螺旋丝！',
                QMessageBox.Ok)
            if not(self.cb):
                spiralbeads = SpiralBeads(phantom,diameter = self.diam, pitch = self.pitch,number_beads = self.beadsnum)
                indices, profile_segments = spiralbeads.locate_beads()
                for i in range(1):
                    pitch = self.pitch
                    segment = profile_segments[i]
                    thickness = spiralbeads.get_thickness(pitch, segment)
                print "The measured slice thickness is %f"%thickness
            else:return
        self.slice8_shice = thickness
        self.sliceThickness_lineEdit_shice8.setText(self.strlen(thickness))
        data_element = dcm.data_element("SliceThickness")
        self.slice8_biaocheng = data_element.value
        self.sliceThickness_lineEdit_bc8.setText(self.strlen(data_element.value))
        err1=(thickness-data_element.value)/data_element.value*100
        self.slice8_err = err1
        self.sliceThickness_lineEdit_error8.setText(self.strlen(err1))
    def button_click_slice28(self):
        fname = self.file_name;
        fname = fname.replace('/', '\\\\')
        dcm= dicom.read_file(fname)                  
        try:
            phantom = CT_phantom(dcm)
            spiralbeads = SpiralBeads(phantom,diameter = 75, pitch = self.pitch,number_beads = self.beadsnum)
        except ValueError:
            QMessageBox.information(self,
                u'出错',
                u'当前图像文件无法自动测量层厚！',
                QMessageBox.Ok)
            return
        except Exception, e:
            print e
            QMessageBox.information(self,
                u'出错',
                u'错误：' + str(e),
                QMessageBox.Ok)
            return
        profile = spiralbeads.get_profile(displayImage=False)
        thickness = spiralbeads.get_lthickness(profile,bc = dcm.SliceThickness)
        if (thickness == None):
            QMessageBox.information(self,
                u'出错',
                u'当前图像未检测到螺旋丝！',
                QMessageBox.Ok)
            if not(self.cb):
                spiralbeads = SpiralBeads(phantom,diameter = self.diam, pitch = self.pitch,number_beads = self.beadsnum)
                indices, profile_segments = spiralbeads.locate_beads()
                for i in range(1):
                    pitch = self.pitch
                    segment = profile_segments[i]
                    thickness = spiralbeads.get_thickness(pitch, segment)
                print "The measured slice thickness is %f"%thickness
            else:return
        self.slice28_shice = thickness
        self.sliceThickness_lineEdit_shice28.setText(self.strlen(thickness))
        data_element = dcm.data_element("SliceThickness")
        self.slice28_biaocheng = data_element.value
        self.sliceThickness_lineEdit_bc28.setText(self.strlen(data_element.value))
        err1=(thickness-data_element.value)/data_element.value*100
        self.slice28_err = err1
        self.sliceThickness_lineEdit_error28.setText(self.strlen (err1))

    def button_click_slice2(self):
        fname = self.file_name;
        fname = fname.replace('/', '\\\\')
        dcm= dicom.read_file(fname)                  
        try:
            phantom = CT_phantom(dcm)
            spiralbeads = SpiralBeads(phantom,diameter = 75, pitch = self.pitch,number_beads = self.beadsnum)
        except ValueError:
            QMessageBox.information(self,
                u'出错',
                u'当前图像文件无法自动测量层厚！',
                QMessageBox.Ok)
            return
        except Exception, e:
            print e
            QMessageBox.information(self,
                u'出错',
                u'错误：' + str(e),
                QMessageBox.Ok)
            return
        profile = spiralbeads.get_profile(displayImage=False)
        thickness = spiralbeads.get_lthickness(profile,bc = dcm.SliceThickness)
        if (thickness == None):
            QMessageBox.information(self,
                u'出错',
                u'当前图像未检测到螺旋丝！',
                QMessageBox.Ok)
            if not(self.cb):
                spiralbeads = SpiralBeads(phantom,diameter = self.diam, pitch = self.pitch,number_beads = self.beadsnum)
                indices, profile_segments = spiralbeads.locate_beads()
                for i in range(1):
                    pitch = self.pitch
                    segment = profile_segments[i]
                    thickness = spiralbeads.get_thickness(pitch, segment)
                print "The measured slice thickness is %f"%thickness
            else:return
        self.slice2_shice = thickness
        self.sliceThickness_lineEdit_shice2.setText(self.strlen(thickness))
        data_element = dcm.data_element("SliceThickness")
        self.slice2_biaocheng = data_element.value
        self.sliceThickness_lineEdit_bc2.setText(self.strlen(data_element.value))
        err1=(thickness-data_element.value)/data_element.value*100
        self.slice2_err = err1
        self.sliceThickness_lineEdit_error2.setText(self.strlen(err1))

    
        
    def button_click_ctxianxing(self):
        try:
            self.a0=float(self.linear_lineEdit_ctlilun1.text())
            self.a1= float(self.linear_lineEdit_ctlilun2.text())
            self.a2 =float (self.linear_lineEdit_ctlilun3.text())
            self.a3 = float (self.linear_lineEdit_ctlilun4.text())
            self.a4= float (self.linear_lineEdit_ctlilun5.text())
            self.a5= float (self.linear_lineEdit_ctlilun6.text())
            self.a6 = float (self.linear_lineEdit_ctlilun7.text())
            self.a7 = float (self.linear_lineEdit_ctlilun8.text())
        except ValueError:
            QMessageBox.information(self,
                u'出错',
                u'输入的理论CT值有误，请重新输入！',
                QMessageBox.Ok)
            return
        ctarray = [0,0,0,0,0,0,0,0]
        x = []
        y = []
        if self.checkbox1.isChecked():
            ctarray[0] = 1
            x.append(self.array1[0])
            y.append(self.a0)
        if self.checkbox2.isChecked():
            ctarray[1] = 1
            x.append(self.array1[1])
            y.append(self.a1)
        if self.checkbox3.isChecked():
            ctarray[2] = 1
            x.append(self.array1[2])
            y.append(self.a2)
        if self.checkbox4.isChecked():
            ctarray[3] = 1
            x.append(self.array1[3])
            y.append(self.a3)
        if self.checkbox5.isChecked():
            ctarray[4] = 1
            x.append(self.array1[4])
            y.append(self.a4)
        if self.checkbox6.isChecked():
            ctarray[5] = 1
            x.append(self.array1[5])
            y.append(self.a5)
        if self.checkbox7.isChecked():
            ctarray[6] = 1
            x.append(self.array1[6])
            y.append(self.a6)
        if self.checkbox8.isChecked():
            ctarray[7] = 1
            x.append(self.array1[7])
            y.append(self.a7)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        slope = round(slope, 3)
        intercept = round(intercept, 3)
        print slope, intercept
        def f(x, a, b):
            return a + b * x
        max2=0
        for i in range(len(x)):
            if abs(f(x[i],intercept,slope)-y[i])>max2:
                max2=abs(f(x[i],intercept,slope)-y[i])
            self.linear_lineEdit_CT.setText(self.strlen(max2))
            self.CT_err=max2
    def button_click_resolution_1(self):
         fname = self.file_name;
         fname = fname.replace('/', '\\\\')
         windwidth = 9
         if self.a8!=0 and self.place==1:
             try:
                 bead = TungstenBead(
                                     fname,roiwidth=32,
                                     windowwidth=windwidth,
                                     bead_position=[self.a8, self.a9]
                                     )
             except Exception, e:
                 print e
                 QMessageBox.information(self,
                     u'出错',
                     u'错误：' + str(e),
                     QMessageBox.Ok)
                 return
         else:
             try:
                 bead = TungstenBead(
                                     fname, roiwidth=32,
                                     windowwidth=windwidth,
                                     )
             except Exception, e:
                 print e
                 QMessageBox.information(self,
                     u'出错',
                     u'错误：' + str(e),
                     QMessageBox.Ok)
                 return
         self.resolution10 = bead.resolution10 * 10
         self.resolution50 = bead.resolution50 * 10
         self.resolution_lineEdit_C_10.setText(self.strlen(bead.resolution10*10))
         self.resolution_lineEdit_C_50.setText(self.strlen(bead.resolution50*10))
         self.resolution_lineEdit_x.setText("%d"%(bead.beadpositionx))
         self.resolution_lineEdit_y.setText("%d"%(bead.beadpositiony))
         if self.resolution_checkbox1.isChecked():
             MTF_show( bead.resolution10*10, bead.resolution50*10, bead.freq * 10, bead.mtf)
    def button_click_resolution_2(self):
         fname = self.file_name;
         fname = fname.replace('/', '\\\\')
         windwidth = 9
         if self.a8>1 and self.place==1:
             try:
                 bead = TungstenBead(
                                     fname,roiwidth=32,
                                     windowwidth=windwidth,
                                     bead_position=[self.a8, self.a9]
                                     )
             except Exception, e:
                 print e
                 QMessageBox.information(self,
                     u'出错',
                     u'错误：' + str(e),
                     QMessageBox.Ok)
                 return
         else:
             try:
                 bead = TungstenBead(
                                     fname, roiwidth=32,
                                     windowwidth=windwidth,
                                     )
             except Exception, e:
                 print e
                 QMessageBox.information(self,
                     u'出错',
                     u'错误：' + str(e),
                     QMessageBox.Ok)
                 return
         self.resolution10x = bead.resolution10 * 10
         self.resolution50x = bead.resolution50 * 10
         self.resolution_lineEdit_G_10.setText(self.strlen(bead.resolution10*10))
         self.resolution_lineEdit_G_50.setText(self.strlen(bead.resolution50*10))
         self.resolution_lineEdit_x.setText("%d"%(bead.beadpositionx))
         self.resolution_lineEdit_y.setText("%d"%(bead.beadpositiony))
         if self.resolution_checkbox2.isChecked():
             MTF_show( bead.resolution10*10, bead.resolution50*10, bead.freq * 10, bead.mtf)
    def button_click_saveReport(self):
          mgy=self.mgy
          mgy1=self.mgy1
          angle=self.angle
          av2=self.av2
          av3= self.av3
          av5 = self.av5
          av7 = self.av7
          noise=self.noise
          min1=self.min1
          max1=self.max1
          mean1=self.mean1
          lcd=self.lcd
          resolution=self.resolution
          homogeneity=self.homogeneity
          av=self.av
          slice_err8= self.slice8_err
          slice_err28 = self.slice28_err
          slice2_err = self.slice2_err
          resolution10=self.resolution10
          resolution50=self.resolution50
          resolution10x = self.resolution10x
          resolution50x= self.resolution50x
          CT_err=self.CT_err
          styleBoldRed = xlwt.easyxf('font: color-index black, bold off;borders:left THIN,right THIN,top THIN,bottom THIN;align:  vert center, horiz center')  # 设置字体，颜色为红色，加粗
          oldWb = xlrd.open_workbook("template.xls",
                                     formatting_info=True)
          newWb = copy(oldWb)
          sheet1 = newWb.get_sheet(0)
          sheet1.write_merge(4, 4, 2, 2, self.strlen(self.slice8_biaocheng), styleBoldRed)
          sheet1.write_merge(4, 4, 3, 3, self.strlen(self.slice8_shice), styleBoldRed)
          sheet1.write_merge(4, 4, 4, 4, self.strlen(self.slice8_err), styleBoldRed)
          sheet1.write_merge(5, 5, 2, 2, self.strlen(self.slice28_biaocheng), styleBoldRed)
          sheet1.write_merge(5, 5, 3, 3,self.strlen(self.slice28_shice), styleBoldRed)
          sheet1.write_merge(5, 5, 4, 4, self.strlen(self.slice28_err), styleBoldRed)
          sheet1.write_merge(6, 6, 2, 2, self.strlen(self.slice2_biaocheng), styleBoldRed)
          sheet1.write_merge(6, 6, 3, 3, self.strlen(self.slice2_shice), styleBoldRed)
          sheet1.write_merge(6, 6, 4, 4, self.strlen(self.slice2_err), styleBoldRed)
          sheet1.write_merge(7, 7, 3, 3, self.strlen(angle), styleBoldRed)
          sheet1.write_merge(9, 11, 3, 3, self.strlen(av), styleBoldRed)
          sheet1.write_merge(12, 13, 3, 3, self.strlen(homogeneity), styleBoldRed)
          sheet1.write_merge(14, 15, 3, 3, self.strlen(noise), styleBoldRed)
          sheet1.write_merge(16, 18, 3, 3, self.strlen(resolution10), styleBoldRed)
          sheet1.write_merge(19, 21, 3, 3, self.strlen(resolution10x), styleBoldRed)
          sheet1.write_merge(22, 22, 3, 4, "HU="+self.strlen(av2), styleBoldRed)
          sheet1.write_merge(23, 23, 3, 4, "HU="+self.strlen(av3), styleBoldRed)
          sheet1.write_merge(24, 24, 3, 4, "HU="+self.strlen(av5), styleBoldRed)
          sheet1.write_merge(25, 25, 3, 4, "HU="+self.strlen(av7), styleBoldRed)
          sheet1.write_merge(26, 26, 3, 3, "%.1f"%(self.array1[0]), styleBoldRed)
          sheet1.write_merge(27, 27, 3, 3, "%.1f"%(self.array1[1]), styleBoldRed)
          sheet1.write_merge(28, 28, 3, 3, "%.1f"%(self.array1[2]), styleBoldRed)
          sheet1.write_merge(29, 29, 3, 3, "%.1f"%(self.array1[3]), styleBoldRed)
          sheet1.write_merge(30, 30, 3, 3, "%.1f"%(self.array1[4]), styleBoldRed)
          sheet1.write_merge(31, 31, 3, 3, "%.1f"%(self.array1[5]), styleBoldRed)
          sheet1.write_merge(32, 32, 3, 3, "%.1f"%(self.array1[6]), styleBoldRed)
          sheet1.write_merge(33, 33, 3, 3, "%.1f"%(self.array1[7]), styleBoldRed)
          file_name = QtWidgets.QFileDialog.getSaveFileName(
              self,
              "Save file",
              os.path.expanduser(_fromUtf8(u'~/CT检测结果')),
              "xls (*.xls)"
          )
          if file_name:
              newWb.save(file_name)
          newWb.save("0.xls")
    def button_click8(self):
        s1=(self.a72 + self.a73 + self.a74 + self.a75) / 6
        s2=self.a71 / 3 +s1
        self.mgy=s2
        self.lineEdit_head_CTDIw.setText(self.strlen(s2))
    def button_click11(self):
        s1 = (self.a82 + self.a83 + self.a84 + self.a85) / 6
        s2 = self.a81 / 3 + s1
        self.mgy1 = s2
        self.lineEdit_body_CTDIw.setText(self.strlen(s2))
    def open_directory(self):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)
        dialog.setViewMode(QtWidgets.QFileDialog.List)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            directory = dialog.selectedFiles()[0]
            print 2
            self.load_files(dicom_files_in_dir(directory))
            print 1
    def export_image(self):
        file_name = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save file",
            os.path.expanduser("~/dicom-export.png"),
            "PNG images (*.png)"
        )
        if file_name:
            self.pixmap_label._image.save(file_name)
    def about(self):
        QMessageBox.information(self,
            u'关于',
            u'北京交通大学电子信息工程学院信号与图像处理实验室版权所有',
            QMessageBox.Ok)
    def show_dock(self):
        #self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,self.file_dock)
        self.file_dock.show()
    def build_menu(self):                
        self.file_menu = QtWidgets.QMenu('&File', self)#文件
        self.file_menu.addAction('&Open Directory', self.open_directory, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('&File Dock',self.show_dock,QtCore.Qt.CTRL + QtCore.Qt.Key_L)
       # self.file_menu.addAction('&Setting', self.change_data , QtCore.Qt.CTRL + QtCore.Qt.Key_D)
        self.file_menu.addAction('&Image Info', self.ImageInfo , QtCore.Qt.CTRL + QtCore.Qt.Key_S)#图片信息
        self.file_menu.addAction('&Save Report', self.button_click_saveReport , QtCore.Qt.CTRL + QtCore.Qt.Key_P)#生成报告
        self.file_menu.addAction('&Quit', self.close, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)#退出       
        self.help_menu = QtWidgets.QMenu("&Help", self)
        self.help_menu.addAction('&About', self.about)#关于
        self.menuBar().addMenu(self.file_menu)
        self.menuBar().addMenu(self.help_menu)
    def ImageInfo(self):
        if self.file_name:
            dcm= dicom.read_file(self.file_name)
            print(str(dcm))
        try:
            data_element0 = dcm.data_element("SliceThickness")
            data_element1 = dcm.data_element("Manufacturer")
            data_element2 = dcm.data_element("KVP")
            data_element3 = dcm.data_element("XRayTubeCurrent")
            data_element4 = dcm.data_element("ExposureTime")
            data_element5 = dcm.data_element("ConvolutionKernel")
            QMessageBox.information(self,
                                    u'图像信息',
                                    'Slice Thickness:     %f\nManufacturer:     %s\nKVP:     %s\nX-RayTubeCurrent:     %d\nExposureTime:     %s\nConvolutionKernel:     %s\nPixelSpacing:     %s\nReconstructionDiameter:     %s' %(data_element0.value,data_element1.value,data_element2.value,data_element3.value,data_element4.value,data_element5.value,dcm.PixelSpacing,dcm.ReconstructionDiameter),
                                    QMessageBox.Ok)
        except:
            QMessageBox.information(self,
                                    u'错误',
                                    u'部分信息未找到！' ,
                                    QMessageBox.Ok)
    def show_structure(self):
        if self.file_name:
            f = dicom.read_file(self.file_name)
            print(str(f))
    def toggle_full_screen(self, toggled):
        if toggled:
            self.setWindowState(QtCore.Qt.WindowFullScreen)
        else:
            self.setWindowState(QtCore.Qt.WindowNoState)
    def on_file_item_change(self):
        if not len(self.file_list.selectedItems()):
            self.file_name = None
        else:
            item = self.file_list.selectedItems()[0]
            self.file_name = item.toolTip()
            dcm = dicom.read_file(self.file_name)
            
            self.dcmKVP = dcm.data_element("KVP").value
            
            
            try:
                self.dcmXRayTubeCurrent = dcm.data_element("XRayTubeCurrent").value
                self.dcmExposureTime = dcm.data_element("ExposureTime").value
            except:
                pass
                
    def load_files(self, files):
        self.file_list.clear()
        self.files = files
        for file_name in self.files:
            item = QtWidgets.QListWidgetItem(os.path.basename(file_name))
            item.setToolTip(file_name)
            self.file_list.addItem(item)
        self.file_list.setMinimumWidth(self.file_list.sizeHintForColumn(0) + 20)
        if self.files:
            self.file_name = self.files[0]
    def get_coordinates(self, i, j):
        x = self.image_position[0] + self.pixel_spacing[0] * i
        y = self.image_position[1] + self.pixel_spacing[1] * j
        z = self.image_position[2]
        return x, y, z
    @property
    def mouse_ij(self):
        return self.mouse_y // self.zoom_factor, self.mouse_x // self.zoom_factor
    @property
    def mouse_xyz(self):
        if self.use_fractional_coordinates:
            correction = (self.zoom_factor - 1.) / (
            2. * self.zoom_factor)
            return self.get_coordinates(self.mouse_x / self.zoom_factor - correction,
                                        self.mouse_y / self.zoom_factor - correction)
        else:
            return self.get_coordinates(self.mouse_x // self.zoom_factor, self.mouse_y // self.zoom_factor)
    def update_coordinates(self):
        if self.file:
            x, y, z = self.mouse_xyz
            i, j = self.mouse_ij
            self.z_label.setText("z: %.2f" % z)
            if i >= 0 and j >= 0 and i < self.data.shape[0] and j < self.data.shape[1]:
                self.resolution_lineEdit_x.setText("x: %.2f" % x)
                self.lineEdit_35.setText("y: %.2f" % y)
                self.ij_label.setText("Pos: (%d, %d)" % self.mouse_ij)
                self.hu_label.setText("HU: %d" % int(self.data[i, j]))
                return
            else:
                self.hu_label.setText("HU: ???")
        else:
            self.hu_label.setText("No image")
    def update_cw(self):
        self.update_coordinates()
        pass
    @property
    def file_name(self):
        return self._file_name
    @file_name.setter
    def file_name(self, value):
        try:
            self._file_name = value
            data = DicomData.from_files([self._file_name])
            self.pix_label.data = data
            self.setWindowTitle(_translate("","医用CT成像设备质量检测系统 - ",None) + self._file_name)
            self.place=0
        except BaseException as exc:
            print(exc)
            self.pix_label.data = None
            self.setWindowTitle(_translate("","医用CT成像设备质量检测系统 - No image",None))
##    def closeEvent(self.file_dock,event):
##        print "it's time to close window!!!"
    def center_selected(self, x=0, y=0):
        self.center_x = x
        self.center_y = y
        print(self.center_x, self.center_y,self.pix_label.width(),self.pix_label.height())
        if self.pix_label.height()<self.pix_label.width():
            s = self.pix_label.height()
            self.center_x = int(float(x/s)*510)
            self.center_y = int(float(y/s)*510)
        else:
            s =self.pix_label.width()
            self.center_x = int(float(x/s)*510)
            self.center_y = int(float((y-(self.pix_label.height()-self.pix_label.width())/2)/s)*510)
        
        info = self.center_x, ',', self.center_y
        reply = QMessageBox.information(self,
                                        u'提示',
                                        u'您确定用选取的坐标(%d,%d)作为图像中心吗？'%(self.center_x,self.center_y),
                                        QMessageBox.Yes| QMessageBox.No)
        if (reply == 16384 ):
           self.place=1
           self.resolution_lineEdit_x.setText("%d"%(self.center_x))
           self.resolution_lineEdit_y.setText("%d" % ( self.center_y))
           self.a8=self.center_x
           self.a9=self.center_y
        return
    def change_data(self):
        vbox=QVBoxLayout()#纵向布局
        hbox1=QHBoxLayout()#横向布局
        hbox2=QHBoxLayout()
        hbox3=QHBoxLayout()
        hbox4=QHBoxLayout()
        hbox5=QHBoxLayout()
        self.dialog = QDialog(self)
        self.dialog.resize(200,200)
        self.cBox = QtGui.QCheckBox()
        hbox5.addWidget(self.cBox)
        self.label_wuzhu = QtGui.QLabel(u"只使用螺旋丝测量层厚")
        hbox5.addWidget(self.label_wuzhu)
        self.selectBtn =  QtGui.QPushButton(u"选择文件")
        self.dialog.setWindowTitle("SETTING")
        
        self.label_dia = QtGui.QLabel(u"直径")
        self.lineEdit_dia = QtGui.QLineEdit()
        hbox1.addWidget(self.label_dia)
        hbox1.addWidget(self.lineEdit_dia)
        self.label_pit = QtGui.QLabel(u"螺距")
        self.lineEdit_pit = QtGui.QLineEdit()
        hbox2.addWidget(self.label_pit)
        hbox2.addWidget(self.lineEdit_pit)
        self.label_n = QtGui.QLabel(u"数量")
        self.lineEdit_n = QtGui.QLineEdit()
        hbox3.addWidget(self.label_n)
        hbox3.addWidget(self.lineEdit_n)
        self.okBtn =  QtGui.QPushButton(u"应用")
        self.cancelBtn =  QtGui.QPushButton(u"取消")
        self.save0Btn =  QtGui.QPushButton(u"另存为")
        hbox4.addWidget(self.save0Btn)
        hbox4.addWidget(self.okBtn)
        hbox4.addWidget(self.cancelBtn)
        vbox.addLayout(hbox5)
        vbox.addWidget(self.selectBtn)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        self.dialog.setLayout(vbox)
        self.lineEdit_dia.setText(str(self.diam))
        self.lineEdit_pit.setText(str(self.pitch))
        self.lineEdit_n.setText(str(self.beadsnum))
        self.lineEdit_dia.setEnabled(False)
        self.lineEdit_pit.setEnabled(False)
        self.lineEdit_n.setEnabled(False)
        if self.cb == 1:
            self.cBox.toggle()
        self.cBox.stateChanged.connect(self.cBoxchange)
        QtCore.QObject.connect(self.selectBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), self.readData)
        QtCore.QObject.connect(self.save0Btn, QtCore.SIGNAL(_fromUtf8("clicked()")), self.saveData)
        QtCore.QObject.connect(self.okBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), self.ok)
        QtCore.QObject.connect(self.cancelBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), self.cancel)

        self.dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.dialog.exec_()
    def progress_wa(self):
        dia_pro = QDialog(self)
        dia_pro.resize(200,200)
        vbox=QVBoxLayout()#纵向布局
        hbox=QHBoxLayout()#横向布局
        dia_pro.setWindowTitle("WAITING!")
        Progress = QtGui.QProgressBar(dia_pro)
      #  self.yes = QtGui.QPushButton("OK")
        vbox.addWidget(Progress)
       # vbox.addWidget(self.yes)
        dia_pro.setLayout(vbox)
       # self.dia_pro.setWindowModality(QtCore.Qt.ApplicationModal)
        styleBoldRed = xlwt.easyxf('font: color-index black, bold off;borders:left THIN,right THIN,top THIN,bottom THIN;align:  vert center, horiz center')
        wb = xlwt.Workbook(encoding = 'utf-8')
        ws = wb.add_sheet('wy_ws')
        ws.write_merge(0,1,0,0,'分组',styleBoldRed)
        ws.write_merge(0,1,1,1,'名称',styleBoldRed)
        ws.write_merge(0,1,2,2,'CT值（水）',styleBoldRed)
        ws.write_merge(0,1,3,3,'噪声（%）',styleBoldRed)
        ws.write_merge(0,1,4,4,'均匀性（HU）',styleBoldRed)
        ws.write_merge(0,0,5,9,'低对比可探测能力',styleBoldRed)
        ws.write_merge(1,1,5,5,'2',styleBoldRed)
        ws.write_merge(1,1,6,6,'3',styleBoldRed)
        ws.write_merge(1,1,7,7,'5',styleBoldRed)
        ws.write_merge(1,1,8,8,'7',styleBoldRed)
        ws.write_merge(1,1,9,9,'LCD值',styleBoldRed)
        i=2
        self.ccdialog = 1
        self.chooseFile()
        if self.ccdialog == 1:
            return
        files = []
        for n in range(len(self.lname)):
            if self.lname[n] == 1:
                files.append(self.files[n])
        completed = 0
        step = len(files)
        dia_pro.show()
        #self.dia_pro.exec_()
        for fname in files:
            completed += 1./step*100
            Progress.setValue(completed)
            dia_pro.show()
            f = fname#str(fname)
            f=f.split("/")
            q = f[-1].split('\\')
            fname = fname.replace('/', '\\\\')
            ds = dicom.read_file(fname)
            test = Water_Phantom(fname)
            av, noise = test.water_roi()
            homogeneity = test.homogeneity()
            sz1 = 2
            sz = int(round(sz1 / ds.PixelSpacing[0]))
            a2 = test.calculate_lcd(sz)
            sz1 = 3
            sz = int(round(sz1 / ds.PixelSpacing[0]))
            a3 = test.calculate_lcd(sz)
            sz1 = 5
            sz = int(round(sz1 / ds.PixelSpacing[0]))
            a5 = test.calculate_lcd(sz)
            sz1 = 7
            sz = int(round(sz1 / ds.PixelSpacing[0]))
            a7 = test.calculate_lcd(sz)

            
            s1=a2 * 2 / 10
            s2=a3 * 3 / 10
            s3=a5 * 5 / 10
            s4=a7 * 7 / 10
  
            s5=(s1+s2+s3+s4)/4
            
            try:
                ws.write(i,0,q[0])
                ws.write(i,1,q[-1])
                ws.write(i,2,self.strlen(av))
                ws.write(i,3,self.strlen(noise*100))
                ws.write(i,4,self.strlen(homogeneity))
                ws.write(i,5,self.strlen(a2))
                ws.write(i,6,self.strlen(a3))
                ws.write(i,7,self.strlen(a5))
                ws.write(i,8,self.strlen(a7))
                ws.write(i,9,self.strlen(s5))
                i=i+1
            except:
                i=i+1
                continue
        file_name = QtWidgets.QFileDialog.getSaveFileName(
              self,
              "Save file",
              os.path.expanduser(_fromUtf8(u'~/water')),
              "xls (*.xls)"
          )
        if file_name:
            wb.save(file_name)
        dia_pro.close()
    def progress_th(self):
        self.dia_pro = QDialog(self)
        self.dia_pro.resize(200,200)
        vbox=QVBoxLayout()#纵向布局
        hbox=QHBoxLayout()#横向布局
        self.dia_pro.setWindowTitle("WAITING!")
        self.Progress = QtGui.QProgressBar(self.dia_pro)
      #  self.yes = QtGui.QPushButton("OK")
        vbox.addWidget(self.Progress)
       # vbox.addWidget(self.yes)
        self.dia_pro.setLayout(vbox)
       # self.dia_pro.setWindowModality(QtCore.Qt.ApplicationModal)
        
        wb = xlwt.Workbook(encoding = 'utf-8')
        ws = wb.add_sheet('wy_ws1')
        ws.write(0,0,'分组')
        ws.write(0,1,'名称')
        ws.write(0,2,'标称')
        ws.write(0,3,'实测')
        ws.write(0,4,'误差')
        i=1
        self.ccdialog = 1
        self.chooseFile()
        if self.ccdialog == 1:
            return
        files = []
        for n in range(len(self.lname)):
            if self.lname[n] == 1:
                files.append(self.files[n])
        self.completed = 0
        step = len(files)
        self.dia_pro.show()
        #self.dia_pro.exec_()
        for fname in files:
            self.completed += 1./step*100
            self.Progress.setValue(self.completed)
            self.dia_pro.show()
            f = fname#str(fname)
            f=f.split("/")
            q = f[-1].split('\\')
            fname = fname.replace('/', '\\\\')
            #print fname
            dcm= dicom.read_file(fname)                  
            try:
                phantom = CT_phantom(dcm)
                spiralbeads = SpiralBeads(phantom,diameter = 75, pitch = self.pitch,number_beads = self.beadsnum)
            except:continue
            profile = spiralbeads.get_profile(displayImage=False)
            thickness = spiralbeads.get_lthickness(profile,bc = dcm.SliceThickness)
            if (thickness == None):              
                try:#not(self.cBox.isChecked()):
                    spiralbeads = SpiralBeads(phantom,diameter = self.diam, pitch = self.pitch,number_beads = self.beadsnum)
                    indices, profile_segments = spiralbeads.locate_beads()
                    
                    pitch = self.pitch
                    segment = profile_segments[0]
                    thickness = spiralbeads.get_thickness(pitch, segment)
                    #print "The measured slice thickness is %f"%thickness
                except:continue
            if thickness:
                err=(thickness-float(dcm.SliceThickness))/float(dcm.SliceThickness)*100
                ws.write(i,0,q[0])
                ws.write(i,1,q[-1])
                ws.write(i,2,dcm.SliceThickness)
                ws.write(i,3,self.strlen(thickness))
                ws.write(i,4,self.strlen(err)+"%")
                i=i+1
        file_name = QtWidgets.QFileDialog.getSaveFileName(
              self,
              "Save file",
              os.path.expanduser(_fromUtf8(u'~/thickness')),
              "xls (*.xls)"
          )
        if file_name:
            wb.save(file_name)
        self.dia_pro.close()

        
    def progress_li(self):
        dia_pro = QDialog(self)
        dia_pro.resize(200,200)
        vbox=QVBoxLayout()#纵向布局
        hbox=QHBoxLayout()#横向布局
        dia_pro.setWindowTitle("WAITING!")
        Progress = QtGui.QProgressBar(dia_pro)
      #  self.yes = QtGui.QPushButton("OK")
        vbox.addWidget(Progress)
       # vbox.addWidget(self.yes)
        dia_pro.setLayout(vbox)
       # self.dia_pro.setWindowModality(QtCore.Qt.ApplicationModal)
        styleBoldRed = xlwt.easyxf('font: color-index black, bold off;borders:left THIN,right THIN,top THIN,bottom THIN;align:  vert center, horiz center')
        wb = xlwt.Workbook(encoding = 'utf-8')
        ws = wb.add_sheet('wy_ws2')
        ws.write(0,0,'分组')
        ws.write(0,1,'名称')
        ws.write_merge(0,0,2,9,'八种材料的测量CT值',styleBoldRed)
        
        i=1
        self.ccdialog = 1
        self.chooseFile()
        if self.ccdialog == 1:
            return
        files = []
        for n in range(len(self.lname)):
            if self.lname[n] == 1:
                files.append(self.files[n])
        completed = 0
        step = len(files)
        dia_pro.show()
        #self.dia_pro.exec_()
        for fname in files:
            completed += 1./step*100
            Progress.setValue(completed)
            dia_pro.show()
            f = fname#str(fname)
            f=f.split("/")
            q = f[-1].split('\\')
            fname = fname.replace('/', '\\\\')
            #print fname
            test = Linearity_Phantom(fname)
            x = test.get_material_CT_values()
            array1 = sorted(x)
           # print(array1[0])
            try:
                ws.write(i,0,q[0])
                ws.write(i,1,q[-1])
                ws.write(i,2,"%.1f"%(array1[0]));
                ws.write(i,3,"%.1f"%(array1[1]));
                ws.write(i,4,"%.1f"%(array1[2]));
                ws.write(i,5,"%.1f"%(array1[3]));
                ws.write(i,6,"%.1f"%(array1[4]));
                ws.write(i,7,"%.1f"%(array1[5]));
                ws.write(i,8,"%.1f"%(array1[6]));
                ws.write(i,9,"%.1f"%(array1[7]));
                i=i+1
            except:
                i=i+1
                continue
        file_name = QtWidgets.QFileDialog.getSaveFileName(
              self,
              "Save file",
              os.path.expanduser(_fromUtf8(u'~/linear')),
              "xls (*.xls)"
          )
        if file_name:
            wb.save(file_name)
        dia_pro.close()
    def progress_MTF(self):
        print "go"
        dia_pro = QDialog(self)
        dia_pro.resize(200,200)
        vbox=QVBoxLayout()#纵向布局
        hbox=QHBoxLayout()#横向布局
        dia_pro.setWindowTitle("WAITING!")
        Progress = QtGui.QProgressBar(dia_pro)
      #  self.yes = QtGui.QPushButton("OK")
        vbox.addWidget(Progress)
       # vbox.addWidget(self.yes)
        dia_pro.setLayout(vbox)
       # self.dia_pro.setWindowModality(QtCore.Qt.ApplicationModal)
        
        wb = xlwt.Workbook(encoding = 'utf-8')
        ws = wb.add_sheet('wy_ws2')
        ws.write(0,0,'分组')
        ws.write(0,1,'名称')
        ws.write(0,2,'10%MTF(Lp/cm)')
        ws.write(0,3,'50%MTF(Lp/cm)')
        
        i=1
        self.ccdialog = 1
        self.chooseFile()
        if self.ccdialog == 1:
            return
        files = []
        for n in range(len(self.lname)):
            if self.lname[n] == 1:
                files.append(self.files[n])
        print files
        #files = self.files
        
        completed = 0
        step = len(files)
        
        dia_pro.show()
        #self.dia_pro.exec_()
        for fname in files:
            
            completed += 1./step*100
            Progress.setValue(completed)
            dia_pro.show()
            f = fname#str(fname)
            f=f.split("/")
            q = f[-1].split('\\')
            fname = fname.replace('/', '\\\\')
            #print fname
            windwidth = 9
            
            try:
                bead = TungstenBead(
                                    fname, roiwidth=32,
                                    windowwidth=windwidth,
                                    )
                ws.write(i,0,q[0])
                ws.write(i,1,q[-1])
                ws.write(i,2,self.strlen(bead.resolution10*10))
                ws.write(i,3,self.strlen(bead.resolution50*10))
                i=i+1
            except:continue
                  
        file_name = QtWidgets.QFileDialog.getSaveFileName(
              self,
              "Save file",
              os.path.expanduser(_fromUtf8(u'~/MTF')),
              "xls (*.xls)"
          )
        if file_name:
            wb.save(file_name)
        dia_pro.close()
    def chooseFile(self):
        dialog = QDialog(self)
        dialog.resize(200,300)
        vBox=QVBoxLayout()#纵向布局
        groupBox = QtGui.QGroupBox()
        groupBox1 = QtGui.QGroupBox()
        vbox = QVBoxLayout(groupBox)#纵向布局
        vbox1 =  QVBoxLayout(groupBox)#纵向布局
        hbox = [0]*len(self.files)
        cBox = [0]*len(self.files)
        label = [0]*len(self.files)
        self.lname  = [0]*len(self.files)
        scroll = QtGui.QScrollArea()
        
            
        for i in range(len(self.files)):
            hbox[i] = QHBoxLayout()#横向布局
            cBox[i] = QtGui.QCheckBox()
            hbox[i].addWidget(cBox[i])
            f = unicode(self.files[i])
            f=f.split("/")
            q = f[-1].split('\\')
            label[i] = QtGui.QLabel(q[-1])
            hbox[i].addWidget(label[i])
            vbox1.addLayout(hbox[i])
            
        groupBox1.setLayout(vbox1)
        scroll.setWidget(groupBox1)
        scroll.setAutoFillBackground(True)
        scroll.setWidgetResizable(True)
        
        self.select = 1  
        selectBtn =  QtGui.QPushButton(u"全选")
        #vbox.addLayout(vbox1)
        vbox.addWidget(scroll)
        vbox.addWidget(selectBtn)
        dialog.setWindowTitle(u"选择列表文件")
        
        label_junyun1 = QtGui.QLabel(u"均匀抽取n层")
        label_junyun2 = QtGui.QLabel(u"   n = ")
        lineEdit_junyun = QtGui.QLineEdit()
        okBtn1 =  QtGui.QPushButton(u"确定")
        hboxj = QHBoxLayout()#横向布局
        hboxj.addWidget(label_junyun1)
        hboxj.addWidget(label_junyun2)
        hboxj.addWidget(lineEdit_junyun)
        hboxj.addWidget(okBtn1)
        label_ge1 = QtGui.QLabel(u"每n层抽取一层")
        label_ge2 = QtGui.QLabel(u" n = ")
        lineEdit_ge = QtGui.QLineEdit()
        okBtn2 =  QtGui.QPushButton(u"确定")
        hboxg = QHBoxLayout()#横向布局
        hboxg.addWidget(label_ge1)
        hboxg.addWidget(label_ge2)
        hboxg.addWidget(lineEdit_ge)
        hboxg.addWidget(okBtn2)
        okBtn =  QtGui.QPushButton(u"确定")
        cancelBtn =  QtGui.QPushButton(u"取消")
        vbox.addLayout(hboxj)
        vbox.addLayout(hboxg)
        hboxs = QHBoxLayout()#横向布局
        hboxs.addWidget(okBtn)
        hboxs.addWidget(cancelBtn)
        vBox.addWidget(groupBox)
        vBox.addLayout(hboxs)
        dialog.setLayout(vBox)
        def cbchange():
            for i in range(len(self.files)):
                if (cBox[i].isChecked()):
                    self.lname[i] = 1
                else:
                    self.lname[i] = 0
            if self.lname == [0]*len(self.files):
                selectBtn.setText(u"全选")
                self.select = 1
            elif self.lname == [1]*len(self.files):
                selectBtn.setText(u"全部取消")
                self.select = 0
            else:pass
        def canc():
            dialog.close()
           # self.ccdialog = 1
        def cBoxall():
            if self.select == 1:
                for i in range (len(self.files)):
                    cBox[i].setCheckState(QtCore.Qt.Checked)#cBox[i].toggle()
                selectBtn.setText(u"全部取消")
                self.select = 0
            else:
                for i in range (len(self.files)):
                    cBox[i].setCheckState(QtCore.Qt.Unchecked)#cBox[i].toggle()
                selectBtn.setText(u"全选")
                self.select = 1
        def togbox():
##            for i in range(len(self.files)):
##                if (cBox[i].isChecked()):
##                    self.lname[i] = 1
##                else:
##                    self.lname[i] = 0
            if self.lname == [0]*len(self.files):
                QMessageBox.information(self,
                u'出错',
                u'请选择文件！',
                QMessageBox.Ok)
                self.ccdialog = 1
            else:
                self.ccdialog = 0
                dialog.close()
                      
        def cBoxj():
            for i in range (len(self.files)):
                cBox[i].setCheckState(QtCore.Qt.Unchecked)
            try:
                nj = int(lineEdit_junyun.text())

                n = 0    #设置n用于防止实际抽取数量大于设定值
                for i in range (0,len(self.files),int(len(self.files)/nj)):
                    n = n+1
                   # print i
                    if n>nj:break
                    cBox[i].toggle()
            except:
                QMessageBox.information(self,
                u'出错',
                u'请输入正确数值！',
                QMessageBox.Ok)
        def cBoxg():
            for i in range (len(self.files)):
                cBox[i].setCheckState(QtCore.Qt.Unchecked)
            try:
                ng = int(lineEdit_ge.text())
                print ng
                for i in range (0,len(self.files),ng):
                    cBox[i].toggle()
            except:
                QMessageBox.information(self,
                u'出错',
                u'请输入正确数值！',
                QMessageBox.Ok)
        for i in range (len(self.files)):
            QtCore.QObject.connect(cBox[i], QtCore.SIGNAL(_fromUtf8('stateChanged(int)')), cbchange)
        QtCore.QObject.connect(cancelBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), canc)
        QtCore.QObject.connect(selectBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), cBoxall)
        QtCore.QObject.connect(okBtn1, QtCore.SIGNAL(_fromUtf8("clicked()")), cBoxj)
        QtCore.QObject.connect(okBtn2, QtCore.SIGNAL(_fromUtf8("clicked()")), cBoxg)
        QtCore.QObject.connect(okBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), togbox)

        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        dialog.exec_()
        
    
    def cBoxchange(self):
        if (self.cBox.isChecked()):
            self.cb = 1
            self.lineEdit_dia.setEnabled(False)
            self.lineEdit_pit.setEnabled(False)
            self.lineEdit_n.setEnabled(False)
        else:
            self.cb = 0
            self.lineEdit_dia.setEnabled(True)
            self.lineEdit_pit.setEnabled(True)
            self.lineEdit_n.setEnabled(True)
    def readData(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        n = unicode(fname)
        if (not (os.path.exists(fname ) and os.path.isfile(fname))):
            print"file does not exist!"
        else:
            f = open(fname,'r')
            for i in f:
                m=i.split()
          #  print m
            f.close()
            self.lineEdit_dia.setText(m[0])
            self.lineEdit_pit.setText(m[1])
            self.lineEdit_n.setText(m[2])
    def saveData(self):
        file_name = QtWidgets.QFileDialog.getSaveFileName(self,
            "Save file",
            os.path.expanduser("~/beadsData") )
        if file_name:
            with open(file_name,'w') as f:
                s=str(self.lineEdit_dia.text())+" "+str(self.lineEdit_pit.text())+" "+str(self.lineEdit_n.text())
              #  print s
                f.write(s)
                f.close()
        
        
    def ok(self):
         print("应用！")
         self.diam = float(self.lineEdit_dia.text())
         self.pitch = float(self.lineEdit_pit.text())
         self.beadsnum = int(self.lineEdit_n.text())
         self.dialog.close()
    def cancel(self):
         print("取消！")
         self.dialog.close()
    def strlen(self,s):
        s = ("{0:.3f}".format(s))
        s1 = float(s)
        #print s1,len(s)
        if abs(s1) <1 :#&& len(s) ==  5:
            return s
        elif abs(s1) < 10:
            s = "%.2f"% s1
        elif abs(s1)<100:
            s = "%.1f"% s1
        else:s = str(int(s1))
        return (s)#"%.3f"% s))
         
