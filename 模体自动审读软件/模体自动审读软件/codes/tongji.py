# -*- coding:utf8 -*-

import os
import random
import matplotlib.pyplot as plt
import xlwt
import numpy as np
import pydicom as dicom
from  thickness import *
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib as mpl

plt.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
##def rename(path):
##    oldpath = path
##    pos1 = path.find('_num_')
##    pos2 = path.find('_',pos1+5)
##    newpath = path[:pos1]+path[pos2:]
##    os.rename(path,newpath)  # 对文件进行重命名
def scatter_hist(x, y, ax, ax_histx, ax_histy,cl = 'blue'):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y,alpha = 0.5)#c=cl

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(0, lim + binwidth, binwidth)#-lim
    ax_histx.hist(x, bins=np.arange(0,int(x.max()+1),1))#range(int(x.max()+1))
    ax_histy.hist(y, bins=np.arange(0,int(y.max()+1),1), orientation='horizontal')#range(int(y.max()+1))   
def Rename(path,flist = []):
    files = os.listdir(path)  # 获取当前目录的所有文件及文件夹
#    print path,files
    
    for f in files:
        if 1:
            file_path = os.path.join(path, f)  # 获取绝对路径
            if os.path.isdir(file_path):  # 判断是否是文件夹
                Rename(file_path,flist)  # 如果是文件夹，就递归调用自己
            else:
                flist.append(file_path)
##                print(file_path)
##                extension_name = os.path.splitext(file_path)  # 将文件的绝对路径中的后缀名分离出来
##                if extension_name[1] == '.dcm':
##                    newname = extension_name[0]+'.dic'
##                    os.rename(file_path,newname)
##        else:
##            continue  # 可能会报错，所以用了try-except,如果要求比较严格，不需要报错，就删除异常处理，自己调试
##    print flist
    return flist
if __name__=="__main__":
    '''
    flist = Rename(u'E://others//肿瘤医院//A',flist=[])
    wb = xlwt.Workbook(encoding = 'utf-8')
    ws = wb.add_sheet('wy_ws1')
    ws.write(0,0,'文件名')
    ws.write(0,1,'标称')
    ws.write(0,2,'原实测')
    ws.write(0,3,'err1')
    ws.write(0,4,'高斯拟合计算值')
    ws.write(0,5,'err2')
    ws.write(0,6,'最大值')
    ws.write(0,7,'面积')
    i=1
##    print flist
    biaocheng = []
    thickness = []
    thickness2 = []
    area = []
    promax = []
    for fname in flist:
        try:
            dcm = dicom.read_file(fname)
            phantom = CT_phantom(dcm)
            spiralbeads = SpiralBeads(phantom, diameter=75, pitch=90,number_beads=180)
            profile = spiralbeads.get_profile(displayImage=False)
            biaocheng.append(dcm.SliceThickness)
            thick = spiralbeads.get_lthickness(profile,bc = dcm.SliceThickness)
            
            
            if spiralbeads.phantom.FUBU and thick:
                print fname,thick
                thickness2.append( spiralbeads.thickness2)
                thickness.append( thick)
                area.append( spiralbeads.area2)
                promax.append( spiralbeads.pro_max)
                ws.write(i,4,thickness2[-1])
                ws.write(i,5,(thickness2[-1]-biaocheng[-1])/biaocheng[-1])
                ws.write(i,6,promax[-1])
                ws.write(i,7,area[-1])
                ws.write(i,0,fname)
                ws.write(i,1,biaocheng[-1])
                ws.write(i,2,thickness[-1])
                ws.write(i,3,(thickness[-1]-biaocheng[-1])/biaocheng[-1])
                i=i+1
            
        except:
            print fname
            biaocheng = biaocheng[:i-1]
            thickness = thickness[:i-1]
            thickness2 = thickness2[:i-1]
            area = area[:i-1]
            promax = promax[:i-1]
            
##        i=i+1
    wb.save("cenghoutongji4.xls")
    '''
    io = "cenghoutongji4.xls"
    data = pd.read_excel(io,sheet_name = "wy_ws1")#keep_default_na = False,skip_blank_lines = True)
    data = data.dropna(axis = 0,how = "any")
    print data.head()
    biaocheng = data[u"标称"]
    
    thickness = data[u"原实测"]
    area = data[u"面积"]
    promax = data[u"最大值"]
    data1 = data.drop(data[data["err1"]>1].index)
    data1 = data1.drop(data1[data1["err2"]>1].index)
    plt.plot(data1["err1"],'r')
    plt.plot(data1["err2"],'b')
##    plt.plot(0,'g')
    plt.show()
    
    fig = plt.figure(figsize=(8, 8))
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    biaocheng = data1[u"标称"]
    x= np.array(biaocheng)
    thickness2 = data1[u"高斯拟合计算值"]
    y = np.array(thickness2)
    scatter_hist(x, y, ax, ax_histx, ax_histy)#,cl = '#00CED1'蓝色
    text = u'标称值与原实测值'
##    fig.text(0.76,0.78,text, size = 9,\
##          color = "b",bbox = dict(facecolor = "b", alpha = 0.2))
    fig.suptitle(text)
    ax.set_xlabel(u"标称")
    ax.set_ylabel(u"实测")
    plt.savefig("biaochengguanxi.jpg")
    plt.show()
    
