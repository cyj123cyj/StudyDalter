import re
import os
import configparser
import scipy
import numpy
import pydicom
import pylab

from scipy.ndimage import label  # 多维图像处理
from scipy.optimize import curve_fit  # 指数幂数函数拟合确定参数
##from scipy.stats import norm#multivariate_normal
from math import floor, sqrt, tan  # 浮点数向上取整，求平方根
from calculation_logic.linear import find_rod_locations
from calculation_logic.resolution import roi_generator
from utils.util import find_CT_phantom_outer_edge, canny, find_edge_new, find_CT_phantom_outer_boundary

# 读取配置文件
curpath = os.path.dirname(os.path.realpath(__file__))
cfgpath = os.path.join(curpath, "../thick.ini")
print('ini文件的路径:', cfgpath)  # ini文件的路径
conf = configparser.ConfigParser()  # 创建管理对象
conf.read(cfgpath, encoding="utf-8-sig")  # 读ini文件
sections = conf.sections()  # 获取所有的section
# print('ini文件的section：',sections)  #返回list

theta_method = conf.items('thickness_theta')  # 某section中的内容
# print('thickness_theta部分：',theta_method)  #list里面对象是元组

number_samples = int(theta_method[0][1])  # 25000,圆环上取样点数
diameter = float(theta_method[1][1])  # 75，钨珠所在圆环的直径（mm）
pitch = float(theta_method[2][1])  # 90，螺距
number_beads = int(theta_method[3][1])  # 180，一个螺距上的珠子数


def alphanum_key(s):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]


def Gauss(x, a, x0, sigma, b):  # 数据拟合所用高斯函数公式，a指高斯曲线的峰值，x0为其对应的横坐标，sigma为标准差，b为背景值
    return a * numpy.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b


def fit_Gauss(x, y):
    mn = x[numpy.argmax(y)]  # max(y)所对应的横坐标x
    sig = sqrt(sum(y * (x - mn) ** 2) / sum(y))
    if sig > 0.5:
        sig = 0.5  # 标准差
    bg = min(y)  # y的最小值
    try:
        popt, pcov = curve_fit(Gauss, x, y, p0=[max(y), mn, sig, bg])  # 高斯拟合，拟合序列：x，y；Gauss参数：p0
        print("mn=")
        print(mn, numpy.argmax(y), y, sig)  # 高斯函数公式的参数
    except RuntimeError:
        return None
    return popt, pcov  # Gauss公式所需参数


#####横截面的层厚计算 Transverse plane
class SpiralBeads:
    # phantom geometry
    # key: value = spiral bead pitch: number of beads for a full 2pi circle
    # the sprial bead pitch is in millimeter.

    # modified the parameter list so that only one group of beads
    #   is used for slice thickness calculation
    ##    def __init__(self, phantom, type_A=False, **kwargs):
    def __init__(self, phantom,  # 待计算的综合模体
                 diameter=166.3,  # 设置：钨珠所在圆环的直径（mm）
                 pitch=90,  # 设置：螺距
                 number_beads=180,  # 设置：一个螺距上的珠子数
                 interval=75,  # 设置：两钨丝间平行距离（mm）
                 dip=26.56,  # 设置：钨丝与水平方向的夹角
                 **kwargs):
        self.phantom = phantom
        self.sliceThickness = self.phantom.dicom.SliceThickness  # 层厚标称值
        print('层厚标称值 = ', self.sliceThickness)
        self.number_samples = number_samples  # 5000*5  #圆环上的采样点个数

        self.rou = floor(diameter / 2.0 / self.phantom.dicom.PixelSpacing[0])  # 半径/像素间距，由物理尺寸计算像素阵尺寸
        self.pitch = pitch

        self.phantom_high = 94  # 冠状面综合模94mm
        self.whole_high = 198  # 水模加综合模
        self.dis_sagittal = 25
        self.dip = dip * scipy.pi / 180
        self.interval = interval / 2.0 / self.phantom.dicom.PixelSpacing[0]  # 半径/像素间距，由物理尺寸计算像素阵尺寸
        self.High = self.phantom_high / self.phantom.dicom.PixelSpacing[0]
        self.whole = self.whole_high / self.phantom.dicom.PixelSpacing[0]
        self.dis_sagi = self.dis_sagittal / self.phantom.dicom.PixelSpacing[0]  # 像素尺寸
        self.move_left = 0  # 记录像素值曲线的平移
        self.move_right = 0
        self.ROI_A = []  # 像素值曲线向两侧展宽的范围

        x0, y0, xs, ys, x1, x2, y1, y2 = find_edge_new(self.phantom.image, self.phantom.dicom.PixelSpacing[0],
                                                       returnWaterEdge=False)
        self.x0 = x0  # 冠状面和矢状面模体中心横坐标
        self.y0 = y0
        self.h2 = y1 - y2
        self.y2 = y2
        self.dis2radRatio = None
        self.profile = self.get_profile()
        self.LabelPos = None  # 存储钨丝位置坐标

    def smooth_curve(self, curve, w=100):  # 平滑像素值曲线
        curve_m = scipy.ndimage.filters.maximum_filter(curve, w)
        curve_mg = scipy.ndimage.filters.gaussian_filter1d(curve_m, w)
        curve_mgm = scipy.ndimage.filters.maximum_filter(curve_mg, w)
        return curve_mg, curve_mgm

    def get_profile_coro_sagi(self, displayImage=False):  ######冠状面、矢状面获得左右两边的钨丝所在的原始像素值曲线
        pa = self.phantom.image  # 模体图像CT值矩阵
        dis = self.interval  # 以像素个数为单位的半径
        space = self.phantom.dicom.PixelSpacing[0]
        # y = []
        # for i in range(0,self.number_samples):
        #     y.append(self.High*i/self.number_samples+self.y0-self.h2)  #钨丝所在位置，取样点纵坐标
        # y = y+self.y0-self.whole
        y = [self.High * i / self.number_samples + self.y2 for i in range(0, self.number_samples)]
        # (0,self.number_samples)
        # y = [self.High*0.1+0.7*self.High * i / self.number_samples + self.y2 for i in range(0, self.number_samples)]
        profile_left_off = []  # 左侧钨丝
        profile_right_off = []
        # offsets = [-8,-6,-4,-2,-1, 0, 1,2,4,6,8]  #圆环半径偏差值
        offsets = [i for i in range(-10, 10, 1)]
        # print(len(y),len(x_left),pa.shape)
        # self.number_samples = int(self.number_samples*0.8)
        for off in offsets:
            x_left = numpy.ones(self.number_samples) * (self.x0 - dis + off)  # 取样点横坐标
            x_right = numpy.ones(self.number_samples) * (self.x0 + dis + off)
            pf_left = scipy.ndimage.map_coordinates(pa, numpy.vstack((y, x_left)),  # 将pa中坐标为(y,x)的像素提取出来
                                                    order=3, mode='wrap')
            pf_right = scipy.ndimage.map_coordinates(pa, numpy.vstack((y, x_right)),  # 将pa中坐标为(y,x)的像素提取出来
                                                     order=3, mode='wrap')
            profile_left_off.append(pf_left)
            profile_right_off.append(pf_right[::-1])  # 右侧像素值曲线反转
        profile_left_off = numpy.array(profile_left_off)
        profile_right_off = numpy.array(profile_right_off)

        profile_left = profile_left_off.max(axis=0)  # 综合三条像素值曲线得到一条，同一位置选择最大像素值
        profile_right = profile_right_off.max(axis=0)
        self.ROI_A = [profile_left_off, profile_right_off]

        if displayImage:
            pylab.figure()
            pylab.imshow(pa, interpolation="nearest", cmap='gray')  # 原始模体图像
            pylab.plot(x_left, y, 'g.', markersize=1, linewidth=1)  # 绿线标出钨丝位置
            pylab.plot(x_right, y, 'g.', markersize=1, linewidth=1)
            pylab.show()
            pylab.figure()

            # pylab.plot(profile_right, 'r')
            for i in profile_left_off:
                pylab.plot(i)
            pylab.show()
            pylab.figure()
            pylab.plot(profile_left, 'g')
            pylab.plot(profile_right, 'r')

            pylab.show()
        return profile_left, profile_right

    def find_x_off(self, inds):  # 原始左右两侧像素值曲线，钨丝位置的索引
        offsets = [-1, 0, 1]
        offsets = [i for i in range(-10, 10, 1)]
        off_left = offsets[
            scipy.stats.mode(self.ROI_A[0][:, min(inds):max(inds) + 1].argmax(axis=0))[0][0]]  # 钨丝所在位置，offsets的众数
        off_right = offsets[scipy.stats.mode(self.ROI_A[1][:, min(inds):max(inds) + 1].argmax(axis=0))[0][0]]
        # print(off_left,off_right)
        return off_left, off_right

    def locate_profile_sagittal(self, displayImage=False):  # 由左右两边的像素值曲线组合定位到钨丝
        pa = self.phantom.image
        d_wire = int(self.dis_sagi * self.number_samples / self.High)  # 每条钨丝间的距离，以取样点数为单位
        profile_left, profile_right = self.get_profile_coro_sagi(displayImage=False)
        profile_left = profile_left[2000:self.number_samples - 2000]
        profile_right = profile_right[2000:self.number_samples - 2000]
        p_l, pl_max = self.smooth_curve(profile_left)
        p_r, pr_max = self.smooth_curve(profile_right)
        tl = (p_l.max() - p_l.min()) * 0.9 + p_l.min()
        tr = (p_r.max() - p_r.min()) * 0.9 + p_r.min()
        indicesl = numpy.where(numpy.logical_and(pl_max == p_l, p_l > tl))[0]
        indicesr = numpy.where(numpy.logical_and(pr_max == p_r, p_r > tr))[0]
        print(indicesl)
        print(indicesr)
        if len(indicesl) > 3 or len(indicesr) > 3:  # 很可能是中心定位有误，使报错'该像素值曲线没有定位到钨丝'
            pl_max = numpy.ones(len(profile_left))
            pr_max = numpy.ones(len(profile_right))
        else:  # 对齐左右的像素值曲线
            indl_c = indicesl[int(len(indicesl) / 2)]
            indr_c = indicesr[int(len(indicesr) / 2)]
            if indl_c > indr_c:
                d = (indl_c - indr_c) % d_wire  # 取余，找出要平移的距离
                pr_max = scipy.ndimage.shift(pr_max, int(d), mode='wrap')
                self.move_right = int(d)
            else:
                d = (indr_c - indl_c) % d_wire  # 取余，找出要平移的距离
                pl_max = scipy.ndimage.shift(pl_max, int(d), mode='wrap')
                self.move_left = int(d)
        profile = numpy.maximum(pl_max, pr_max)
        # profile = pl_max+pr_max
        if displayImage:
            pylab.figure()
            pylab.plot(profile_left, 'g')  # 像素值曲线
            pylab.plot(profile_right, 'b')  # 像素值曲线
            pylab.plot(profile, 'r')  # 像素值曲线
            pylab.show()
            pylab.plot(pl_max, 'g')  # 像素值曲线
            pylab.plot(pr_max, 'b')  # 像素值曲线
            pylab.plot(profile, 'r')  # 像素值曲线
            pylab.show()
        return profile

    def get_thickness_sagittal(self, profile):  ######冠状面的层厚计算 Coronal plane
        DEBUG = False
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        if (profile.max() - profile.mean()) <= 300:  # 设定阈值，判断像素值曲线是否找到了钨丝所在的位置，没有找到，返回None，找到了才继续计算
            print(profile.max() - profile.mean())
            print('该像素值曲线没有定位到钨丝')
            return None

        pro = scipy.ndimage.filters.gaussian_filter1d(profile, 50)
        pro[numpy.where(pro < pro.mean())] = pro.mean()  # 将像素值曲线低于均值的部分设为均值
        if self.sliceThickness < 1:  # 设置不同的阈值
            k = 0.996
        elif self.sliceThickness < 2:
            k = 0.97
        elif self.sliceThickness < 3:
            k = 0.8
        else:
            k = 0.65
        threshold = (pro.max() - pro.min()) * k + pro.min()
        inds = numpy.where(pro > threshold)[0]

        lb, nlb = scipy.ndimage.label(pro > threshold)  # nlb看定位到了几条钨丝
        thickness = len(inds) * self.phantom_high / self.number_samples / nlb
        thickness = thickness / tan(self.dip)

        print('层厚测量值：', thickness)
        if DEBUG:
            pylab.plot(pro)
            pylab.plot(numpy.ones(len(pro)) * threshold, 'r')
            pylab.show()

        inds = inds + 2000
        off_left, off_right = self.find_x_off(inds)  # 找到象素值曲线x的偏移量

        YLabel_left = (self.y2 + (inds) * self.High / self.number_samples).astype(
            numpy.int) - self.move_right * self.High / self.number_samples
        YLabel_right = (2 * (self.y2) + self.High - YLabel_left).astype(
            numpy.int) + self.move_right * self.High / self.number_samples
        XLabel_left = numpy.ones(len(YLabel_left)) * (self.x0 - self.interval + off_left)
        XLabel_right = numpy.ones(len(YLabel_right)) * (self.x0 + self.interval + off_right)
        YLabel = numpy.hstack((YLabel_left, YLabel_right))
        XLabel = numpy.hstack((XLabel_left, XLabel_right))
        self.LabelPos = [XLabel, YLabel]
        if DEBUG:
            pylab.figure()
            pylab.imshow(self.phantom.image, interpolation="nearest", cmap='gray')
            pylab.plot(self.LabelPos[0], self.LabelPos[1], 'r.', markersize=1, linewidth=1)  # 钨丝位置
            pylab.show()

        return thickness  # 层厚测量值

    def locate_profile_coronal(self, displayImage=False):  # 由左右两边的像素值曲线组合定位到钨丝
        pa = self.phantom.image
        profile_left, profile_right = self.get_profile_coro_sagi(displayImage=False)  # displayImage
        profile_left = profile_left[2000:self.number_samples - 2000]
        profile_right = profile_right[2000:self.number_samples - 2000]

        p_l, pl_max = self.smooth_curve(profile_left)
        p_r, pr_max = self.smooth_curve(profile_right)
        tl = (p_l.max() - p_l.min()) * 0.9 + p_l.min()
        tr = (p_r.max() - p_r.min()) * 0.9 + p_r.min()

        indicesl = numpy.where(numpy.logical_and(pl_max == p_l, p_l > tl))[0]
        indicesr = numpy.where(numpy.logical_and(pr_max == p_r, p_r > tr))[0]
        # print(indicesl)
        # print(indicesr)
        print("indices", indicesl, indicesr, pl_max)

        if len(indicesl) > 1 and len(indicesr) > 1:
            # for i in indicesl:
            #     len(p_l[indicesl[i]])
            # print("indices",indicesl,indicesr)
            # indicesl = [numpy.argmax([len(p_l[indicesl[i]]) for i in indicesl])]
            # indicesr = [numpy.argmax([len(p_r[indicesr[i]]) for i in indicesr])]
            indicesl = [numpy.mean(numpy.where(pl_max == max(pl_max)))]
            indicesr = [numpy.mean(numpy.where(pr_max == max(pr_max)))]
        if len(indicesl) > 1 and len(indicesr) == 1:
            d = abs(indicesl - indicesr[0])
            d_ind = numpy.where(d == min(d))  # 本该只有一个峰值，若其中一条像素值曲线有不止一个峰值，找离另一条曲线峰值距离最近的峰值索引
            indicesl = [indicesl[d_ind]]
        if len(indicesl) == 1 and len(indicesr) > 1:
            d = abs(indicesr - indicesl[0])
            d_ind = numpy.where(d == min(d))  # 距离最近的索引
            indicesr = [indicesr[d_ind]]  # 使len(indicesl) == 1 且 len(indicesr) == 1

        if len(indicesl) == 1 and len(indicesr) == 1:
            pl_max = scipy.ndimage.shift(pl_max, int(indicesr[0] - indicesl[0]),
                                         mode='wrap')  # 对齐峰值，将像素值曲线向右回环平移len(pro)/2
            self.move_left = int(indicesr[0] - indicesl[0])
        profile = numpy.maximum(pl_max, pr_max)
        # profile = pl_max+pr_max
        # displayImage = True
        if displayImage:
            pylab.figure()
            pylab.plot(profile_left, 'g')  # 像素值曲线
            pylab.plot(profile_right, 'b')  # 像素值曲线
            pylab.plot(profile, 'r')  # 像素值曲线
            pylab.show()
            pylab.plot(pl_max, 'g')  # 像素值曲线
            pylab.plot(pr_max, 'b')  # 像素值曲线
            pylab.plot(profile, 'r')  # 像素值曲线
            pylab.show()
        # if max(l)>1:
        #     if max(r)>1:
        #         return numpy.array([])
        #     return numpy.array(pr_max)
        # elif max(r)>1:
        #     return numpy.array(pl_max)
        # else:
        #     return numpy.maximum(pl_max, pr_max)
        return profile

    def get_thickness_coronal(self, profile):  ######冠状面的层厚计算 Coronal plane
        DEBUG = False
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        if (profile.max() - profile.mean()) <= 300:  # 350设定阈值，判断像素值曲线是否找到了钨丝所在的位置，没有找到，返回None，找到了才继续计算
            print(profile.max() - profile.mean())
            print('该像素值曲线没有定位到钨丝')
            return None

        pro = scipy.ndimage.filters.gaussian_filter1d(profile, 50)
        pro[numpy.where(pro < pro.mean())] = pro.mean()  # 将像素值曲线低于均值的部分设为均值
        if self.sliceThickness < 1:  # 设置不同的阈值
            k = 0.97
        elif self.sliceThickness < 2:
            k = 0.8
        else:
            k = 0.6
        threshold = (pro.max() - pro.min()) * k + pro.min()
        inds = numpy.where(pro > threshold)[0]
        thickness = len(inds) * self.phantom_high / self.number_samples
        thickness = thickness * tan(self.dip)
        print('层厚测量值：', thickness)
        if DEBUG:
            pylab.plot(pro)
            pylab.plot(numpy.ones(len(pro)) * threshold, 'r')
            pylab.show()

        off_left, off_right = self.find_x_off(inds)

        YLabel_left = (self.y2 + (inds - self.move_left) * self.High / self.number_samples).astype(numpy.int)
        YLabel_right = (2 * (self.y2) + self.High - YLabel_left).astype(numpy.int) - (
            self.move_left) * self.High / self.number_samples

        # y = [self.High * 0.1 + 0.7 * self.High * i / self.number_samples + self.y2 for i in
        #      range(0, self.number_samples)]
        # YLabel_left = (self.High * 0.1 +self.y2 + (inds - self.move_left) *0.7* self.High / self.number_samples).astype(numpy.int)
        # YLabel_right = (2 * (self.y2) + self.High - YLabel_left).astype(
        #     numpy.int) - self.move_left * self.High / self.number_samples

        XLabel_left = numpy.ones(len(YLabel_left)) * (self.x0 - self.interval + off_left)
        XLabel_right = numpy.ones(len(YLabel_right)) * (self.x0 + self.interval + off_right)
        YLabel = numpy.hstack((YLabel_left, YLabel_right))
        XLabel = numpy.hstack((XLabel_left, XLabel_right))
        self.LabelPos = [XLabel, YLabel]

        if DEBUG:
            pylab.figure()
            pylab.imshow(self.phantom.image, interpolation="nearest", cmap='gray')
            pylab.plot(self.LabelPos[0], self.LabelPos[1], 'r.', markersize=1, linewidth=1)  # 钨丝位置
            pylab.show()
        return thickness  # 层厚测量值

    def locate_profile_coronal1(self, displayImage=False):
        pa = self.phantom.image
        profile_left, profile_right = self.get_profile_coro_sagi(displayImage=False)
        profile_right = profile_right[::-1]
        y = [self.High * i / self.number_samples + self.y2 for i in range(0, self.number_samples)]
        x_left = (self.x0 - self.interval)  # 取样点横坐标
        x_right = (self.x0 + self.interval)

        thick_left = (self.get_thickness_coronal1(profile_left))[0]
        inds_left_y = [self.High * i / self.number_samples + y[0] for i in
                       (self.get_thickness_coronal1(profile_left)[1])]
        inds_left_x = [x_left for m in range(len(inds_left_y))]
        thick_right = (self.get_thickness_coronal1(profile_right))[0]
        inds_right_y = [self.High * i / self.number_samples + y[0] for i in
                        (self.get_thickness_coronal1(profile_right)[1])]
        inds_right_x = [x_right for m in range(len(inds_right_y))]

        # 右边
        indsx = list(inds_left_x) + list(inds_right_x)
        indsy = list(inds_left_y) + list(inds_right_y)

        self.LabelPos = [indsx, indsy]
        ##        print(self.LabelPos,len(indsx),len(indsy))
        if self.sliceThickness < 1:  # 设置不同的阈值
            k = 0.4
        elif self.sliceThickness < 2:
            k = 0.625
        else:
            k = 1.1
        if thick_left > 0 and thick_right > 0 and max(thick_left / thick_right, thick_right / thick_left) > 1.5:
            thick = numpy.array([max(thick_left, thick_right)])
        else:
            thick = numpy.array([thick_left, thick_right])
        print("none", thick, numpy.count_nonzero(thick))
        # thickness = ((thick_left+thick_right)*2*0.75 + (thick_up+thick_down)/2*0.83)/ numpy.count_nonzero(thick)#4
        # thickness = ((thick_left + thick_right) * 2 + (thick_up + thick_down) / 2) / numpy.count_nonzero(thick) * k
        thickness = len(indsx) / numpy.count_nonzero(thick) * self.phantom_high / self.number_samples
        thickness = thickness * tan(self.dip)
        if displayImage:  # 圆环上取样点的像素值曲线图
            pylab.plot(profile_left, 'b')
            pylab.plot(profile_right, 'b')
            pylab.show()
        DEBUG = False  # True
        if DEBUG:
            pylab.figure()
            pylab.imshow(self.phantom.image, interpolation="nearest", cmap='gray')
            pylab.plot(self.LabelPos[0], self.LabelPos[1], 'r.', markersize=1, linewidth=1)  # 钨丝位置
            pylab.show()
        ##            print ("the shape of the profile:", profile.shape)
        ##            for i in range(profile.shape[0]):
        ##                pylab.plot(profile[i])
        ##            pylab.show()
        print("thickness = ", thickness)
        return thickness

    def get_thickness_coronal1(self, profile):
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        pro = scipy.ndimage.median_filter(profile, 5)
        mean = pro.mean()  # pro为长度为25000的1维数组
        maxv = pro.max()
        ##        print maxv-mean
        if (maxv - mean) <= 100:  # 20200509,100   300  设定阈值，判断像素值曲线是否找到了钨丝所在的位置，没有找到，返回None，找到了才继续计算
            ##            print (maxv-mean)
            ##            print ('该像素值曲线没有定位到钨丝')
            return [0, []]

        pro[numpy.where(pro < pro.mean())] = pro.mean()
        pro = pro - pro.mean()

        mean = pro.mean()  # pro为长度为25000的1维数组
        maxv = pro.max()
        threshold = (mean + maxv) / 2

        # pylab.plot(pro)
        # pylab.plot([0, len(pro)], [threshold, threshold])
        # pylab.show()
        inds = numpy.where(pro > threshold)[0]
        # pro = scipy.ndimage.shift(pro, int(len(pro) / 2 - numpy.argmax(pro)), mode='wrap')
        # inds = numpy.where(pro > threshold)[0]
        # print(inds)
        kk = numpy.array([inds[i + 1] - inds[i] for i in range(len(inds) - 1)])
        print("inds", inds[0], inds[-1], max(kk), numpy.where(kk > 2))
        if len(numpy.where(kk > 2)[0]) == 1:
            l1, l2 = [0, numpy.where(kk > 2)[0][0]], [numpy.where(kk > 2)[0][0], len(kk)]
            print("inds", l1, l2)
            if len(l1) < len(l2):
                inds = inds[l2]
            else:
                inds = inds[l1]

        if inds[-1] == self.number_samples - 1 or inds[0] == 0:
            # print("inds",inds,kk)
            return [0, []]
        ##        print(inds[-1],inds[0],len(inds))
        length = (len(inds) - 1)  # * spacing

        ##        print(length*scipy.tan(26.57/180*scipy.pi),length/scipy.tan(26.57/180*scipy.pi))

        return length, inds

    def get_profile_transverse(self, displayImage=False):

        pa = self.phantom.image  # pa = self.phantom.dicom.pixel_array  #模体图像像素矩阵
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        r = int(75.6 / spacing / 2)  # 测层厚的边到中心的距离7.5mm
        xr, yr = int(self.phantom.center_x), int(self.phantom.center_y)  # 模体中心

        rod_R = int(numpy.array(find_rod_locations(pa, spacing, returnR=True)).mean() - 7.8 / spacing)  # 7.5

        ##        print(xr,yr,r,rod_R)
        roi = roi_generator(pa.shape, xr, yr, rod_R)  # 实心圆
        roi = pa * roi
        offsets = list(range(-6, 8, 1))  # [-1,0,1]
        ##        profile = []

        pf_left = []
        pf_right = []
        pf_up = []
        pf_down = []

        # 左边
        y = [list(range(yr - r + off, yr + r + off, 1)) for off in offsets]
        x = [[xr - r + off for m in range(2 * r)] for off in offsets]
        pf_left = [scipy.ndimage.map_coordinates(roi, numpy.vstack((y[i], x[i])),  # 将pa中坐标为(y,x)的像素提取出来
                                                 order=3, mode='wrap') for i in range(len(x))]

        pf_left = numpy.array(pf_left)
        off = offsets[scipy.stats.mode(pf_left.argmax(axis=0))[0][0]]
        pf_left = pf_left.max(axis=0)

        ##        print(pf_left)

        thick_left = (self.get_thickness_transverse(pf_left))[0]
        inds_left_y = [i + yr - r + off for i in (self.get_thickness_transverse(pf_left)[1])]
        inds_left_x = [xr - r + off for m in range(len(inds_left_y))]

        # 右边

        y = [list(range(yr - r + off, yr + r + off, 1)) for off in offsets]
        x = [[xr + r + off for m in range(2 * r)] for off in offsets]
        pf_right = [scipy.ndimage.map_coordinates(roi, numpy.vstack((y[i], x[i])),  # 将pa中坐标为(y,x)的像素提取出来
                                                  order=3, mode='wrap') for i in range(len(x))]

        # 存储边上的像素值，右边
        pf_right = numpy.array(pf_right)
        off = offsets[scipy.stats.mode(pf_right.argmax(axis=0))[0][0]]
        ##        print(pf_right.argmax(axis=0))
        pf_right = pf_right.max(axis=0)

        ##        print(len(pf_right),off,len(y))
        thick_right = (self.get_thickness_transverse(pf_right))[0]

        inds_right_y = [i + yr - r + off for i in (self.get_thickness_transverse(pf_right)[1])]
        inds_right_x = [xr + r + off for m in range(len(inds_right_y))]

        # 上边
        x = [list(range(xr - r + off, xr + r + off, 1)) for off in offsets]
        y = [[yr - r + off for m in range(2 * r)] for off in offsets]
        pf_up = [scipy.ndimage.map_coordinates(roi, numpy.vstack((y[i], x[i])),  # 将pa中坐标为(y,x)的像素提取出来
                                               order=3, mode='wrap') for i in range(len(x))]

        pf_up = numpy.array(pf_up)
        off = offsets[scipy.stats.mode(pf_up.argmax(axis=0))[0][0]]
        pf_up = pf_up.max(axis=0)

        thick_up = (self.get_thickness_transverse(pf_up))[0]

        inds_up_x = [i + xr - r + off for i in (self.get_thickness_transverse(pf_up)[1])]
        inds_up_y = [yr - r + off for m in range(len(inds_up_x))]
        # 下边
        x = [list(range(xr - r + off, xr + r + off, 1)) for off in offsets]
        # x = [list(range(xr - r , xr + r , 1)) for off in offsets]
        y = [[yr + r + off for m in range(2 * r)] for off in offsets]

        pf_down = [scipy.ndimage.map_coordinates(roi, numpy.vstack((y[i], x[i])),  # 将pa中坐标为(y,x)的像素提取出来
                                                 order=3, mode='wrap') for i in range(len(x))]

        pf_down = numpy.array(pf_down)
        # pf_down1 = pf_down

        off = offsets[scipy.stats.mode(pf_down.argmax(axis=0))[0][0]]
        pf_down = pf_down.max(axis=0)

        thick_down = (self.get_thickness_transverse(pf_down))[0]

        inds_down_x = [i + xr - r + off for i in (self.get_thickness_transverse(pf_down)[1])]
        inds_down_y = [yr + r + off for m in range(len(inds_down_x))]
        # if 1:
        #     print(pf_down1,numpy.shape(pf_down1))
        #     print("x,y:",x,y)
        #     print("return;",self.get_thickness_transverse(pf_down))
        #     pylab.imshow(roi,cmap="gray")
        #     pylab.scatter(x[2],y[2],color='r')
        #     pylab.show()
        #     y1 = [334 for i in  range(176,347)]
        #     x1 = [i for i in  range(176,347)]
        #     pf = [scipy.ndimage.map_coordinates(roi, numpy.vstack((y1[i], x1[i])),  # 将pa中坐标为(y,x)的像素提取出来
        #                                              order=3, mode='wrap') for i in range(len(x1))]
        #     pylab.plot(pf,'r')
        #     pylab.show()
        indsx = list(inds_left_x) + list(inds_right_x) + list(inds_up_x) + list(inds_down_x)
        indsy = list(inds_left_y) + list(inds_right_y) + list(inds_up_y) + list(inds_down_y)

        self.LabelPos = [indsx, indsy]
        ##        print(self.LabelPos,len(indsx),len(indsy))
        if self.sliceThickness < 1:  # 设置不同的阈值
            k = 0.4
        elif self.sliceThickness < 2:
            k = 0.625
        else:
            k = 1.1
        thick = numpy.array([thick_left, thick_right, thick_up, thick_down])
        print("none", thick, numpy.count_nonzero(thick))
        # thickness = ((thick_left+thick_right)*2*0.75 + (thick_up+thick_down)/2*0.83)/ numpy.count_nonzero(thick)#4
        thickness = ((thick_left + thick_right) * 2 + (thick_up + thick_down) / 2) / numpy.count_nonzero(thick) * k

        if displayImage:  # 圆环上取样点的像素值曲线图
            pylab.plot(pf_left, 'b')
            pylab.plot(pf_right, 'b')
            pylab.plot(pf_up, 'r')
            pylab.plot(pf_down, 'r')
            pylab.show()
        ##            print ("the shape of the profile:", profile.shape)
        ##            for i in range(profile.shape[0]):
        ##                pylab.plot(profile[i])
        ##            pylab.show()
        print("thickness = ")
        print(thickness)
        return thickness

    def get_thickness_transverse(self, profile):
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        pro = scipy.ndimage.median_filter(profile, 5)
        mean = pro.mean()  # pro为长度为25000的1维数组
        maxv = pro.max()
        ##        print maxv-mean
        if (maxv - mean) <= 100:  # 20200509,100   300  设定阈值，判断像素值曲线是否找到了钨丝所在的位置，没有找到，返回None，找到了才继续计算
            ##            print (maxv-mean)
            ##            print ('该像素值曲线没有定位到钨丝')
            return [0, []]

        pro[numpy.where(pro < pro.mean())] = pro.mean()
        mean = pro.mean()  # pro为长度为25000的1维数组
        maxv = pro.max()
        threshold = (mean + maxv) / 2
        inds0 = numpy.where(pro > threshold)[0]
        pro = scipy.ndimage.shift(pro, int(len(pro) / 2 - numpy.argmax(pro)), mode='wrap')
        inds = numpy.where(pro > threshold)[0]
        # print(inds)
        kk = [inds[i + 1] - inds[i] for i in range(len(inds) - 1)]
        if max(kk) > 2:
            # print("inds",inds,kk)
            return [0, []]
        ##        print(inds[-1],inds[0],len(inds))
        length = (len(inds) - 1) * spacing

        ##        print(length*scipy.tan(26.57/180*scipy.pi),length/scipy.tan(26.57/180*scipy.pi))
        ##        pylab.plot(pro)
        ##        pylab.plot([0,len(pro)],[threshold,threshold])
        ##        pylab.show()
        return length, inds0

    # 将极坐标转换为笛卡尔坐标
    # angel：待转化的极坐标的角度数组；rho：极坐标半径；center_coor：极坐标极点；
    def angle2coor(self, angle, rho, center_coor, as_index=False):
        # convert polar angle to cartesian coordinates
        #   This controls where the starting point is in the CT phantom image
        #   currently, the zero angle is located at the right-most point of a circle
        yc, xc = center_coor  # 极点的直角坐标
        x = xc + rho * numpy.cos(angle)  # 将数组angle转化为直角坐标，0角位于圆的最右边
        y = yc + rho * numpy.sin(angle)
        # because the coordinates may be used to do subpixel sampling
        # here, the coordinates can be float values
        # but you have an option to convert them to image pixel indices
        if as_index:  # 若坐标被用作图像像素索引
            return (scipy.uint16(scipy.round_(y, 0)),  # 将浮点型坐标值转化为整型
                    scipy.uint16(scipy.round_(x, 0)))
        else:
            return (y, x)  # 数组angle的直角坐标

    # 得到螺旋珠所在的圆环上的点的角度和像素值曲线
    def get_profile(self, displayImage=False):
        # rename variables for convenience
        pa = self.phantom.dicom.pixel_array  # 模体图像像素矩阵
        rad = self.rou  # 以像素个数为单位的圆环半径
        xr, yr = self.phantom.center_x, self.phantom.center_y  # 模体中心
        # need a profile to get degMax (position of the brightest bead)
        thetaPlt = numpy.linspace(0, 2 * scipy.pi, self.number_samples)  # 在0到2pi内均匀取25000个数值
        profile = []  # 用来存储像素值的空白数组
        # in the beginning, the precision of the model is not good
        # therefore, used to have average over different radii
        # but the current phantom is good enough
        # therefore, no need to do averaging any more

        # to romve possible inaccuracy in bead-mounting
        #    and to romve the air gaps in the profile
        #    using the maximum to replace the mean 06/19/2018
        offsets = [-1, 0, 1]  # 圆环半径偏差值
        for off in offsets:
            y, x = self.angle2coor(thetaPlt, rad + off, (yr, xr), as_index=False)  # 将取样点的极坐标转化为直角坐标
            # scipy provides below function to do interpolation
            # using spline interpolation with the order of 3
            # when an edge is encountered, the "wrap" mode is used
            pf = scipy.ndimage.map_coordinates(pa, numpy.vstack((y, x)),  # 将pa中坐标为(y,x)的像素提取出来
                                               order=3, mode='wrap')
            profile.append(pf)  # 存储圆环上的像素值
        profile = numpy.array(profile)
        if displayImage:  # 圆环上取样点的像素值曲线图
            print("the shape of the profile:", profile.shape)
            for i in range(profile.shape[0]):
                pylab.plot(profile[i])
            pylab.show()
        ##        profile = profile.mean(axis=0)
        profile = profile.max(axis=0)  # 综合三条像素值曲线得到一条，同一位置选择最大像素值
        ##        print "profile shape =", profile.shape

        if displayImage:
            pylab.figure()
            pylab.imshow(pa, interpolation="nearest", cmap='gray')  # 原始模体图像
            pylab.plot(x, y, 'g.', markersize=1, linewidth=1)  # 绿线标出钨珠所在圆环
            pylab.show()
            # print "profile mean = %s"%(profile.mean())
            pylab.plot(profile)  # scipy.ndimage.filters.median_filter(profile,5))  #角度-像素值曲线
            pylab.show()
        return {'theta': thetaPlt, 'profile': profile}

    # 由钨珠分布的角度大小计算层厚测量值
    def get_lthickness(self, profile, bc=0.625):
        DEBUG = False
        pitch = self.pitch  # 螺距
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        # 针对不同的标称值设定中值滤波器的参数和阈值的相对位置
        if bc <= 0.55:
            a = 2.2
            k = 55 / 98.  # 141/223.
        elif bc <= 0.625:
            a = 3.08
            k = 127 / 223.
        elif bc <= 1.1:
            a = 30.8
            k = 55 / 108.
        elif bc <= 1.25:
            a = 66
            k = 55 / 119.
        elif bc <= 2.2:
            a = 118.8
            k = 55 / 112.
        elif bc <= 5:
            a = 132
            k = 55 / 108.
        elif bc <= 5.5:
            a = 162.8
            k = 55 / 113.
        else:
            a = 176
            k = 55 / 115.

        pro = scipy.ndimage.median_filter(profile['profile'], int(a / spacing))  # 中值滤波以平滑profile，参数设置：滤波窗口的像素长（邻域）
        mean = pro.mean()  # pro为长度为25000的1维数组
        maxv = pro.max()
        ##        print maxv-mean
        if (maxv - mean) <= 60:  # 20200509,100     设定阈值，判断像素值曲线是否找到了钨丝所在的位置，没有找到，返回None，找到了才继续计算
            print(maxv - mean)
            print('该像素值曲线没有定位到钨丝')
            return None

        threshold = k * (mean + maxv)  # 像素值阈值，k
        ##        threshold =  (pro.mean()*2 + pro.max())/3
        inds = numpy.where(pro > threshold)[0]  # 筛选出大于阈值的像素，返回其索引
        inds_theta = inds * scipy.pi * 2 / self.number_samples  # 钨丝所在角度值范围

        flag_move1 = 0
        if inds[0] == 0:  # 若像素值曲线刚好是从钨珠所在位置开始截取的
            pro = scipy.ndimage.shift(pro, int(len(pro) / 2), mode='wrap')  # 将像素值曲线向右回环平移len(pro)/2
            inds = numpy.where(pro > threshold)[0]  # 新的索引
            flag_move1 = 1
        tht = profile['theta']

        span = (tht[inds[-1]] - tht[inds[0]]) / scipy.pi * (pitch / 2)  # 由钨珠分布的角度和螺距计算层厚
        thickness = span
        print("高斯拟合前钨珠角度法所得层厚测量值 = %s" % span)
        #        FUBUMOTI = 1
        flag_move2 = 0
        if abs(numpy.argmax(pro) - len(pro) / 2) > len(pro) / 5:  # 尽量平移像素值曲线，使峰值不太偏
            distance = int(len(pro) / 2 - numpy.argmax(pro))
            pro = scipy.ndimage.shift(pro, int(len(pro) / 2 - numpy.argmax(pro)), mode='wrap')
            inds = numpy.where(pro > threshold)[0]  # 新的索引
            flag_move2 = 1
        pro[numpy.where(pro < pro.mean())] = pro.mean()  # 将像素值曲线低于均值的部分设为均值
        # print (pro.mean())

        condition = (pro.max() - pro.min()) * 0.3 + pro.min()
        spread = pro > condition
        lb, nlb = scipy.ndimage.label(spread)
        print('钨丝范围连通域数目:', nlb)
        if nlb > 1:
            print('该像素值曲线没有定位到钨丝')
            return None

        if self.phantom.FUBU:  # 如果是腹模
            ##            width = float(self.number_samples)/250
            ##            sigma = scipy.std(pro)
            popt, pcov = fit_Gauss(tht, pro)  # 像素值曲线高斯拟合所需的参数
            ##            print popt
            pro2 = Gauss(tht, *popt)  # 高斯拟合像素值曲线
            threshold2 = 0.5 * (pro2.max() + pro2.min())  # 拟合曲线的像素值阈值，0.5
            inds2 = numpy.where(pro2 > threshold2)[0]  # 高于阈值的索引

            if flag_move1:  # 判断高斯拟合前像素值曲线是否平移了
                pro_ini = scipy.ndimage.shift(pro2, int(len(pro) / 2), mode='wrap')  # 平移前的像素值曲线
            else:
                pro_ini = pro
            if flag_move2:
                pro_ini = scipy.ndimage.shift(pro_ini, len(pro) - distance, mode='wrap')
            else:
                pro_ini = pro
            inds_2 = numpy.where(pro_ini > threshold2)[0]
            inds_theta = inds_2 * scipy.pi * 2 / self.number_samples  # 钨丝所在角度值范围

            self.thickness2 = (tht[inds2[-1]] - tht[inds2[0]]) / scipy.pi * (pitch / 2)  # 由钨珠分布的角度和螺距计算层厚
            thickness = self.thickness2
            print("高斯拟合后钨珠角度法所得层厚测量值 = %s" % thickness)
            self.area2 = sum((pro2[inds2[0]:inds2[-1]] - pro2.min()) / pro2.max())
            self.pro_max = pro2.max()
            ##            area,err = scipy.integrate.quad(Gauss,tht[inds2[0]] , tht[inds2[-1]],args = popt)#(
            print("area:" + str(self.area2) + "thickness2:" + str(self.thickness2) + "max:" + str(
                pro2.max()))  # 用高斯拟合的计算结果
        if DEBUG:
            pylab.plot(pro, 'b')  # 原像素值曲线
            pylab.plot(pro2, 'g')  # 高斯拟合的像素值曲线
            ##            pylab.plot(profile['profile'])
            # pylab.plot(numpy.ones(len(pro))*mean)  #用直线标示原像素值曲线的均值
            pylab.plot(numpy.ones(len(pro)) * condition, 'y')  # 最大值
            pylab.plot(numpy.ones(len(pro)) * threshold, 'r')  # 像素阈值
            pylab.show()

        xr, yr = self.phantom.center_x, self.phantom.center_y  # 模体中心
        inds_yx = self.angle2coor(inds_theta, self.rou, (yr, xr), as_index=False)  # 钨丝位置
        self.LabelPos = [inds_yx[1], inds_yx[0]]  # x,y
        if DEBUG:
            pylab.figure()
            pylab.imshow(self.phantom.image, interpolation="nearest", cmap='gray')
            pylab.plot(self.LabelPos[0], self.LabelPos[1], 'r.', markersize=1, linewidth=1)  # 钨丝位置
            pylab.show()

        return thickness  # 层厚测量值


SECTION_TYPE = [0, 1]
GEOMETRY = {
    0: [  # diameters
        161,  # mm, the outer diameter
        110,  # mm, the diameter of the circle where the 8 linearity rods locate
        15,  # mm, the linearity rod diameter
        15,  # mm, the MTF wire rod diameter
        3,  # mm, the diameter of the 4 geometric distortion holes

        # distances
        90,  # mm, pitch of the spiral beads
        32,  # mm, the length of the hole modules
        10,  # mm, the depth of the hole modules
        0,  # mm, the distance from the center to the geometric distrotion holes, UNKNOWN
        30,  # mm, the distance from the center to the MTF wire rod center
    ],
    1: [  #
        161,  # 145,
        100,
        15,
        15,
        3,
        #
        70,  # mm, pitch of the spiral beads
        32,  # ?
        10,  # ?
        0,  # UNKNOWN
        25,
    ],
    2: [  #
        161,  # 113, # ?
        90,
        12,
        12,
        3,
        #
        60,  # mm, planned pitch of the spiral beads
        32,  # ?
        10,  # ?
        0,  # UNKNOWN
        0,  # UNKNOWN
    ]
}


class CT_phantom:
    """
    The structure of phantom.
    In this design, there are two phantom sections.
    One is a water phantom, including a cylindrical container with water
        inside and probably shells outside the container
    the other is a comprehensive phantom, including several components
        bead spiral for thickness, square holes for spatial resolution,
        four small cylindrical holes for geometrical distortion,
        eight cylindrical rods for CT number linearity,
        and a tungen wire for spatial resolution
    This class is to identify which phantom section the image is
    and to locate each component in the phantom section
    """

    def __init__(self, dcm_img):
        if type(dcm_img) in [type("string"), type(u"string")]:
            # assume this is a dicom file
            try:
                dcm = pydicom.read_file(dcm_img, force=True)
                self.filename = dcm_img
            except:
                print("Not a dicom file: %s" % dcm_img)
                return
        else:
            dcm = dcm_img
        self.dicom = dcm
        self.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        self.image[self.image > 3000] = self.image.min()  # 去除心脏CT图中右侧字的影响

        if round(dcm.ImageOrientationPatient[0]) == 1 and round(dcm.ImageOrientationPatient[-2]) == 1:  # 横截面
            self.direction = "transverse"
            self.section_type = self.get_section_type()

            if self.section_type == 1:  # 输入横断面为综合模体时，才计算模体中心
                re = find_CT_phantom_outer_edge(self.image, dcm.PixelSpacing[0], return_coors=True)  # 定位模体外边缘
                self.center_x = re[0]  # 模体中心坐标
                self.center_y = re[1]
                self.outer_radius = re[2]  # 模体外边缘半径
                self.outer_coor_xs = re[3]  # 外边界坐标
                self.outer_coor_ys = re[4]

            # find the structure
            # if self.section_type == SECTION_TYPE[1]:
            #     self.determine_size()

        if round(dcm.ImageOrientationPatient[0]) == 1 and round(dcm.ImageOrientationPatient[-1]) == -1:  # 冠状面
            self.direction = "coronal"
            re = find_CT_phantom_outer_boundary(self.image)
            self.yup, self.ylow, self.xleft, self.xright = re

        if round(dcm.ImageOrientationPatient[1]) == 1 and round(dcm.ImageOrientationPatient[-1]) == -1:  # 矢状面
            self.direction = "sagittal"
            re = find_CT_phantom_outer_boundary(self.image)
            self.yup, self.ylow, self.xleft, self.xright = re

    def get_section_type(self):
        """
        determine whether the phantom is the water or the comprehensive section

        since the water section has simpler structure, the edge pixels are less
        therefore, the number of edge pixels is used to tell difference
        """
        edges = canny(self.image, sigma=2.0,
                      low_threshold=50, high_threshold=100)

        if edges.sum() < 6000:
            return SECTION_TYPE[0]  # 0 水模
        else:
            return SECTION_TYPE[1]  # 1 综合模


def determine_size(self):
    self.find_MTF_wire_rod()
    # self.find_rod_locations()


def find_MTF_wire_rod(self):
    """
    the design of phantom can be characterized by the distance between
    the phantom center and the MTF wire rod center
    """
    DEBUG = False
    re = [self.center_x, self.center_y, self.outer_radius,
          self.outer_coor_xs, self.outer_coor_ys]
    xc, yc, r, xe, ye = re

    h, w = self.dicom.pixel_array.shape
    mask = scipy.mgrid[0:h, 0:w]
    # detect in this region to see where the wire is
    detection_dist = 40  # mm to cover the MTF wire rod
    detection_dist /= self.dicom.PixelSpacing[0]
    dist_map = numpy.hypot((mask[0] - yc), (mask[1] - xc))

    detection_zone = dist_map < detection_dist

    # to determine how to smooth the image
    #   with a high SNR, smaller kernel may be used
    std = numpy.std(self.image[numpy.where(detection_zone)])
    try:
        kernel = self.dicom.ConvolutionKernel
    except:
        kernel = None

    if kernel == "BONE" or std > 40:
        sigmaV = 3
    else:
        sigmaV = 1

    # print "using sigma = %s, std = %s"%(sigmaV, self.image[scipy.where(detection_zone)].std())
    edge = canny(self.image, sigma=sigmaV,
                 low_threshold=10, high_threshold=100)
    edge *= detection_zone

    # to find the largest region
    #   which can be assumed to be the wire rod
    lb, nlb = label(edge == 0)
    # print nlb
    if nlb == 1:
        # could not detect the MTF wire rod
        print("Could not detect the MTF wire rod!")
        return
    hist, le = numpy.histogram(lb, bins=range(2, nlb + 2))
    ind = numpy.argsort(hist)[-1] + 2
    rod = lb == ind

    # the distance between the center of the MTF wire and the center of the phantom
    rodyc, rodxc = [numpy.mean(e) for e in numpy.where(rod)]
    dist_cc = numpy.hypot(xc - rodxc, yc - rodyc)
    dist_cc_mm = dist_cc * self.dicom.PixelSpacing[0]
    # print "distance between the MTF rod and the center:", dist_cc_mm

    if DEBUG:
        import pylab
        # pylab.imshow(lb)
        # pylab.show()
        pylab.imshow(rod)
        pylab.show()

    ind = -1
    err = 1000.
    for k in GEOMETRY.keys():
        abs_err = abs(dist_cc_mm - GEOMETRY[k][-1])
        if err > abs_err:
            ind = k
            err = abs_err
    ##        print "geometry type:", ind
    self.geometry = GEOMETRY[ind]


# *** 2022.3新加，适用于改进的D型模体
class thickness_new:
    # 分别在横截面、冠状面和矢状面上计算改进D型模体的层厚
    def __init__(self, phantom,  # 待计算厚度的模体（一层）
                 interval_bottom=86,  # 立方体底部正方形的边长 (mm)
                 interval_height=94,  # 立方体的高 (mm)
                 dip=26.57,  # 立方体表面的钨丝直线与水平方向的夹角 (°)
                 height_synthesis=100,  # 综合模高度 (mm)
                 height_water=104,  # 水模高度 (mm)
                 ):
        self.phantom = phantom
        self.image = phantom.image
        self.sliceThickness = phantom.dicom.SliceThickness  # 层厚标称值
        # print('层厚标称值 = ', self.sliceThickness)

        self.interval_bottom = interval_bottom / phantom.dicom.PixelSpacing[0]  # 物理尺寸->像素数目
        self.dip = dip * numpy.pi / 180

        self.height_synthesis = height_synthesis
        self.height_water = height_water

        if phantom.direction == "transverse":
            if phantom.section_type == 1:  # 综合模
                self.xc, self.yc = phantom.center_x, phantom.center_y  # 模体中心坐标 (横截面)
        elif phantom.direction == "coronal":
            self.yup, self.ylow, self.xleft, self.xright = phantom.yup, phantom.ylow, phantom.xleft, phantom.xright
        elif phantom.direction == "sagittal":  # 目前冠状面和矢状面定位边界方法相同
            self.yup, self.ylow, self.xleft, self.xright = phantom.yup, phantom.ylow, phantom.xleft, phantom.xright

    def transverse(self, DEBUG=False):
        if self.phantom.section_type == 0:  # 水模
            print("请输入合理的综合模横断面切片")
            return None

        left = round(self.xc - self.interval_bottom / 2)  # 横截面中正方形四条边的位置
        right = round(self.xc + self.interval_bottom / 2)
        up = round(self.yc - self.interval_bottom / 2)
        low = round(self.yc + self.interval_bottom / 2)

        if self.sliceThickness == 0.625:  # 当层厚较小时，钨丝灯像素数也较少
            thre = 800
            num = 5
        if self.sliceThickness == 5:
            thre = 300
            num = 20
        else:
            thre = 400
            num = 7

        # 判断输入横断面是否包含可用于层厚测算的钨丝像素
        # 当层厚较薄时，可能会收到立方体其他表面钨丝的干扰
        if DEBUG:
            print('up: ', (self.image[up + 1:up + 6, left:right + 1] > thre).sum())
            print('low: ', (self.image[low - 5:low, left:right + 1] > thre).sum())

            pylab.imshow(self.image, "gray", **dict(vmin=-300))
            pylab.scatter([self.xc], [self.yc], marker='+', c='b')
            pylab.axvline(left, color='r', lw=0.5)
            pylab.axvline(right, color='r', lw=0.5)
            pylab.axhline(up, color='r', lw=0.5)
            pylab.axhline(up + 5, color='g', lw=0.5)
            pylab.axhline(low, color='r', lw=0.5)
            pylab.axhline(low - 5, color='g', lw=0.5)
            pylab.title(os.path.basename(self.phantom.filename))
            pylab.show()

        if (self.image[up + 1:up + 6, left:right + 1] > thre).sum() <= num or \
                (self.image[low - 5:low, left:right + 1] > thre).sum() <= num:
            print("未检测到用于层厚测量的钨丝")
            return None
        ## ** 部分情况下，当横截面靠近立方体上下表面时，区域内钨丝数目可突破以上约束；根据立方体上下表面钨丝在横截面的位置设置新的约束
        line_up = self.image[up + 1:up + 6, left:right + 1].max(axis=0)  # 最大值投影
        line_low = self.image[low - 5:low, left:right + 1].max(axis=0)
        lb_up, nlb_up = label(line_up > 400)
        lb_low, nlb_low = label(line_low > 400)
        length = right - left + 1
        # 下表面_上中 下右
        if nlb_up == 2 and nlb_low == 1 and numpy.where(lb_low == 1)[0][-1] > length - 5 and \
                (numpy.where(lb_up == 2)[0][0] > length / 2 - 7 and numpy.where(lb_up == 2)[0][0] < length / 2 + 7):
            print("钨丝距离立方体边缘过近")
            return None
        # 下表面_上左 下中
        if nlb_up == 1 and nlb_low == 1 and numpy.where(lb_up == 1)[0][0] < 5 and \
                (numpy.where(lb_low == 1)[0][-1] > length / 2 - 5 and numpy.where(lb_low == 1)[0][-1] < length / 2 + 5):
            print("钨丝距离立方体边缘过近")
            return None
        # 上表面_上中 下左
        if nlb_up == 2 and nlb_low == 1 and numpy.where(lb_low == 1)[0][0] < 5 and \
                (numpy.where(lb_up == 1)[0][-1] > length / 2 - 5 and numpy.where(lb_up == 1)[0][-1] < length / 2 + 5):
            print("钨丝距离立方体边缘过近")
            return None
        # 上表面_上右 下中
        if nlb_up == 1 and nlb_low == 1 and numpy.where(lb_up == 1)[0][-1] > length - 5 and \
                (numpy.where(lb_low == 1)[0][0] > length / 2 - 5 and numpy.where(lb_low == 1)[0][0] < length / 2 + 5):
            print("钨丝距离立方体边缘过近")
            return None
        ## **

        # 四条边会定位到空气间隙，向内收缩至基材边界
        arr_l = self.image[up:low + 1, left:left + 10].sum(axis=0)
        left = left + numpy.where((arr_l - arr_l[0]) > 7000)[0][0]
        arr_r = self.image[up:low + 1, right - 9:right + 1].sum(axis=0)[::-1]
        right = right - numpy.where((arr_r - arr_r[0]) > 7000)[0][0]
        arr_u = self.image[up:up + 10, left:right + 1].sum(axis=1)
        up = up + numpy.where((arr_u - arr_u[0]) > 7000)[0][0]
        arr_lo = self.image[low - 9:low + 1, left:right + 1].sum(axis=1)[::-1]
        low = low - numpy.where((arr_lo - arr_lo[0]) > 7000)[0][0]

        # 横断面的层厚由落在正方形上下两边的钨丝线段计算得到
        area1 = self.image[up:up + 5, left:right + 1]  # 钨丝区域框定(扩展5个宽度)
        line1 = area1.max(axis=0)  # 最大值投影
        area2 = self.image[low - 4:low + 1, left:right + 1]
        line2 = area2.max(axis=0)

        if self.sliceThickness == 5:  # 标称值5mm_钨丝的阈值
            lb1, nlb1 = label(line1 > 200)  # 正方形上边/下边还可能存在其余的钨丝或钨珠像素
            lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
            lb2, nlb2 = label(line2 > 200)
            lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

            # 像素数目 > 10，可用于作为层厚判断的钨丝
            # lb1_eff = numpy.where(lb1_len > 10)[0][0] + 1  # 上边钨丝连通域序号
            # lb2_eff = numpy.where(lb2_len > 10)[0][-1] + 1  # 下边钨丝连通域序号
            # 选取包含像素数目最大的连通域作为层厚判断的钨丝
            lb1_eff = lb1_len.argmax() + 1  # 连通域序号
            lb2_eff = lb2_len.argmax() + 1

            # 判断：上/下边的钨丝离立方体上/下边界是否过近
            if numpy.where(lb1 == lb1_eff)[0][0] < 10 or numpy.where(lb1 == lb1_eff)[0][-1] > len(line1) - 10 or \
                    numpy.where(lb2 == lb2_eff)[0][0] < 10 or numpy.where(lb2 == lb2_eff)[0][-1] > len(line2) - 10:
                print("钨丝距离立方体边缘过近")
                return None

            # 钨丝阈值确定：全高半宽
            tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
            # peak1 = (tungsten_pre1[tungsten_pre1 > (tungsten_pre1.max() - 30)]).mean()  # 波峰
            peak1 = tungsten_pre1[tungsten_pre1.argsort()[::-1][:10]].mean()  # 波峰
            base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
            thre1 = peak1 * 0.5 + base1 * 0.5

            tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
            # peak2 = (tungsten_pre2[tungsten_pre2 > (tungsten_pre2.max() - 30)]).mean()
            peak2 = tungsten_pre2[tungsten_pre2.argsort()[::-1][:10]].mean()
            base2 = line2[(line2 > 110) & (line2 < 150)].mean()
            thre2 = peak2 * 0.5 + base2 * 0.5

        elif self.sliceThickness == 2.5:  # 标称值2.5mm_钨丝的阈值
            lb1, nlb1 = label(line1 > 400)  # 正方形上边/下边还可能存在其余的钨丝或钨珠像素
            lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
            lb2, nlb2 = label(line2 > 400)
            lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

            # 选取包含像素数目最大的连通域作为层厚判断的钨丝
            lb1_eff = lb1_len.argmax() + 1  # 连通域序号
            lb2_eff = lb2_len.argmax() + 1

            # 判断：上/下边的钨丝离立方体上/下边界是否过近
            if numpy.where(lb1 == lb1_eff)[0][0] < 10 or numpy.where(lb1 == lb1_eff)[0][-1] > len(line1) - 10 or \
                    numpy.where(lb2 == lb2_eff)[0][0] < 10 or numpy.where(lb2 == lb2_eff)[0][-1] > len(line2) - 10:
                print("钨丝距离立方体边缘过近")
                return None

            # 钨丝阈值确定：全高半宽
            tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
            peak1 = (tungsten_pre1[tungsten_pre1 > (tungsten_pre1.max() - 150)]).mean()  # 波峰
            # peak1 = tungsten_pre1[tungsten_pre1.argsort()[::-1][:5]].mean()  # 波峰
            base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
            thre1 = peak1 * 0.4 + base1 * 0.6

            tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
            peak2 = (tungsten_pre2[tungsten_pre2 > (tungsten_pre2.max() - 150)]).mean()
            # peak2 = tungsten_pre2[tungsten_pre2.argsort()[::-1][:5]].mean()
            base2 = line2[(line2 > 110) & (line2 < 150)].mean()
            thre2 = peak2 * 0.4 + base2 * 0.6

        elif self.sliceThickness == 1.25:  # 标称值1.25mm_钨丝的阈值
            lb1, nlb1 = label(line1 > 800)  # 正方形上边/下边还可能存在其余的钨丝或钨珠像素
            lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
            lb2, nlb2 = label(line2 > 800)
            lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

            # 选取包含像素数目最大的连通域作为层厚判断的钨丝
            lb1_sort = lb1_len.argsort() + 1  # 连通域序号根据对应像素数目由小到大排序
            # *** 考虑上边中部出现其它钨丝长度大于用于层厚判断钨丝长度时的情况
            if len(lb1_sort) == 2 and lb1_len.mean() >= 5 and (
                    (numpy.where(lb1 == lb1_sort[-1]) + left).mean() > 250 and (
                    numpy.where(lb1 == lb1_sort[-1]) + left).mean() < 300):
                lb1_eff = lb1_sort[-2]
            # ***
            else:
                lb1_eff = lb1_sort[-1]

            lb2_eff = lb2_len.argmax() + 1

            # 判断：上/下边的钨丝离立方体上/下边界是否过近
            if numpy.where(lb1 == lb1_eff)[0][0] < 5 or numpy.where(lb1 == lb1_eff)[0][-1] > len(line1) - 5 or \
                    numpy.where(lb2 == lb2_eff)[0][0] < 5 or numpy.where(lb2 == lb2_eff)[0][-1] > len(line2) - 5:
                print("钨丝距离立方体边缘过近")
                return None

            # 钨丝阈值确定：全高半宽
            tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
            peak1 = (tungsten_pre1[tungsten_pre1 > (tungsten_pre1.max() - 150)]).mean()  # 波峰
            # peak1 = tungsten_pre1[tungsten_pre1.argsort()[::-1][:2]].mean()  # 波峰
            base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
            thre1 = peak1 * 0.45 + base1 * 0.55

            tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
            peak2 = (tungsten_pre2[tungsten_pre2 > (tungsten_pre2.max() - 150)]).mean()
            # peak2 = tungsten_pre2[tungsten_pre2.argsort()[::-1][:2]].mean()
            base2 = line2[(line2 > 110) & (line2 < 150)].mean()
            thre2 = peak2 * 0.45 + base2 * 0.55

        elif self.sliceThickness == 0.625:  # 标称值0.625mm_钨丝的阈值
            lb1, nlb1 = label(line1 > 800)  # 正方形上边/下边还可能存在其余的钨丝或钨珠像素
            lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
            lb2, nlb2 = label(line2 > 800)
            lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

            # 选取包含像素数目最大的连通域作为层厚判断的钨丝
            lb1_sort = lb1_len.argsort() + 1  # 连通域序号根据对应像素数目由小到大排序
            # *** 考虑上边中部出现其它钨丝长度大于用于层厚判断钨丝长度时的情况
            if len(lb1_sort) == 2 and lb1_len.mean() >= 4 and (
                    (numpy.where(lb1 == lb1_sort[-1]) + left).mean() > 250 and (
                    numpy.where(lb1 == lb1_sort[-1]) + left).mean() < 300):
                lb1_eff = lb1_sort[-2]
            # ***
            else:
                lb1_eff = lb1_sort[-1]

            lb2_eff = lb2_len.argmax() + 1

            # 判断：上/下边的钨丝离立方体上/下边界是否过近
            if numpy.where(lb1 == lb1_eff)[0][0] <= 3 or numpy.where(lb1 == lb1_eff)[0][-1] >= len(line1) - 3 or \
                    numpy.where(lb2 == lb2_eff)[0][0] <= 3 or numpy.where(lb2 == lb2_eff)[0][-1] >= len(line2) - 3:
                print("钨丝距离立方体边缘过近")
                return None

            # 钨丝阈值确定：全高半宽
            tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
            peak1 = tungsten_pre1.max()  # 波峰
            base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
            thre1 = peak1 * 0.85 + base1 * 0.15

            tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
            peak2 = tungsten_pre2.max()
            base2 = line2[(line2 > 110) & (line2 < 150)].mean()
            thre2 = peak2 * 0.85 + base2 * 0.15

        else:
            print("输入切片层厚暂不支持测算")
            return None

        length1 = (tungsten_pre1 > thre1).sum()  # 上边钨丝长度(像素)
        length2 = (tungsten_pre2 > thre2).sum()  # 下边钨丝长度(像素)

        thickness1 = numpy.tan(self.dip) * length1 * self.phantom.dicom.PixelSpacing[1]  # 由上边钨丝长度确定的层厚
        thickness2 = numpy.tan(self.dip) * length2 * self.phantom.dicom.PixelSpacing[1]  # 由下边钨丝长度确定的层厚

        XLabel1 = left + numpy.where(lb1 == lb1_eff)[0][0] + numpy.where(tungsten_pre1 > thre1)[0]  # 上边钨丝的横纵坐标
        YLabel1 = numpy.array([int(numpy.median(numpy.arange(up, up + 5)))] * len(XLabel1))
        XLabel2 = left + numpy.where(lb2 == lb2_eff)[0][0] + numpy.where(tungsten_pre2 > thre2)[0]  # 下边钨丝的横纵坐标
        YLabel2 = numpy.array([int(numpy.median(numpy.arange(low - 4, low + 1)))] * len(XLabel2))
        self.LabelPos = [numpy.hstack([XLabel1, XLabel2]).astype('int64'),
                         numpy.hstack([YLabel1, YLabel2]).astype('int64')]

        if DEBUG:
            print('up_连通域长度: ', lb1_len)
            print('low_连通域长度: ', lb2_len)

            print("up_钨丝像素数目：", length1)
            print("low_钨丝像素数目：", length2)

            # print("层厚：", thickness1, thickness2)

            pylab.ion()
            pylab.figure()
            pylab.imshow(self.image, "gray", **dict(vmin=-300))
            pylab.scatter([self.xc], [self.yc], marker='+', c='b')
            pylab.axvline(left, color='r', lw=0.5)
            pylab.axvline(right, color='r', lw=0.5)
            pylab.axhline(up, color='r', lw=0.5)
            pylab.axhline(up + 4, color='g', lw=0.5)
            pylab.axhline(low, color='r', lw=0.5)
            pylab.axhline(low - 4, color='g', lw=0.5)
            pylab.title(os.path.basename(self.phantom.filename))
            pylab.show()
            # pylab.close()
            fig, ax = pylab.subplots(1, 2, figsize=(8, 4))
            ax[0].plot(line1, marker='.', lw=1)
            ax[0].set_title("up")
            ax[1].plot(line2, marker='.', lw=1)
            ax[1].set_title("low")
            pylab.ioff()
            pylab.show()
            # pylab.close()

        return (thickness1, thickness2)

    def coronal(self, mode, DEBUG=False):
        if mode == 1:  # *** 输入图像中包含完整的水模+综合模  ***  mode1
            # print('height: ', (self.ylow - self.yup) * self.phantom.dicom.PixelSpacing[0])  # 模体高度
            if (self.ylow - self.yup) * self.phantom.dicom.PixelSpacing[0] < 190:
                print("请输入合理的综合模+水模冠状面切片")
                return None

            # 由模体几何设计得到冠状面位于立方体表面时，宽度约为180mm
            # print((self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1])
            if (self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1] < 178:  # 冠状面位置与立方体没有相交
                print("未检测到钨丝")
                return None
            elif (self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1] >= 178 and \
                    (self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1] < 182:  # 冠状面位置位于立方体表面
                print("钨丝距离立方体边缘过近")
                return None

            # 综合模上、下边界
            # 判断综合模、水模的上下位置关系：像素值大的区域为综合模
            upsum = self.image[self.yup:self.yup + 100, self.xleft:self.xright + 1].sum()
            lowsum = self.image[self.ylow - 99:self.ylow + 1, self.xleft:self.xright + 1].sum()
            if upsum > lowsum:  # 综合模在水模上方
                yup = self.yup
                ylow_ = yup + round(
                    (self.ylow - self.yup) * (self.height_synthesis / (self.height_synthesis + self.height_water)))
            else:  # 综合模在水模下方
                ylow_ = self.ylow
                yup = ylow_ - round(
                    (self.ylow - self.yup) * (self.height_synthesis / (self.height_synthesis + self.height_water)))

            # 冠状面中长方形左、右两条边的位置
            xc = 0.5 * (self.xleft + self.xright)
            xle = round(xc - self.interval_bottom / 2)
            xri = round(xc + self.interval_bottom / 2)

            if DEBUG:
                print((self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1])
                pylab.figure()
                pylab.imshow(self.image, "gray", **dict(vmin=-300))
                pylab.axhline(yup, color='r', lw=0.5)
                pylab.axhline(ylow_, color='r', lw=0.5)
                pylab.axvline(self.xleft, color='r', lw=0.5)
                pylab.axvline(self.xright, color='r', lw=0.5)
                pylab.plot([xle + 1, xle + 1], [yup, ylow_], c='g', lw=0.5)
                pylab.plot([xle + 7, xle + 7], [yup, ylow_], c='g', lw=0.5)
                pylab.plot([xri - 7, xri - 7], [yup, ylow_], c='g', lw=0.5)
                pylab.plot([xri - 1, xri - 1], [yup, ylow_], c='g', lw=0.5)
                pylab.title(os.path.basename(self.phantom.filename))
                pylab.show()

            # 冠状面的层厚由落在长方形左右两边的钨丝线段计算得到
            area1 = self.image[yup:ylow_ + 1, xle + 1:xle + 8]  # 钨丝区域框定(扩展8个宽度)
            line1 = area1.max(axis=1)  # 最大值投影
            area2 = self.image[yup:ylow_ + 1, xri - 7:xri]
            line2 = area2.max(axis=1)

            if self.sliceThickness == 5:  # 标称值5mm_钨丝的阈值
                lb1, nlb1 = label(line1 > 200)  # 长方形左边/右边还可能存在其余的钨丝像素
                lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
                lb2, nlb2 = label(line2 > 200)
                lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

                # 选取包含像素数目最大的连通域作为层厚判断的钨丝
                lb1_eff = lb1_len.argmax() + 1  # 连通域序号
                lb2_eff = lb2_len.argmax() + 1

                # 判断：左/右边的钨丝离立方体前/后边界是否过近
                if numpy.where(lb1 == lb1_eff)[0][0] < 5 or numpy.where(lb1 == lb1_eff)[0][-1] > len(line1) - 5 or \
                        numpy.where(lb2 == lb2_eff)[0][0] < 5 or numpy.where(lb2 == lb2_eff)[0][-1] > len(line2) - 5:
                    print("钨丝距离立方体边缘过近")
                    return None

                # 钨丝阈值确定：全高半宽
                tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
                peak1 = (tungsten_pre1[tungsten_pre1 > (tungsten_pre1.max() - 30)]).mean()  # 波峰
                # peak1 = tungsten_pre1[tungsten_pre1.argsort()[::-1][:10]].mean()  # 波峰
                base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
                thre1 = peak1 * 0.6 + base1 * 0.4

                tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
                peak2 = (tungsten_pre2[tungsten_pre2 > (tungsten_pre2.max() - 30)]).mean()
                # peak2 = tungsten_pre2[tungsten_pre2.argsort()[::-1][:10]].mean()
                base2 = line2[(line2 > 110) & (line2 < 150)].mean()
                thre2 = peak2 * 0.6 + base2 * 0.4

            elif self.sliceThickness == 2.5:  # 标称值2.5mm_钨丝的阈值
                lb1, nlb1 = label(line1 > 400)  # 长方形左边/右边还可能存在其余的钨丝或钨珠像素
                lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
                lb2, nlb2 = label(line2 > 400)
                lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

                if nlb1 >= 3 or nlb2 >= 3:  # 冠状面位置位于立方体表面处
                    print("钨丝距离立方体边缘过近")
                    return None

                # 选取包含像素数目最大的连通域作为层厚判断的钨丝
                lb1_eff = lb1_len.argmax() + 1  # 连通域序号
                lb2_eff = lb2_len.argmax() + 1

                # 判断：左/右边的钨丝离立方体前/后边界是否过近
                if numpy.where(lb1 == lb1_eff)[0][0] < 10 or numpy.where(lb1 == lb1_eff)[0][-1] > len(line1) - 10 or \
                        numpy.where(lb2 == lb2_eff)[0][0] < 10 or numpy.where(lb2 == lb2_eff)[0][-1] > len(line2) - 10:
                    print("钨丝距离立方体边缘过近")
                    return None

                # 钨丝阈值确定：全高半宽
                tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
                peak1 = (tungsten_pre1[tungsten_pre1 > (tungsten_pre1.max() - 150)]).mean()  # 波峰
                # peak1 = tungsten_pre1[tungsten_pre1.argsort()[::-1][:5]].mean()  # 波峰
                base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
                thre1 = peak1 * 0.45 + base1 * 0.55

                tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
                peak2 = (tungsten_pre2[tungsten_pre2 > (tungsten_pre2.max() - 150)]).mean()
                # peak2 = tungsten_pre2[tungsten_pre2.argsort()[::-1][:5]].mean()
                base2 = line2[(line2 > 110) & (line2 < 150)].mean()
                thre2 = peak2 * 0.45 + base2 * 0.55

            elif self.sliceThickness == 1.25:  # 标称值1.25mm_钨丝的阈值
                lb1, nlb1 = label(line1 > 800)  # 长方形左边/右边还可能存在其余的钨丝或钨珠像素
                lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
                lb2, nlb2 = label(line2 > 800)
                lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

                if nlb1 >= 3 or nlb2 >= 3:  # 冠状面位置位于立方体表面处
                    print("钨丝距离立方体边缘过近")
                    return None

                # 选取包含像素数目最大的连通域作为层厚判断的钨丝
                lb1_eff = lb1_len.argmax() + 1  # 连通域序号
                lb2_eff = lb2_len.argmax() + 1

                # 判断：左/右边的钨丝离立方体前/后边界是否过近
                if numpy.where(lb1 == lb1_eff)[0][0] < 10 or numpy.where(lb1 == lb1_eff)[0][-1] > len(line1) - 10 or \
                        numpy.where(lb2 == lb2_eff)[0][0] < 10 or numpy.where(lb2 == lb2_eff)[0][-1] > len(line2) - 10:
                    print("钨丝距离立方体边缘过近")
                    return None

                # 钨丝阈值确定：全高半宽
                tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
                peak1 = (tungsten_pre1[tungsten_pre1 > (tungsten_pre1.max() - 150)]).mean()  # 波峰
                # peak1 = tungsten_pre1[tungsten_pre1.argsort()[::-1][:2]].mean()  # 波峰
                base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
                thre1 = peak1 * 0.7 + base1 * 0.3

                tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
                peak2 = (tungsten_pre2[tungsten_pre2 > (tungsten_pre2.max() - 150)]).mean()
                # peak2 = tungsten_pre2[tungsten_pre2.argsort()[::-1][:2]].mean()
                base2 = line2[(line2 > 110) & (line2 < 150)].mean()
                thre2 = peak2 * 0.7 + base2 * 0.3

            elif self.sliceThickness == 0.4882810116:  # 标称值0.4882810116mm_钨丝的阈值
                lb1, nlb1 = label(line1 > 800)  # 长方形左边/右边还可能存在其余的钨丝或钨珠像素
                lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
                lb2, nlb2 = label(line2 > 800)
                lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

                if nlb1 >= 3 or nlb2 >= 3:  # 冠状面位置位于立方体表面处
                    print("钨丝距离立方体边缘过近")
                    return None

                # 选取包含像素数目最大的连通域作为层厚判断的钨丝
                lb1_eff = lb1_len.argmax() + 1  # 连通域序号
                lb2_eff = lb2_len.argmax() + 1

                # 判断：左/右边的钨丝离立方体前/后边界是否过近
                if numpy.where(lb1 == lb1_eff)[0][0] < 5 or numpy.where(lb1 == lb1_eff)[0][-1] > len(line1) - 5 or \
                        numpy.where(lb2 == lb2_eff)[0][0] < 5 or numpy.where(lb2 == lb2_eff)[0][-1] > len(line2) - 5:
                    print("钨丝距离立方体边缘过近")
                    return None

                # 钨丝阈值确定：全高半宽
                tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
                peak1 = tungsten_pre1.max()  # 波峰
                base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
                thre1 = peak1 * 0.91 + base1 * 0.09

                tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
                peak2 = tungsten_pre2.max()
                base2 = line2[(line2 > 110) & (line2 < 150)].mean()
                thre2 = peak2 * 0.91 + base2 * 0.09

            else:
                print("输入切片层厚暂不支持测算")
                return None

            length1 = (tungsten_pre1 > thre1).sum()  # 左边钨丝长度(像素)
            length2 = (tungsten_pre2 > thre2).sum()  # 右边钨丝长度(像素)

            thickness1 = numpy.tan(self.dip) * length1 * self.phantom.dicom.PixelSpacing[0]  # 由左边钨丝长度确定的层厚
            thickness2 = numpy.tan(self.dip) * length2 * self.phantom.dicom.PixelSpacing[0]  # 由右边钨丝长度确定的层厚

            YLabel1 = yup + numpy.where(lb1 == lb1_eff)[0][0] + numpy.where(tungsten_pre1 > thre1)[0]  # 上边钨丝的横纵坐标
            XLabel1 = numpy.array([int(numpy.median(numpy.arange(xle + 1, xle + 8)))] * len(YLabel1))
            YLabel2 = yup + numpy.where(lb2 == lb2_eff)[0][0] + numpy.where(tungsten_pre2 > thre2)[0]  # 下边钨丝的横纵坐标
            XLabel2 = numpy.array([int(numpy.median(numpy.arange(xri - 7, xri)))] * len(YLabel2))
            self.LabelPos = [numpy.hstack([XLabel1, XLabel2]).astype('int64'),
                             numpy.hstack([YLabel1, YLabel2]).astype('int64')]

            if DEBUG:
                print('left_连通域长度: ', lb1_len)
                print('right_连通域长度: ', lb2_len)

                print('left_钨丝像素数目: ', length1)
                print('right_钨丝像素数目: ', length2)

                pylab.ion()
                pylab.figure()
                pylab.imshow(self.image, "gray", **dict(vmin=-300))
                pylab.axhline(yup, color='r', lw=0.5)
                pylab.axhline(ylow_, color='r', lw=0.5)
                pylab.axvline(self.xleft, color='r', lw=0.5)
                pylab.axvline(self.xright, color='r', lw=0.5)
                pylab.plot([xle + 1, xle + 1], [yup, ylow_], c='g', lw=0.5)
                pylab.plot([xle + 7, xle + 7], [yup, ylow_], c='g', lw=0.5)
                pylab.plot([xri - 7, xri - 7], [yup, ylow_], c='g', lw=0.5)
                pylab.plot([xri - 1, xri - 1], [yup, ylow_], c='g', lw=0.5)
                pylab.title(os.path.basename(self.phantom.filename))
                pylab.show()
                # pylab.close()
                fig, ax = pylab.subplots(1, 2, figsize=(8, 4))
                ax[0].plot(line1, marker='.', lw=1)
                ax[0].set_title("left")
                ax[1].plot(line2, marker='.', lw=1)
                ax[1].set_title("right")
                pylab.ioff()
                pylab.show()
                # pylab.close()

            return (thickness1, thickness2)

        if mode == 2:  # *** 输入图像中水模不完整+综合模(相对)完整  ***  mode2
            # print('height: ', (self.ylow - self.yup) * self.phantom.dicom.PixelSpacing[0])  # 模体高度
            if (self.ylow - self.yup) * self.phantom.dicom.PixelSpacing[0] < 100:
                print("请输入合理的综合模+水模冠状面切片")
                return None

            # 由模体几何设计得到冠状面位于立方体表面时，宽度约为180mm
            # print((self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1])
            if (self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1] < 178:  # 冠状面位置与立方体没有相交
                print("未检测到钨丝")
                return None
            elif (self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1] >= 178 and \
                    (self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1] < 182:  # 冠状面位置位于立方体表面
                print("钨丝距离立方体边缘过近")
                return None

            # 冠状面中长方形左、右两条边的位置
            xc = 0.5 * (self.xleft + self.xright)
            xle = round(xc - self.interval_bottom / 2)
            xri = round(xc + self.interval_bottom / 2)

            # 综合模上、下边界(综合模和水模临近处边界定位相对准确，另一侧的综合模边界由于图像原因无法准确定位)
            # 判断综合模、水模的上下位置关系：像素值大的区域为综合模
            upsum = self.image[self.yup:self.yup + 100, self.xleft:self.xright + 1].sum()
            lowsum = self.image[self.ylow - 99:self.ylow + 1, self.xleft:self.xright + 1].sum()
            height_synthesis_pixel = round(self.height_synthesis / self.phantom.dicom.PixelSpacing[1])
            if upsum > lowsum:  # 综合模在水模上方
                yup = self.yup
                transition = self.image[yup + height_synthesis_pixel - 49:yup + height_synthesis_pixel + 1,
                             xle:xri]  # 默认yup + height_synthesis_pixel落入水模中，向上探寻综合模与水模交接处的边界
                ylow_ = yup + height_synthesis_pixel - 50 + transition.sum(
                    axis=1).argmin()  # 通过综合模与水模交界处的“黑线”，定位综合模靠近水模的边界

            else:  # 综合模在水模下方
                ylow_ = self.ylow
                transition = self.image[ylow_ - height_synthesis_pixel:ylow_ - height_synthesis_pixel + 50,
                             xle:xri]  # # 默认ylow_ - height_synthesis_pixel落入水模中，向下探寻综合模与水模交接处的边界
                yup = ylow_ - height_synthesis_pixel + transition.sum(axis=1).argmin()  # 通过综合模与水模交界处的“黑线”，定位综合模靠近水模的边界

            # 冠状面的层厚由落在长方形左右两边的钨丝线段计算得到
            area1 = self.image[yup:ylow_ + 1, xle + 1:xle + 8]  # 钨丝区域框定(扩展8个宽度)
            line1 = area1.max(axis=1)  # 最大值投影
            area2 = self.image[yup:ylow_ + 1, xri - 7:xri]
            line2 = area2.max(axis=1)

            if DEBUG:
                print((self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1])
                pylab.ion()
                pylab.figure()
                pylab.imshow(self.image, "gray", vmin=-300, vmax=300)
                pylab.axhline(yup, color='r', lw=0.5)
                pylab.axhline(ylow_, color='r', lw=0.5)
                pylab.axvline(self.xleft, color='r', lw=0.5)
                pylab.axvline(self.xright, color='r', lw=0.5)
                pylab.plot([xle + 1, xle + 1], [yup, ylow_], c='g', lw=0.5)
                pylab.plot([xle + 7, xle + 7], [yup, ylow_], c='g', lw=0.5)
                pylab.plot([xri - 7, xri - 7], [yup, ylow_], c='g', lw=0.5)
                pylab.plot([xri - 1, xri - 1], [yup, ylow_], c='g', lw=0.5)
                pylab.title(os.path.basename(self.phantom.filename))
                pylab.show()

                fig, ax = pylab.subplots(1, 2, figsize=(8, 4))
                ax[0].plot(line1, marker='.', lw=1)
                ax[0].set_title("left")
                ax[1].plot(line2, marker='.', lw=1)
                ax[1].set_title("right")
                pylab.ioff()
                pylab.show()

            if self.sliceThickness == 5:  # 标称值5mm_钨丝的阈值
                lb1, nlb1 = label(line1 > 200)  # 长方形左边/右边还可能存在其余的钨丝像素
                lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
                lb2, nlb2 = label(line2 > 200)
                lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

                # 选取包含像素数目最大的连通域作为层厚判断的钨丝
                lb1_eff = lb1_len.argmax() + 1  # 连通域序号
                lb2_eff = lb2_len.argmax() + 1

                # 判断：左/右边的钨丝离立方体前/后边界是否过近
                if numpy.where(lb1 == lb1_eff)[0][0] <= 5 or numpy.where(lb1 == lb1_eff)[0][-1] >= len(line1) - 5 or \
                        numpy.where(lb2 == lb2_eff)[0][0] <= 5 or numpy.where(lb2 == lb2_eff)[0][-1] >= len(line2) - 5:
                    print("钨丝距离立方体边缘过近")
                    return None

                # 钨丝阈值确定：全高半宽
                tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
                peak1 = (tungsten_pre1[tungsten_pre1 > (tungsten_pre1.max() - 30)]).mean()  # 波峰
                # peak1 = tungsten_pre1[tungsten_pre1.argsort()[::-1][:10]].mean()  # 波峰
                base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
                thre1 = peak1 * 0.5 + base1 * 0.5

                tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
                peak2 = (tungsten_pre2[tungsten_pre2 > (tungsten_pre2.max() - 30)]).mean()
                # peak2 = tungsten_pre2[tungsten_pre2.argsort()[::-1][:10]].mean()
                base2 = line2[(line2 > 110) & (line2 < 150)].mean()
                thre2 = peak2 * 0.5 + base2 * 0.5

            elif self.sliceThickness == 1.25:  # 标称值1.25mm_钨丝的阈值
                lb1, nlb1 = label(line1 > 800)  # 长方形左边/右边还可能存在其余的钨丝或钨珠像素
                lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
                lb2, nlb2 = label(line2 > 800)
                lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

                if nlb1 >= 3 or nlb2 >= 3:  # 冠状面位置位于立方体表面处
                    print("钨丝距离立方体边缘过近")
                    return None

                # 选取包含像素数目最大的连通域作为层厚判断的钨丝
                lb1_eff = lb1_len.argmax() + 1  # 连通域序号
                lb2_eff = lb2_len.argmax() + 1

                # 判断：左/右边的钨丝离立方体后边界是否过近
                if numpy.where(lb1 == lb1_eff)[0][0] <= 6 or numpy.where(lb1 == lb1_eff)[0][-1] >= len(line1) - 6 or \
                        numpy.where(lb1 == lb1_eff)[0][0] <= 6 or numpy.where(lb2 == lb2_eff)[0][-1] >= len(line2) - 6:
                    print("钨丝距离立方体边缘过近")
                    return None

                # 钨丝阈值确定：全高半宽
                tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
                peak1 = (tungsten_pre1[tungsten_pre1 > (tungsten_pre1.max() - 150)]).mean()  # 波峰
                # peak1 = tungsten_pre1[tungsten_pre1.argsort()[::-1][:2]].mean()  # 波峰
                base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
                thre1 = peak1 * 0.7 + base1 * 0.3

                tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
                peak2 = (tungsten_pre2[tungsten_pre2 > (tungsten_pre2.max() - 150)]).mean()
                # peak2 = tungsten_pre2[tungsten_pre2.argsort()[::-1][:2]].mean()
                base2 = line2[(line2 > 110) & (line2 < 150)].mean()
                thre2 = peak2 * 0.7 + base2 * 0.3

            elif self.sliceThickness == 0.4882810116:  # 标称值0.4882810116mm_钨丝的阈值
                lb1, nlb1 = label(line1 > 800)  # 长方形左边/右边还可能存在其余的钨丝或钨珠像素
                lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])  # 连通域标记筛选出用于层厚测量的钨丝部分
                lb2, nlb2 = label(line2 > 800)
                lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])

                if nlb1 >= 3 or nlb2 >= 3:  # 冠状面位置位于立方体表面处
                    print("钨丝距离立方体边缘过近")
                    return None

                # 选取包含像素数目最大的连通域作为层厚判断的钨丝
                lb1_eff = lb1_len.argmax() + 1  # 连通域序号
                lb2_eff = lb2_len.argmax() + 1

                # 判断：左/右边的钨丝离立方体前/后边界是否过近
                if numpy.where(lb1 == lb1_eff)[0][0] <= 4 or numpy.where(lb1 == lb1_eff)[0][-1] >= len(line1) - 4 or \
                        numpy.where(lb2 == lb2_eff)[0][0] <= 4 or numpy.where(lb2 == lb2_eff)[0][-1] >= len(line2) - 4:
                    print("钨丝距离立方体边缘过近")
                    return None

                # 钨丝阈值确定：全高半宽
                tungsten_pre1 = line1[lb1 == lb1_eff]  # 连通域指定的钨丝区域（大致范围）
                peak1 = tungsten_pre1.max()  # 波峰
                base1 = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
                thre1 = peak1 * 0.91 + base1 * 0.09

                tungsten_pre2 = line2[lb2 == lb2_eff]  # 连通域指定的钨丝区域（大致范围）
                peak2 = tungsten_pre2.max()
                base2 = line2[(line2 > 110) & (line2 < 150)].mean()
                thre2 = peak2 * 0.91 + base2 * 0.09

            length1 = (tungsten_pre1 > thre1).sum()  # 左边钨丝长度(像素)
            length2 = (tungsten_pre2 > thre2).sum()  # 右边钨丝长度(像素)

            thickness1 = numpy.tan(self.dip) * length1 * self.phantom.dicom.PixelSpacing[0]  # 由左边钨丝长度确定的层厚
            thickness2 = numpy.tan(self.dip) * length2 * self.phantom.dicom.PixelSpacing[0]  # 由右边钨丝长度确定的层厚

            YLabel1 = yup + numpy.where(lb1 == lb1_eff)[0][0] + numpy.where(tungsten_pre1 > thre1)[0]  # 上边钨丝的横纵坐标
            XLabel1 = numpy.array([int(numpy.median(numpy.arange(xle + 1, xle + 8)))] * len(YLabel1))
            YLabel2 = yup + numpy.where(lb2 == lb2_eff)[0][0] + numpy.where(tungsten_pre2 > thre2)[0]  # 下边钨丝的横纵坐标
            XLabel2 = numpy.array([int(numpy.median(numpy.arange(xri - 7, xri)))] * len(YLabel2))
            self.LabelPos = [numpy.hstack([XLabel1, XLabel2]).astype('int64'),
                             numpy.hstack([YLabel1, YLabel2]).astype('int64')]

            return (thickness1, thickness2)

    def sagittal(self, DEBUG=False):
        # *** 输入图像中包含完整的水模+综合模  ***
        # print('height: ', (self.ylow - self.yup) * self.phantom.dicom.PixelSpacing[0])  # 模体高度
        if (self.ylow - self.yup) * self.phantom.dicom.PixelSpacing[0] < 190:
            print("请输入合理的综合模+水模矢状面切片")
            return None

        # 由模体几何设计得到矢状面位于立方体表面时，宽度约为180mm
        # print((self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1])
        if (self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1] < 178:  # 冠状面位置与立方体没有相交
            print("未检测到钨丝")
            return None
        elif (self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1] >= 178 and \
                (self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1] < 182:  # 冠状面位置位于立方体表面
            print("钨丝距离立方体边缘过近")
            return None

        # 综合模上、下边界
        # 判断综合模、水模的上下位置关系：像素值大的区域为综合模
        upsum = self.image[self.yup:self.yup + 100, self.xleft:self.xright + 1].sum()
        lowsum = self.image[self.ylow - 99:self.ylow + 1, self.xleft:self.xright + 1].sum()
        if upsum > lowsum:  # 综合模在水模上方
            yup = self.yup
            ylow_ = yup + round(
                (self.ylow - self.yup) * (self.height_synthesis / (self.height_synthesis + self.height_water)))
        else:  # 综合模在水模下方
            ylow_ = self.ylow
            yup = ylow_ - round(
                (self.ylow - self.yup) * (self.height_synthesis / (self.height_synthesis + self.height_water)))

        # 冠状面中长方形左、右两条边的位置
        xc = 0.5 * (self.xleft + self.xright)
        xle = round(xc - self.interval_bottom / 2)
        xri = round(xc + self.interval_bottom / 2)

        if DEBUG:
            print((self.xright - self.xleft) * self.phantom.dicom.PixelSpacing[1])
            pylab.figure()
            pylab.imshow(self.image, "gray", **dict(vmin=-300))
            pylab.axhline(yup, color='r', lw=0.5)
            pylab.axhline(ylow_, color='r', lw=0.5)
            pylab.axvline(self.xleft, color='r', lw=0.5)
            pylab.axvline(self.xright, color='r', lw=0.5)
            pylab.plot([xle + 1, xle + 1], [yup, ylow_], c='g', lw=0.5)
            pylab.plot([xle + 6, xle + 6], [yup, ylow_], c='g', lw=0.5)
            pylab.plot([xri - 6, xri - 6], [yup, ylow_], c='g', lw=0.5)
            pylab.plot([xri - 1, xri - 1], [yup, ylow_], c='g', lw=0.5)
            pylab.title(os.path.basename(self.phantom.filename))
            pylab.show()

        # 冠状面的层厚由落在长方形左右两边的钨丝线段计算得到
        area1 = self.image[yup:ylow_ + 1, xle + 1:xle + 7]  # 钨丝区域框定(扩展8个宽度)
        line1 = area1.max(axis=1)  # 最大值投影
        area2 = self.image[yup:ylow_ + 1, xri - 6:xri]
        line2 = area2.max(axis=1)

        intercept = round(40 / self.phantom.dicom.PixelSpacing[0])  # 用于层厚判断的两段钨丝间距（像素）  82

        if self.sliceThickness == 5:  # 标称值5mm_钨丝的阈值
            lb1, nlb1 = label(line1 > 400)  # 长方形左边/右边还可能存在其余的钨丝像素
            # lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])
            lb1_median = numpy.array([numpy.median(numpy.where(lb1 == i)[0]) for i in range(1, nlb1 + 1)])  # 连通域中位数

            lb2, nlb2 = label(line2 > 400)
            # lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])
            lb2_median = numpy.array([numpy.median(numpy.where(lb2 == i)[0]) for i in range(1, nlb2 + 1)])

            try:
                for i in range(len(lb1_median) - 1):  # 筛选出间距服从intercept的两连通域序号
                    residue = lb1_median[i + 1:] - lb1_median[i]
                    if len(numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0]) > 0:
                        j = numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0][0]
                        break
                lb1_twice = [i + 1, i + 2 + j]  # 左边钨丝连通域序号
                for i in range(len(lb2_median) - 1):  # 筛选出间距服从intercept的两连通域序号
                    residue = lb2_median[i + 1:] - lb2_median[i]
                    if len(numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0]) > 0:
                        j = numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0][0]
                        break
                lb2_twice = [i + 1, i + 2 + j]  # 右边钨丝连通域序号
            except:  # 无法同时在左右两边同时定位到相距40mm的两段钨丝，则矢状面位置距离立方体左/右边界过近
                print("钨丝距离立方体边缘过近")
                return None

            # 判断：左/右边的钨丝离立方体左/右边界是否过近
            if numpy.where(lb1 == lb1_twice[0])[0][0] < 15 or numpy.where(lb2 == lb2_twice[1])[0][-1] > len(line2) - 15:
                print("钨丝距离立方体边缘过近")
                return None

            # 钨丝阈值确定：全高半宽
            tungsten_left_pre1 = line1[lb1 == lb1_twice[0]]  # 连通域指定的钨丝区域（大致范围）
            tungsten_left_pre2 = line1[lb1 == lb1_twice[1]]
            peak_left1 = (tungsten_left_pre1[tungsten_left_pre1 > (tungsten_left_pre1.max() - 150)]).mean()  # 波峰
            peak_left2 = (tungsten_left_pre2[tungsten_left_pre2 > (tungsten_left_pre2.max() - 150)]).mean()
            # peak_left1 = tungsten_left_pre1[tungsten_left_pre1.argsort()[::-1][:2]].mean()  # 波峰
            # peak_left2 = tungsten_left_pre2[tungsten_left_pre2.argsort()[::-1][:2]].mean()  # 波峰
            base_left = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
            thre_left1 = peak_left1 * 0.6 + base_left * 0.4
            thre_left2 = peak_left2 * 0.6 + base_left * 0.4

            tungsten_right_pre1 = line2[lb2 == lb2_twice[0]]  # 连通域指定的钨丝区域（大致范围）
            tungsten_right_pre2 = line2[lb2 == lb2_twice[1]]
            peak_right1 = (tungsten_right_pre1[tungsten_right_pre1 > (tungsten_right_pre1.max() - 150)]).mean()  # 波峰
            peak_right2 = (tungsten_right_pre2[tungsten_right_pre2 > (tungsten_right_pre2.max() - 150)]).mean()
            # peak_right1 = tungsten_right_pre1[tungsten_right_pre1.argsort()[::-1][:2]].mean()  # 波峰
            # peak_right2 = tungsten_right_pre2[tungsten_right_pre2.argsort()[::-1][:2]].mean()  # 波峰
            base_right = line2[(line2 > 110) & (line2 < 150)].mean()  # 波谷
            thre_right1 = peak_right1 * 0.6 + base_right * 0.4
            thre_right2 = peak_right2 * 0.6 + base_right * 0.4

        elif self.sliceThickness == 2.5:  # 标称值2.5mm_钨丝的阈值
            lb1, nlb1 = label(line1 > 800)  # 长方形左边/右边还可能存在其余的钨丝像素
            # lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])
            lb1_median = numpy.array([numpy.median(numpy.where(lb1 == i)[0]) for i in range(1, nlb1 + 1)])  # 连通域中位数

            lb2, nlb2 = label(line2 > 800)
            # lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])
            lb2_median = numpy.array([numpy.median(numpy.where(lb2 == i)[0]) for i in range(1, nlb2 + 1)])

            try:
                for i in range(len(lb1_median) - 1):  # 筛选出间距服从intercept的两连通域序号
                    residue = lb1_median[i + 1:] - lb1_median[i]
                    if len(numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0]) > 0:
                        j = numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0][0]
                        break
                lb1_twice = [i + 1, i + 2 + j]  # 左边钨丝连通域序号
                for i in range(len(lb2_median) - 1):  # 筛选出间距服从intercept的两连通域序号
                    residue = lb2_median[i + 1:] - lb2_median[i]
                    if len(numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0]) > 0:
                        j = numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0][0]
                        break
                lb2_twice = [i + 1, i + 2 + j]  # 右边钨丝连通域序号
            except:  # 无法同时在左右两边同时定位到相距40mm的两段钨丝，则矢状面位置距离立方体左/右边界过近
                print("钨丝距离立方体边缘过近")
                return None

            # 判断：左/右边的钨丝离立方体左/右边界是否过近
            if numpy.where(lb1 == lb1_twice[0])[0][0] < 15 or numpy.where(lb2 == lb2_twice[1])[0][-1] > len(line2) - 15:
                print("钨丝距离立方体边缘过近")
                return None

            # 钨丝阈值确定：全高半宽
            tungsten_left_pre1 = line1[lb1 == lb1_twice[0]]  # 连通域指定的钨丝区域（大致范围）
            tungsten_left_pre2 = line1[lb1 == lb1_twice[1]]
            peak_left1 = tungsten_left_pre1.max()  # 波峰
            peak_left2 = tungsten_left_pre2.max()
            base_left = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
            thre_left1 = peak_left1 * 0.6 + base_left * 0.4
            thre_left2 = peak_left2 * 0.6 + base_left * 0.4

            tungsten_right_pre1 = line2[lb2 == lb2_twice[0]]  # 连通域指定的钨丝区域（大致范围）
            tungsten_right_pre2 = line2[lb2 == lb2_twice[1]]
            peak_right1 = tungsten_right_pre1.max()  # 波峰
            peak_right2 = tungsten_right_pre2.max()
            base_right = line2[(line2 > 110) & (line2 < 150)].mean()  # 波谷
            thre_right1 = peak_right1 * 0.6 + base_right * 0.4
            thre_right2 = peak_right2 * 0.6 + base_right * 0.4

        elif self.sliceThickness == 1.25:  # 标称值1.25mm_钨丝的阈值
            lb1, nlb1 = label(line1 > 800)  # 长方形左边/右边还可能存在其余的钨丝像素
            # lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])
            lb1_median = numpy.array([numpy.median(numpy.where(lb1 == i)[0]) for i in range(1, nlb1 + 1)])  # 连通域中位数

            lb2, nlb2 = label(line2 > 800)
            # lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])
            lb2_median = numpy.array([numpy.median(numpy.where(lb2 == i)[0]) for i in range(1, nlb2 + 1)])

            try:
                for i in range(len(lb1_median) - 1):  # 筛选出间距服从intercept的两连通域序号
                    residue = lb1_median[i + 1:] - lb1_median[i]
                    if len(numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0]) > 0:
                        j = numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0][0]
                        break
                lb1_twice = [i + 1, i + 2 + j]  # 左边钨丝连通域序号
                for i in range(len(lb2_median) - 1):  # 筛选出间距服从intercept的两连通域序号
                    residue = lb2_median[i + 1:] - lb2_median[i]
                    if len(numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0]) > 0:
                        j = numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0][0]
                        break
                lb2_twice = [i + 1, i + 2 + j]  # 右边钨丝连通域序号
            except:  # 无法同时在左右两边同时定位到相距40mm的两段钨丝，则矢状面位置距离立方体左/右边界过近
                print("钨丝距离立方体边缘过近")
                return None

            # 判断：左/右边的钨丝离立方体左/右边界是否过近
            if numpy.where(lb1 == lb1_twice[0])[0][0] < 15 or numpy.where(lb2 == lb2_twice[1])[0][-1] > len(line2) - 15:
                print("钨丝距离立方体边缘过近")
                return None

            # 钨丝阈值确定：全高半宽
            tungsten_left_pre1 = line1[lb1 == lb1_twice[0]]  # 连通域指定的钨丝区域（大致范围）
            tungsten_left_pre2 = line1[lb1 == lb1_twice[1]]
            peak_left1 = tungsten_left_pre1.max()  # 波峰
            peak_left2 = tungsten_left_pre2.max()
            base_left = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
            thre_left1 = peak_left1 * 0.9 + base_left * 0.1
            thre_left2 = peak_left2 * 0.9 + base_left * 0.1

            tungsten_right_pre1 = line2[lb2 == lb2_twice[0]]  # 连通域指定的钨丝区域（大致范围）
            tungsten_right_pre2 = line2[lb2 == lb2_twice[1]]
            peak_right1 = tungsten_right_pre1.max()  # 波峰
            peak_right2 = tungsten_right_pre2.max()
            base_right = line2[(line2 > 110) & (line2 < 150)].mean()  # 波谷
            thre_right1 = peak_right1 * 0.9 + base_right * 0.1
            thre_right2 = peak_right2 * 0.9 + base_right * 0.1

        elif self.sliceThickness == 0.4882810116:  # 标称值0.4882810116mm_钨丝的阈值
            lb1, nlb1 = label(line1 > 800)  # 长方形左边/右边还可能存在其余的钨丝像素
            # lb1_len = numpy.array([(lb1 == i).sum() for i in range(1, nlb1 + 1)])
            lb1_median = numpy.array([numpy.median(numpy.where(lb1 == i)[0]) for i in range(1, nlb1 + 1)])  # 连通域中位数

            lb2, nlb2 = label(line2 > 800)
            # lb2_len = numpy.array([(lb2 == i).sum() for i in range(1, nlb2 + 1)])
            lb2_median = numpy.array([numpy.median(numpy.where(lb2 == i)[0]) for i in range(1, nlb2 + 1)])

            try:
                for i in range(len(lb1_median) - 1):  # 筛选出间距服从intercept的两连通域序号
                    residue = lb1_median[i + 1:] - lb1_median[i]
                    if len(numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0]) > 0:
                        j = numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0][0]
                        break
                lb1_twice = [i + 1, i + 2 + j]  # 左边钨丝连通域序号
                for i in range(len(lb2_median) - 1):  # 筛选出间距服从intercept的两连通域序号
                    residue = lb2_median[i + 1:] - lb2_median[i]
                    if len(numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0]) > 0:
                        j = numpy.where((residue >= intercept - 2) & (residue <= intercept + 2))[0][0]
                        break
                lb2_twice = [i + 1, i + 2 + j]  # 右边钨丝连通域序号
            except:  # 无法同时在左右两边同时定位到相距40mm的两段钨丝，则矢状面位置距离立方体左/右边界过近
                print("钨丝距离立方体边缘过近")
                return None

            # 判断：左/右边的钨丝离立方体左/右边界是否过近
            if numpy.where(lb1 == lb1_twice[0])[0][0] < 15 or numpy.where(lb2 == lb2_twice[1])[0][-1] > len(line2) - 15:
                print("钨丝距离立方体边缘过近")
                return None

            # 钨丝阈值确定：全高半宽
            tungsten_left_pre1 = line1[lb1 == lb1_twice[0]]  # 连通域指定的钨丝区域（大致范围）
            tungsten_left_pre2 = line1[lb1 == lb1_twice[1]]
            peak_left1 = tungsten_left_pre1.max()  # 波峰
            peak_left2 = tungsten_left_pre2.max()
            base_left = line1[(line1 > 110) & (line1 < 150)].mean()  # 波谷
            thre_left1 = peak_left1 * 0.99 + base_left * 0.01
            thre_left2 = peak_left2 * 0.99 + base_left * 0.01

            tungsten_right_pre1 = line2[lb2 == lb2_twice[0]]  # 连通域指定的钨丝区域（大致范围）
            tungsten_right_pre2 = line2[lb2 == lb2_twice[1]]
            peak_right1 = tungsten_right_pre1.max()  # 波峰
            peak_right2 = tungsten_right_pre2.max()
            base_right = line2[(line2 > 110) & (line2 < 150)].mean()  # 波谷
            thre_right1 = peak_right1 * 0.99 + base_right * 0.01
            thre_right2 = peak_right2 * 0.99 + base_right * 0.01

        else:
            print("输入切片层厚暂不支持测算")
            return None

        length_left1 = (tungsten_left_pre1 > thre_left1).sum()  # 左边钨丝(上)长度(像素)
        length_left2 = (tungsten_left_pre2 > thre_left2).sum()  # 左边钨丝(下)长度(像素)
        length_right1 = (tungsten_right_pre1 > thre_right1).sum()  # 右边钨丝(上)长度(像素)
        length_right2 = (tungsten_right_pre2 > thre_right2).sum()  # 右边钨丝(下)长度(像素)

        thickness_left1 = length_left1 * self.phantom.dicom.PixelSpacing[0] / numpy.tan(self.dip)  # 由左边钨丝长度确定的层厚
        thickness_left2 = length_left2 * self.phantom.dicom.PixelSpacing[0] / numpy.tan(self.dip)
        thickness_right1 = length_right1 * self.phantom.dicom.PixelSpacing[0] / numpy.tan(self.dip)  # 由右边钨丝长度确定的层厚
        thickness_right2 = length_right2 * self.phantom.dicom.PixelSpacing[0] / numpy.tan(self.dip)

        # 左上钨丝的横纵坐标
        YLabel_left1 = yup + numpy.where(lb1 == lb1_twice[0])[0][0] + numpy.where(tungsten_left_pre1 > thre_left1)[0]
        XLabel_left1 = numpy.array([int(numpy.median(numpy.arange(xle + 1, xle + 7)))] * len(YLabel_left1))
        # 左下钨丝的横纵坐标
        YLabel_left2 = yup + numpy.where(lb1 == lb1_twice[1])[0][0] + numpy.where(tungsten_left_pre2 > thre_left2)[0]
        XLabel_left2 = numpy.array([int(numpy.median(numpy.arange(xle + 1, xle + 7)))] * len(YLabel_left2))
        # 右上钨丝的横纵坐标
        YLabel_right1 = yup + numpy.where(lb2 == lb2_twice[0])[0][0] + numpy.where(tungsten_right_pre1 > thre_right1)[0]
        XLabel_right1 = numpy.array([int(numpy.median(numpy.arange(xri - 6, xri)))] * len(YLabel_right1))
        # 右下钨丝的横纵坐标
        YLabel_right2 = yup + numpy.where(lb2 == lb2_twice[1])[0][0] + numpy.where(tungsten_right_pre2 > thre_right2)[0]
        XLabel_right2 = numpy.array([int(numpy.median(numpy.arange(xri - 6, xri)))] * len(YLabel_right1))

        self.LabelPos = [numpy.hstack([XLabel_left1, XLabel_left2, XLabel_right1, XLabel_right2]).astype('int64'),
                         numpy.hstack([YLabel_left1, YLabel_left2, YLabel_right1, YLabel_right2]).astype('int64')]

        if DEBUG:
            # print('left_连通域长度: ', lb1_len)
            # print('right_连通域长度: ', lb2_len)

            print('left_连通域中位数: ', lb1_median)
            print('right_连通域中位数: ', lb2_median)

            print('left_选中连通域中序号: ', lb1_twice)
            print('right_选中连通域中序号: ', lb2_twice)

            # print('left_钨丝像素数目: ', length_left1, length_left2)
            # print('right_钨丝像素数目: ', length_right1, length_right2)

            pylab.ion()
            pylab.figure()
            pylab.imshow(self.image, "gray", **dict(vmin=-300))
            pylab.axhline(yup, color='r', lw=0.5)
            pylab.axhline(ylow_, color='r', lw=0.5)
            pylab.axvline(self.xleft, color='r', lw=0.5)
            pylab.axvline(self.xright, color='r', lw=0.5)
            pylab.plot([xle + 1, xle + 1], [yup, ylow_], c='g', lw=0.5)
            pylab.plot([xle + 6, xle + 6], [yup, ylow_], c='g', lw=0.5)
            pylab.plot([xri - 6, xri - 6], [yup, ylow_], c='g', lw=0.5)
            pylab.plot([xri - 1, xri - 1], [yup, ylow_], c='g', lw=0.5)
            pylab.title(os.path.basename(self.phantom.filename))
            pylab.show()
            # pylab.close()
            fig, ax = pylab.subplots(1, 2, figsize=(8, 4))
            ax[0].plot(line1, marker='.', lw=1)
            ax[0].set_title("left")
            ax[1].plot(line2, marker='.', lw=1)
            ax[1].set_title("right")
            pylab.ioff()
            pylab.show()
            # pylab.close()

        # return ((thickness_left1, thickness_left2), (thickness_right1, thickness_right2))
        return ((thickness_left1 + thickness_left2) * 0.5, (thickness_right1 + thickness_right2) * 0.5)

    ####################################################################################


if __name__ == "__main__":
    # 横例
    # phantom = CT_phantom(r"C:\Users\10746\Desktop\GECT\transverse\primary\0.625mm_2\182.dcm")
    # Thick = thickness_new(phantom)
    # thick = Thick.transverse(DEBUG=True)
    # print(thick)

    # 冠例
    for i in range(20, 21):
        print("%03d.dcm" % i)
        phantom = CT_phantom(
            r"D:\Ph.D\博二\博二下\医学CT模体\my_work\2022年数据\reviewd\CT_heart\coronal\5_2\%03d.dcm" % i)
        Thick = thickness_new(phantom)
        thick = Thick.coronal(mode=2, DEBUG=True)
        print(thick)

    # 矢例
    # phantom = CT_phantom(r"C:\Users\10746\Desktop\GECT\sagittal\0.4882810116mm\160.dcm")
    # Thick = thickness_new(phantom)
    # thick = Thick.sagittal(DEBUG=True)
    # print(thick)

    # *** 2022.3 ***

    # 1. 横断面
    # os.chdir(r"C:\Users\10746\Desktop\GECT\transverse\secondary\0.625mm_2")
    # a, b = [], []
    # for dcm in os.listdir('./'):
    #     print(os.path.basename(dcm))
    #     phantom = CT_phantom(dcm)
    #     Thick = thickness_new(phantom)
    #     thick = Thick.transverse(DEBUG=False)
    #     try:
    #         a.append(thick[0])
    #         b.append(thick[1])
    #     except:
    #         pass
    # print('%.2f' % numpy.array(a).mean())
    # print('%.2f' % numpy.array(b).mean())

    # 2. 冠状面
    # os.chdir(r"D:\Ph.D\博二\博二下\医学CT模体\my_work\2022年数据\reviewd\CT_heart\coronal\1.25")
    # a, b = [], []
    # for dcm in os.listdir('./'):
    #     print(os.path.basename(dcm))
    #     phantom = CT_phantom(dcm)
    #     Thick = thickness_new(phantom)
    #     thick = Thick.coronal(mode=2, DEBUG=False)
    #     try:
    #         a.append(thick[0])
    #         b.append(thick[1])
    #     except:
    #         pass
    # print('%.2f' % numpy.array(a).mean())
    # print('%.2f' % numpy.array(b).mean())

    # 3. 矢状面
    # os.chdir(r"D:\Ph.D\博二\博二下\医学CT模体\my_work\2021年数据\reviewd\children-D_\sagittal\secondary\2.5mm")
    # # a1, a2, b1, b2 = [], [], [], []
    # a, b = [], []
    # for dcm in os.listdir('./'):
    #     print(os.path.basename(dcm))
    #     phantom = CT_phantom(dcm)
    #     Thick = thickness_new(phantom)
    #     thick = Thick.sagittal(DEBUG=False)
    #     try:
    #         # a1.append(thick[0][0])
    #         # a2.append(thick[0][1])
    #         # b1.append(thick[1][0])
    #         # b2.append(thick[1][1])
    #         a.append(thick[0])
    #         b.append(thick[1])
    #     except:
    #         pass
    # # print('%.2f' % numpy.array(a1).mean(), '%.2f' % numpy.array(a2).mean())
    # # print('%.2f' % numpy.array(b1).mean(), '%.2f' % numpy.array(b2).mean())
    # print('%.2f' % numpy.array(a).mean())
    # print('%.2f' % numpy.array(b).mean())

# *** 2022.3 ***

# fname = "D:/motituxiang/motituxiang/横断-体检中心0515\Z408"
# ##    fname = "D:/motituxiang/motituxiang/冠状和矢状-3\Z1002"
# dcm= dicom.read_file(fname)
# phantom = CT_phantom(dcm)
# spiralbeads = SpiralBeads(phantom, interval = 72, dip = 26.56)#diameter=75, pitch=90,number_beads=180
# profile1 = spiralbeads.get_profile_transverse(displayImage=True)
# fname = u"D:\\CT-D-phantom\\矢状面\\Z354"  # Z589,615,467
# fname = "D:\\医学模体图像\\301-1比1螺距\\301-1比1螺距\\BSoft\SERIES8\\IMAGE27"
# fname = "D:\\医学模体图像\\SE4-冠-5.0\\IM27"
# fname = "D:\\医学模体图像\\三维模横断面螺旋扫描 (2)\\三维模横断面螺旋扫描\\5.0\\Z305"
# fname = "D:\\医学模体图像\\0803\\PA1\PA1\\ST0\\SE2\\IM19"
# fname = "D:\\医学模体图像\\2\\3-5.0\\1 (19).dcm"
# dcm = dicom.read_file(fname)
# phantom = CT_phantom(dcm)
# # spiralbeads = SpiralBeads(phantom, interval=72, dip=26.56)
# # # th = spiralbeads.get_thickness_tranverse_test(displayImage=True)
# profile1 = spiralbeads.get_profile_transverse(displayImage=True)  # ！！！！横断面！！！！
# profile = spiralbeads.locate_profile_coronal(displayImage=True)  # sagittal    ！！！！冠状面、矢状面！！！！
# thickness = spiralbeads.get_thickness_coronal(profile)
# fname = "D:\\医学模体图像\\三维-横断-轴扫\\5.0\\Z598"
# dcm= dicom.read_file(fname)
# phantom = CT_phantom(dcm)
# spiralbeads = SpiralBeads(phantom, interval = 72, dip = 26.56)
# profile1 = spiralbeads.get_profile_transverse(displayImage=True)
# profile = spiralbeads.locate_profile_coronal(displayImage = False)  #sagittal
# thickness = spiralbeads.get_thickness_coronal(profile)

# pname = u"D:\\CT-D-phantom\\矢状面"
#
# files = sorted(os.listdir(pname),key=alphanum_key)
#
# zonghe=[]
# biaocheng=[]
# slice=[]
# WuCha=[]
# a=0
# wb=xlwt.Workbook()
# sh=wb.add_sheet('zonghe')
#
# for f in files[0:20]:
#     fname = os.path.join(pname, f)
#     print (fname)
#     dcm = dicom.read_file(fname)
#     phantom = CT_phantom(dcm)
#
#     spiralbeads = SpiralBeads(phantom, interval = 72, dip = 26.56)
#     profile = spiralbeads.locate_profile_sagittal(displayImage=False)
#     #profile_left,profile_right = spiralbeads.get_profile_coro_sagi(displayImage=False)
#     #profile_right = profile_right[2000:spiralbeads.number_samples-2000]
#     thickness = spiralbeads.get_thickness_sagittal(profile)
#     if thickness is None:
#         print ("cannot estimate the slice thickness!")
#     else:
#         print ("The measured slice thickness is %f"%thickness)
#
#         wuchazhi=(thickness-dcm.SliceThickness)*100/dcm.SliceThickness
#         zonghe.append(str(fname).replace('矢状',''))
#         biaocheng.append(str(dcm.SliceThickness))
#         slice.append(str(thickness))
#         WuCha.append(str(wuchazhi))
#         a=a+1
#         sh.write(a,0,zonghe[-1:])
#         sh.write(a,1,biaocheng[-1:])
#         sh.write(a,2,slice[-1:])
#         sh.write(a,3,WuCha[-1:])
#
# sh.write(0,0,u'图像名称')
# sh.write(0,1,u'层厚标称值')
# sh.write(0,2,u'层厚测量值')
# sh.write(0,3,u'测量误差%')
#wb.save('冠状面-左右各偏移两个像素.xls')

#pylab.boxplot(float(slice))
#pylab.show()

