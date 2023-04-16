from __future__ import division

# from ._qt import QtCore, QtWidgets, Signal, pyqtSlot
from qtpy import QtWidgets, QtCore, QtGui
#import dicom_data
import glob
import numpy as np
import pydicom as dicom
#from imports import pydicom

# Anatomical planes
TRANSVERSE = AXIAL = 0
FRONTAL = CORONAL = 1
MEDIAN = SAGITTAL = 2
ALLOWED_PLANES = (AXIAL, CORONAL, SAGITTAL)


class DicomWidget(QtWidgets.QLabel):
    """Widget for displaying DICOM data.

    """
    def __init__(self, parent, **kwargs):
        # Qt initialization
        QtWidgets.QLabel.__init__(self, parent)
        self.setCursor(QtCore.Qt.CrossCursor)
        #self.setScaledContents (True)
        self.setMouseTracking(True)
        
        # Inner data
        self._zoom_level = kwargs.get("zoom_level", 0)
        self._data = kwargs.get("data", None)
        self._scaled_image = None
        self._low_hu = kwargs.get("low_hu", -1000)
        self._high_hu = kwargs.get("high_hu", 3000)
        self._plane = kwargs.get("plane", AXIAL)
        self._slice = kwargs.get("slice", 0)
        self._color_table = kwargs.get("color_table", [QtGui.qRgb(i, i, i) for i in range(256)])

        self._image = None
        self._pixmap = None

        # Signals & slots
        self._auto_wire()

        self.update_image()
        self.parent=parent
        self.setMinimumSize(400,400)
    data_changed = QtCore.Signal(name="data_changed")
    zoom_changed = QtCore.Signal(name="zoom_changed")
    calibration_changed = QtCore.Signal(name="calibration_changed")

    # == slice_changed OR plane_changed
    data_selection_changed = QtCore.Signal(name="data_selection_changed")
    slice_changed = QtCore.Signal(name="slice_changed")
    plane_changed = QtCore.Signal(name="plane_changed")

    def resizeEvent(self, event):
        self.text = "0k,Resized to QSize({}, {})".format(
                            event.size().width(), event.size().height())
        #print self.text
        self.update_pixmap()

    def _auto_wire(self):
        """Wire all signals & slots that are necessary for the widget to work."""
        self.zoom_changed.connect(self.on_zoom_changed)
        self.data_changed.connect(self.on_data_changed)
        self.slice_changed.connect(self.on_data_selection_changed)
        self.plane_changed.connect(self.on_data_selection_changed)

    @property
    def zoom_level(self):
        """Zoom level.

        An integer value useful for the GUI
        0 = 1:1, positive values = zoom in, negative values = zoom out
        """
        return self._zoom_level

    @property
    def zoom_factor(self):
        """Real size of data voxel in screen pixels."""
        print self._zoom_level
        if self._zoom_level > 0:
            return self._zoom_level + 1
        else:
            return 1.0 / (1 - self._zoom_level)

    @zoom_level.setter
    def zoom_level(self, value):
        if self._zoom_level != value:
            self._zoom_level = value
            self.zoom_changed.emit()

    def decrease_zoom(self, amount=1):
        self.zoom_level -= amount

    def increase_zoom(self, amount=1):
        self.zoom_level += amount

    def reset_zoom(self):
        self.zoom_level = 0

    @QtCore.Slot()
    def on_zoom_changed(self):
        if self._image:
            self.update_image1()

    @QtCore.Slot()
    def on_data_changed(self):
        self.update_image()

    @QtCore.Slot()
    def on_calibration_changed(self):
        self.update_image()

    @QtCore.Slot()
    def on_data_selection_changed(self):

        self.update_image()



    def update_image(self):
        if self._data is not None:
            # Prepare image integer data
            #self.parent.horizontalSlider.setSliderPosition(0)
            #self.parent.horizontalSlider1.setSliderPosition(self.parent.horizontalSlider1.maximum())
            raw_data = self._data.get_slice(self.plane, self.slice)
            shape = raw_data.shape
            #print shape
            
            data = (raw_data - self._low_hu) / self.window_width * 256
            
                #print xlen
            #print 20000000000000000000000000
            #print data[200]
            data[data < 0] = 0
            data[data > 255] = 255
            #print 30000000000000000000000000
            #print data[200]
            #data=(data-self.parent.mid-self.parent.length)/self.parent.length*255/2
            data=(data-self.parent.mid)/(self.parent.length-self.parent.mid)*255
            data[data < 0] = 0
            data[data > 255] = 255
            #print 40000000000000000000000000
            #print data[200]
            data = data.astype("int8")

            self._image = QtGui.QImage(data, data.shape[1], data.shape[0],data.shape[1], QtGui.QImage.Format_Indexed8)#1,1
            self._image.setColorTable(self._color_table)
        else:
            self._image = None
        self.update_pixmap()
    def update_image1(self):
        if self._data is not None:
            # Prepare image integer data
            raw_data = self._data.get_slice(self.plane, self.slice)
            #print raw_data,raw_data.max(),raw_data.min(),self._low_hu,self.window_width
            shape = raw_data.shape
            data = (raw_data - self._low_hu) / self.window_width * 256
            #print self.window_width
            data[data < 0] = 0
            data[data > 255] = 255
            if self.parent.mid==self.parent.length:
                self.parent.length+=1
            if self.parent.mid>self.parent.length:
                t=self.parent.mid
                self.parent.mid=self.parent.length
                self.parent.length=t
            data=(data-self.parent.mid)/(self.parent.length-self.parent.mid)*255
            #print data
            data[data < 0] = 0
            data[data > 255] = 255
            data = data.astype("int8")
            #print data.shape[1], data.shape[0]

            self._image = QtGui.QImage(data, data.shape[1], data.shape[0], data.shape[1],QtGui.QImage.Format_Indexed8)
            self._image.setColorTable(self._color_table)
        else:
            self._image = None
        self.update_pixmap()


    def update_pixmap(self):
        if self._image is not None:
            pixmap = QtGui.QPixmap.fromImage(self._image)
            if 1:#self.zoom_factor != 1:
                if self.width()<self.height():
                    size0= self.width()
                    #size1 = 1##self.height()
                    #print "ok"
                else:
                    size0 = self.height()
                    #size1=0
                #print self._data.shape[2],self._data.shape[1],self.width(),self.height()
                if 1:#self.zoom_factor < 1:
                    pixmap = pixmap.scaled(size0,#size1*(self._data.shape[2])*self.height()/(self._data.shape[1])+self.width()*size0,
                                           size0,#size0*(self._data.shape[1])*self.width()/(self._data.shape[2])+self.height()*size1,
                                                 QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

                    #print pixmap.size()

                else:
                    
                    pixmap = pixmap.scaled(size0,size0,#pixmap.width() * self.zoom_factor*0.5, pixmap.height() * self.zoom_factor*0.5,
                                           QtCore.Qt.KeepAspectRatio)

                
            self._pixmap = pixmap
            self.setPixmap(self._pixmap)
            #self.resize(pixmap.width(), pixmap.height())
        else:
            self.setText("No image.")
       

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        if self._data != d:
            self._data = d
            self.data_changed.emit()

    @property
    def window_center(self):
        return (self.high_hu + self.low_hu) / 2

    @window_center.setter
    def window_center(self, value):
        if value != self.window_center:
            original = self.window_center
            self._low_hu += value - original
            self._high_hu += value - original
            self.calibration_changed.emit()

    @property
    def window_width(self):
        return self._high_hu - self._low_hu

    @window_width.setter
    def window_width(self, value):
        if value < 0:
            value = 0
        original = self.window_width
        if value != original:
            self._low_hu -= (value - original) / 2
            self._high_hu = self._low_hu + value
            self.calibration_changed.emit()

    @property
    def plane(self):
        return self._plane

    @plane.setter
    def plane(self, value):
        if value != self._plane:
            if value not in [dicom_data.ALLOWED_PLANES]:
                raise ValueError("Invalid plane identificator")
            self._plane = value
            self.plane_changed.emit()
            self.data_selection_changed.emit()

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, n):
        if n != self._slice:
            self._slice = n
            self.slice_changed.emit()
            self.data_selection_changed.emit()

    @property
    def slice_count(self):
        if not self._data:
            return 0
        else:
            return self._data.shape[self.plane]


    def mouseDoubleClickEvent(self, event):
        #:Event(self, event):
        # self.last_move_x = event.x()
        # self.last_move_y = event.y()

        self.parent.center_selected(event.x(), event.y())

class DicomData(object):
    ALLOWED_MODALITIES = ('CT', 'MR', 'CR', 'RT')

    def __init__(self, data, **kwargs):
        self._array = data
        self.modality = kwargs.get("modality")

    @classmethod
    def from_files(cls, files):
        data = []
        modality = None

        for file_path in files:
            f = dicom.read_file(file_path)
            print("Reading %s..." % file_path)

            # Get modality
            if modality:
                if modality != f.Modality:
                    raise RuntimeError("Cannot mix images from different modalities")
            elif f.Modality not in cls.ALLOWED_MODALITIES:
                raise RuntimeError("%s modality not supported" % f.Modality)
            else:
                modality = f.Modality
            data.append(cls._read_pixel_data(f))
        return cls(np.array(data), modality=modality)

    @classmethod
    def _read_pixel_data(cls, f):
        if f.Modality == "CT":
            data =  f.pixel_array
            
##            if data.shape[0]!= data.shape[1]:
##                xlen = min(data.shape[0],data.shape[1])
##                xlenl= max(data.shape[0],data.shape[1])
##                data = data[0:xlen,0:xlen]
#            data=data[0:510,0:50]
##            if data.shape[0]!= data.shape[1]:
##                if data.shape[0]> data.shape[1]:
##                    data =np.c_[data, np.zeros((np.shape(data)[0],np.shape(data)[0]-np.shape(data)[1]))]
##                else:
##                    data =np.r_[data, np.zeros(((np.shape(data)[1]-np.shape(data)[0]),(np.shape(data)[1])))]
            data = f.RescaleSlope * data + f.RescaleIntercept
            
            #print np.shape(data)
            #print 12112,data,f.RescaleSlope ,f.pixel_array ,f.RescaleIntercept
            return np.array(data)
        else:
            #print 565656
            return np.array(f.pixel_array)

    @property
    def shape(self):
        return self._array.shape

    @property
    def array(self):
        """The underlying numpy array."""
        return self._array

    def get_slice(self, plane, n):
        if plane not in ALLOWED_PLANES:
            raise ValueError("Invalid plane identificator (allowed are 0,1,2)")
        index = [slice(None, None, None) for i in range(3)]
        index[plane] = n
        #print 888888
        #print np.size(self._array[index]),np.size(self._array),index,plane
        return self._array[tuple(index)]

    def get_slice_shape(self, plane):
        # TODO: 
        shape = list(self.shape)
        shape.pop(plane)
        return shape
