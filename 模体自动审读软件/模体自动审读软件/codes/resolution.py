# -*- coding: utf-8 -*-
"""
Created on Wed May 03 10:49:52 2017

@author: xma_m
"""
import os, sys, scipy, pylab
import numpy as np
import pydicom as dicom
from util import find_CT_phantom_outer_edge, read_dicom_file
currentfilefolder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(currentfilefolder, 'Util')))

def freq_at_mtf_perc(feq, mtf, perc):
    """
    find the frequencies at given %mtf
    the first two parameters are the mtf plot
    more specifically, the first is the frequency list
                       the second is the mtf at the frequencies
    the last parameter is a %mtf whose frequency needs to be calculated
    """
    #NOTE: mtf here is a function with max of 1
    if mtf.min() > perc:
        return -1

    # in case there is a point satisfying the percentage
    if scipy.any(mtf == perc):
        return feq[scipy.where(mtf == perc)]
    # try to find the two points where the targeting percentage is in between
    inds1 = scipy.where(mtf >= perc)[0].tolist()
    ind2 = scipy.where(mtf < perc)[0][0]
    # to get rid of the indices that come from the wiggle at the end of the spectrum
    #  in some cases, the MTF curve raises up at the higher frequency
    ind1 = inds1.pop(-1)
    while ind1 != len(inds1):
        ind1 = inds1.pop(-1)
    #print "the indices enclosing the target frequecy:", ind1,ind2
    f1, f2, m1, m2 = feq[ind1], feq[ind2], mtf[ind1], mtf[ind2]
    #print "the frequencies and the MTF's: "f1, f2, m1, m2
    return f2 - (f2-f1)*(perc - m2)/(m1-m2)

def FFT_2D(psf, sz=0, fftshift=True):
        """
        perform 2D Fourier transform (FT) for a 2D PSF function
        with or without zero-padding
        when sz is less than the size of the psf, do not do zero-padding
        else do zero-padding
        if fftshift is set, do fftshift so that the zero-frequency is at the center
        """
        h, w = psf.shape
        if h < w:
            d = h
        else:
            d = w

        if sz > d:
            zero_padded = np.zeros((sz, sz))
            zero_padded[(sz-h)/2: (sz+h)/2,
                        (sz-w)/2: (sz+w)/2] = psf
        else:
            zero_padded = psf
        
        #mtf_2d = np.fft.fftshift(np.fft.fftn(PSF_ROI))
        #mtf_2d = np.fft.fftshift(np.fft.fftn(zero_padded))
        mtf_2d = np.fft.fftn(zero_padded)
        if fftshift:
            mtf_2d = np.fft.fftshift(mtf_2d)
        return scipy.absolute(mtf_2d)
    
def fft2DRadProf(ft, d=1.):
    coord = np.fft.fftshift(np.fft.fftfreq(ft.shape[0], d))
    image = np.fft.fftshift(ft)

    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    center = np.squeeze(np.array([x[0, coord == 0.0], y[coord == 0.0, 0]]))
    r = np.hypot(x - center[0], y - center[1]) #compute the distance of every point to (0,0)
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind] #list of distance in a sorted manner
    i_sorted = image.flat[ind] #list of "frequency"(?) sorted according to distance

    # Get the integer part of the radii (bin size = 1)
    r_int = np.rint(r_sorted).astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of points in each radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr
    cl = np.rint(np.sum(coord >= 0)) #so that coord and radial_prof have the same number of points??????

    return coord[coord >= 0], radial_prof[:int(cl)]

    
def roi_generator(imageshape, xc,yc, radius):
        img = scipy.ones(imageshape)
        img[xc,yc,]=0
        newimage = scipy.ndimage.distance_transform_edt(img)
        roi = newimage < radius
        return roi
    
class TungstenBead:
    def __init__(self, phantomimage, roiwidth = 32, windowwidth = 11,**kwargs):
        dcm, dpa, ds = read_dicom_file(phantomimage)
        self.dcm = dcm
        self.dcm_shape = dcm.pixel_array.shape
        self.__phantomimage = dpa
        self.__pixelspacing = ds
        self.__roiwidth = roiwidth/dcm.PixelSpacing[0]*0.44
        self.__windowwidth = windowwidth
        self.__roi = None
        self.resolution = None
        self.mtf = None
        self.freq = None
        self.pos = self.__InitPosition(**kwargs)
        self.__backgroundvalue = self.__EstimateBackgroundValue()
        self.psf = self.__GetPSF()
        self.__CalculateMTF(**kwargs)
     

    def __InitPosition(self,**kwargs):
#       r = self.__roiwidth / 2
        r = int(self.__roiwidth / 2 + 0.5)
        

        
        if 'bead_position' in kwargs:
#            beadpositionx, beadpositiony = scipy.where(self.__phantomimage == self.__phantomimage.max())
            beadpositionx, beadpositiony = kwargs['bead_position']
        elif kwargs:
            fh=pylab.figure(dpi = 150, figsize = (10, 10))
            pylab.imshow(self.__phantomimage, cmap='gray', interpolation = 'nearest')
            pylab.title('Pinpoint the Tungsten Bead/Line')
            pos = pylab.ginput(1)[0]
            pylab.close(fh)
            xc, yc = pos
            self.__roi = scipy.array(self.__phantomimage[yc-r : yc+r+1, xc-r : xc+r+1], dtype = scipy.float32)
            [beadpositiony, beadpositionx] = scipy.where(self.__roi == self.__roi.max())
            beadpositiony, beadpositionx = beadpositiony + yc -r, beadpositionx + xc - r
        else:
            xc, yc, r_outer,FUBU = find_CT_phantom_outer_edge(self.__phantomimage,self.dcm.PixelSpacing[0], return_coors = False)
            xc = int(xc+0.5/self.__pixelspacing*0.44)
            yc = int(yc+0.5/self.__pixelspacing*0.44)
            roi = roi_generator(self.dcm_shape, xc,yc, 80./self.__pixelspacing*0.44)#70,20200509
            ROI = roi*self.dcm.pixel_array
##            import pylab
##            pylab.imshow(ROI,cmap = "gray")
##            pylab.show()
            [beadpositiony, beadpositionx] = scipy.where(ROI == ROI.max())
            beadpositiony = beadpositiony[0]
            beadpositionx = beadpositionx[0]
            print beadpositiony-r , beadpositiony+r+1, beadpositionx-r ,beadpositionx+r+1,yc,xc
        
        self.__roi = scipy.array(self.__phantomimage[beadpositiony-r : beadpositiony+r+1, beadpositionx-r : beadpositionx+r+1], dtype = scipy.float32)
        self.beadpositionx=beadpositionx
        self.beadpositiony=beadpositiony
#       print kwargs
        print beadpositionx, beadpositiony
        return scipy.array([beadpositionx, beadpositiony])
        
    def __EstimateBackgroundValue(self):
        sumforestimation = self.__roi[0:5,:].sum() + self.__roi[-5:,:].sum() + self.__roi[6:-5,0:5].sum() + self.__roi[6:-5,-5:].sum()
        backgroundvalue = float(sumforestimation)/((self.__roi.shape[0] * 10) + (self.__roi.shape[1] * 10) - 25 * 4)
        #print 'Estimated Background = ', backgroundvalue
        '''
        # displaying the region for background estimation
        a = scipy.zeros(self.__roi.shape)
        a[0:5,:] = 1
        a[-5:,:] = 1
        a[5:-5,0:5] = 1
        a[5:-5,-5:] = 1
        pylab.figure(dpi = 150, figsize = (8, 6))
        imh = pylab.imshow(self.__roi,cmap = 'gray', interpolation = 'nearest')
        pylab.imshow(a, cmap = 'gray', interpolation = 'nearest', alpha = 0.3)
        pylab.title('Area for Background Estimation')
        pylab.colorbar(imh)
        pylab.axis('off')   
        '''
        return backgroundvalue
        
    def __GetPSF(self):
        yc, xc = scipy.where(self.__roi == self.__roi.max())
       
        psf = self.__roi - self.__backgroundvalue
        smallwindow = scipy.matrix(scipy.hanning(self.__windowwidth)).T * scipy.matrix(scipy.hanning(self.__windowwidth))
        fullsizewindow = scipy.zeros(psf.shape, dtype = float)
        yc, xc = yc[0], xc[0]
        print "############", smallwindow.shape
        print yc, self.__windowwidth, xc
        print fullsizewindow.shape
        print fullsizewindow[yc-(self.__windowwidth-1)/2 : yc+(self.__windowwidth-1)/2+1, \
                xc-(self.__windowwidth-1)/2 : xc+(self.__windowwidth-1)/2+1].shape
        #print fullsizewindow[0: yc+(self.__windowwidth-1)/2+1, \
        #        xc-(self.__windowwidth-1)/2 : xc+(self.__windowwidth-1)/2+1].shape
        #print fullsizewindow[yc-(self.__windowwidth-1)/2 : yc+(self.__windowwidth-1)/2+1, \
        #        0 : xc+(self.__windowwidth-1)/2+1].shape
        #print xc+(self.__windowwidth-1)/2+1,xc-(self.__windowwidth-1)/2 
        print "#######"
        if (yc+(self.__windowwidth-1)/2+1)>fullsizewindow.shape[0]:
            yc = -(self.__windowwidth-1)/2-1+fullsizewindow.shape[0]
        print yc
        if yc-(self.__windowwidth-1)/2<0:
            yc = (self.__windowwidth-1)/2
        if xc-(self.__windowwidth-1)/2<0:
            xc = (self.__windowwidth-1)/2
        if (xc+(self.__windowwidth-1)/2+1) > fullsizewindow.shape[1]:
            xc = -(self.__windowwidth-1)/2-1+fullsizewindow.shape[1]
            
        print fullsizewindow[yc-(self.__windowwidth-1)/2 : yc+(self.__windowwidth-1)/2+1, \
                xc-(self.__windowwidth-1)/2 : xc+(self.__windowwidth-1)/2+1].shape
        fullsizewindow[yc-(self.__windowwidth-1)/2 : yc+(self.__windowwidth-1)/2+1, \
                xc-(self.__windowwidth-1)/2 : xc+(self.__windowwidth-1)/2+1] = smallwindow
        mtf2d = FFT_2D(psf*fullsizewindow, sz = 256)
        '''
        fig = pylab.figure(dpi = 150, figsize = (10, 10))
        gs = pylab.matplotlib.gridspec.GridSpec(1, 3)
        ax0 = pylab.subplot(gs[0])
        ax1 = pylab.subplot(gs[1])
        ax2 = pylab.subplot(gs[2])
        ax0.imshow(psf, cmap = 'jet', interpolation = 'nearest')
        imh = ax1.imshow(psf * fullsizewindow, cmap = 'jet', interpolation = 'nearest')
        ax2.imshow(mtf2d, cmap = 'gray', interpolation = 'nearest')
        ax0.set_title('PSF before applying window')
        ax1.set_title('PSF after applying window')
        ax2.set_title('OTF(magnitude)')
        '''
        #fig.subplots_adjust(top = 0.9, bottom = 0.1)
        #cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.5])
        #fig.colorbar(imh, cax=cbar_ax)
        
        return psf * fullsizewindow
    def __CalculateMTF(self,**kwargs):
        self.mtf2d = FFT_2D(self.psf, sz = 256)
        freq, mtf = fft2DRadProf(np.fft.fftshift(self.mtf2d), self.__pixelspacing)
        normalize = lambda mtn: (mtn - mtn.min()) / (mtn[0] - mtn.min())
        mtf = normalize(mtf)
        fq = freq_at_mtf_perc(freq, mtf, 0.1)
        self.mtf = mtf
        self.freq = freq
        self.resolution10 = fq
        fq = freq_at_mtf_perc(freq, mtf, 0.5)
        self.resolution50 = fq
       
       
def MTF_show(r10, r50, freq, mtf):#self,
    #pylab.figure()
    #pylab.imshow(bead.psf, cmap='gray', interpolation='nearest')
    #pylab.figure()
    #pylab.imshow(bead.mtf2d, cmap='gray')
    figurehandle = pylab.figure(dpi=100, figsize=(8, 6))
    pylab.plot(freq, mtf)
    pylab.text(r10, 0.1, '10%% MTF = %.3f' % (r10))
    pylab.text(r50, 0.5, '50%% MTF = %.3f' % (r50))
    pylab.plot(r10, 0.1, '+r')
    pylab.plot(r50, 0.5, '+r')
    pylab.xlabel('frequency/(lp/cm)')
    pylab.ylabel('mtf')
    pylab.title('MTF from Tungsten Wire/Bead')
    #figurehandle.savefig('window width = %d' % windwidth)
    #pylab.ion()
    pylab.show()
    
if __name__ == "__main__":
    import re
    def alphanum_key(s):
        return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]
    pname = "D://pythonct//py_codenew//19220674-GE-RGRMS-E//2-5.0mm//"#U0000003"
    fnames = sorted(os.listdir(pname),key = alphanum_key)
    if 1:#for f in fnames[0:1]:
        fname = "Z\\Z09"#os.path.join(pname, f)
        print fname
        fname = fname.replace('/', '\\\\')
        windwidth = 9
        bead = TungstenBead(#dcm.pixel_array, float(dcm.PixelSpacing[0]),
                            fname,roiwidth=32,
                            windowwidth=windwidth,
                            #bead_position=[254,313],
                            )
        print "bead.resolution10*10", bead.resolution10*10
        print "bead.resolution50*10", bead.resolution50*10
        print "bead.beadpositionx", bead.beadpositionx
        print "bead.beadpositiony", bead.beadpositiony
        MTF_show(bead.resolution10*10, bead.resolution50*10, bead.freq * 10, bead.mtf)
