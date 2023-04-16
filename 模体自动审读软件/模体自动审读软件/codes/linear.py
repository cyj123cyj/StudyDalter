import os
import pydicom as dicom
import scipy
import math
import numpy as np
#from _canny_edge import canny
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import distance_transform_edt, label
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage.filters import gaussian_filter
import util
from util import canny
from util import avg_CT_number
from util import find_CT_phantom_outer_edge
#from misc import find_CT_phantom_outer_edge

# this depends on the design of the phantom
### we determined this number empirically
##DISK_ENCLOSE_RODS = 3190. #mm^2
### changed by Yahui 08/28/17
### due to the fact that the design of the phantom was changed
##DISK_ENCLOSE_RODS = 800. #mm^2
##ROD_AREA = 140. #mm^2
##ROD_AREA = 20. #mm^2
##DISTANCE_ROD2CENTER = 50. #mm^2
##
# below are the accurate numbers
ROD_RADIUS = 7.5 # mm  the designed size is 7.5 mm
PERCENTAGE_ROD = 0.64 # the percentage of the ROD area in CT value calculation, 80% as instructed in WS 519-2016
ROD_AREA = ROD_RADIUS*ROD_RADIUS*scipy.pi # mm^2
DISTANCE_ROD2CENTER = 50. #mm
# there is another design, where the distance between the rod center and the phantom center is 55 mm
# therefore, it is safe to extend the distance by 5 mm.
# and the extra 2 mm is to ensure rods are all included for sure
# the same tolerate limit is used to filter out the rod for MTF
TOLERATE_LIMIT = 6. # mm
RADIUS_ENCLOSE_RODS = (DISTANCE_ROD2CENTER + ROD_RADIUS + 5 + TOLERATE_LIMIT) # mm


def find_rod_locations(image, pixelspacing):
    """
    determine the locations of the 8 rods in the CT value linearity phantom
    assuming the geometric parameters are known, one may identify rod edges first inside a circle
    not all rod edges can be identified because the contrast may not be large enough
    since we know the rods are distributed evenly on a circle, we may fill in the missing rods
      by checking the angles of rods relative to the center of the phantom
    the locations of the rods are returned
    """
    # 8 rods distributed evenly on a circle
    DEBUG = False
    # pixel size in mm2
    pixelarea = pixelspacing*pixelspacing
    # calculate the detection area (the disk area containing rods) and rod area in pixels
    detection_dist = RADIUS_ENCLOSE_RODS / pixelspacing
    rod_size = ROD_AREA / pixelarea * .5 # .9 is a factor to ensure the rods will be detected even with defects
    
    rod2center_distance = DISTANCE_ROD2CENTER / pixelspacing
    if DEBUG:
        print "disk radius that includes the rods: %f (pixels), %f (mm)"%(detection_dist, detection_dist * pixelspacing)
        print "minimum area of a rod: %f (pixels), %f (mm^2)"%(rod_size, ROD_AREA)
        rp, rm = [math.sqrt(e/scipy.pi) for e in [rod_size, rod_size*pixelarea]]
        print "    effective radius: %f (pixels), %f (mm)"%(rp, rm)
        print "distance from the phantom center to the rods: %f (pixels), %f (mm)"%(rod2center_distance, rod2center_distance*pixelspacing)
    # get the cener of the phantom
    xc, yc, r,FUBU = find_CT_phantom_outer_edge(image,pixelspacing, return_coors = False)
    # this has to be fixed, a temporary solution
##    from water import Water_Phantom
##    phan = Water_Phantom(dcm_image_file)
##    outer = phan.get_phantom_outer_boundary()
    
##    outer = image > 100
##    ys, xs = scipy.where(outer)
##    yc, xc = ys.mean(), xs.mean()
##    r = math.sqrt(outer.sum() / scipy.pi)
##    print "center of the phantom:" ,xc ,yc
##    print "radius of the outer boundary:", r

    # find a restricted area that contains the rods
    # a mask that hightlight the restricted area
    h,w = image.shape
    mask = scipy.mgrid[0:h, 0:w]
    mask = scipy.hypot((mask[0] - yc), (mask[1] - xc)) < detection_dist

    # canny edge filtering inside the restricted area
    edge = canny(image, sigma = 2, low_threshold=50, high_threshold=100)
    edge *= mask
    if DEBUG:
        import pylab
        pylab.imshow(edge)
        pylab.show()
    
    # identify a few regions (rod candidates) through the closed edges
    lb, nlb = label(edge == 0)
    centers = center_of_mass(edge==0, lb, index=range(2, nlb+1))  # not including background and edges
    sizes, le = scipy.histogram(lb.ravel(), bins = range(2, nlb+2))#count no background and edges
    if DEBUG:
        print centers
        print sizes
        print "rod size threshold: %s"%rod_size
        print "rod center to phantom center threshold: %s"%rod2center_distance
        pylab.imshow(lb >= 2)
        pylab.show()
        pylab.imshow(lb)
        pylab.show()

    # rank the rods by size descendingly
    inds = scipy.argsort(sizes)[::-1]
    szs = sizes[inds] # region sizes
    les = le[inds]    #   and label IDs are sorted
    # filter the regions by size (if too small then not a rod)
    inds = scipy.where(szs > rod_size)[0]
##    print len(inds)
##    print inds
    rod_cand_ids = les[inds]
    
    rod_rs, rod_angles, rod_ids = [],[],[]
    for i in rod_cand_ids:
        #print i
        c_ind = i-2  # the indices of center of mass and region sizes are not the same as the region label
        r = scipy.hypot(centers[c_ind][0]-yc, centers[c_ind][1]-xc)
##        # was used to check the accuracy of the rod positions
##        print "%8.2f mm or %8.2f pixels"%(r*pixelspacing, r)
        # check whether it is on the circle
        # error tolerant limit is set to 2 mm
        if DEBUG:
            print "distance to the center:", r
            print "expected distance:", rod2center_distance
            print "tolerance:", TOLERATE_LIMIT/pixelspacing
        if scipy.absolute(r-rod2center_distance) > TOLERATE_LIMIT/pixelspacing:
            continue
        if DEBUG:
            print "the distance from rod center to phantom center is", r
            print "the difference from the expected value is", scipy.absolute(r-rod2center_distance)
            print "the effective rod radius is ", math.sqrt(sizes[c_ind]/scipy.pi)
        t = math.atan2(centers[c_ind][0]-yc, centers[c_ind][1]-xc)## * 180 / scipy.pi
        rod_rs.append(r)
        rod_angles.append(t)
        rod_ids.append(i)
        if DEBUG:
            print centers[c_ind], sizes[c_ind],
            print r, t
            pylab.imshow(lb == i)
            pylab.show()

    if DEBUG:
        print "rod centers to the phantom center:"
        print rod_rs
        print "angular position of rod centers:"
        print rod_angles
        print "rod label IDs:"
        print rod_ids

    # rank the rods by the angular position
    inds = scipy.argsort(rod_angles)
    rod_rs = scipy.array(rod_rs)[inds]
    rod_angles = scipy.array(rod_angles)[inds]
    rod_ids = scipy.array(rod_ids)[inds]
    
    # determine which rods are missed
    rod_rs=list(rod_rs)
    rod_angles =list(rod_angles)
    if DEBUG:
        print "sorted rod positions:"
        print rod_rs
        print rod_angles
        
    while rod_angles[0] > -135/180.*scipy.pi:
        rod_angles.insert(0, rod_angles[0] - 45/180.*scipy.pi)
        rod_rs.insert(0, rod_rs[0])
    while rod_angles[-1] < 135/180.*scipy.pi:
        rod_angles.append(rod_angles[-1]+45/180.*scipy.pi)
        rod_rs.append(rod_rs[-1])
        
    diff_angle = [e[0] - e[1] for e in zip(rod_angles[1:], rod_angles[:-1])]
    newloc = scipy.where(scipy.array(diff_angle) > 60/180.*scipy.pi)
    if DEBUG:
        print "angle difference:"
        print diff_angle
        print "location indices of missing rods:"
        print newloc
    for i in newloc[0][::-1]:
##        print i
##        if i == 0:
##            break
        r = (rod_rs[i]+rod_rs[i+1])/2
        a = (rod_angles[i]+rod_angles[i+1])/2
        rod_rs.insert(i+1, r)
        rod_angles.insert(i+1, a)
    if DEBUG:
        print "All rods (in polar coordinates):"
        print rod_rs
        print rod_angles
        print len(rod_angles)
        print len(rod_angles) > 8

    # in case there are more than 8 locations, there must be repeated rod centers
    # we just need to detect them and then remove them
    if len(rod_angles) > 8:
        diffs = scipy.array(rod_angles[1:]) - rod_angles[:-1]
        inds = scipy.where(diffs < scipy.pi/6)[0]
        for ind in inds[::-1]:
            new_angle =  (rod_angles[ind]+rod_angles[ind+1])/2.
            new_rod_r = (rod_rs[ind] + rod_rs[ind+1])/2.
            rod_angles.pop(ind+1)
            rod_rs.pop(ind+1)
            rod_angles[ind] = new_angle
            rod_rs[ind] = new_rod_r

    # modified 5/17/2019
    # in case there are more than 8 rod locations
    # the above code can only remove repeated rods whose angles are too close
    # but cannot exclude rods whose angle appear twice at both pi and -pi
    if len(rod_angles) > 8:
        rod_angles = rod_angles[:8]
        rod_rs = rod_rs[:8]

    if DEBUG:
        print rod_angles
        print rod_rs

    rod_locations = (xc + rod_rs * scipy.cos(rod_angles),
                     yc + rod_rs * scipy.sin(rod_angles))
    x, y = rod_locations

    if DEBUG:
        # notice that, by default, the origin of the image is at the "upper" left
        pylab.imshow(image, cmap=pylab.cm.gray, origin='lower')
        pylab.plot(x, y ,'rx')
        pylab.show()

    return x, y



# alias of the function
#avg_CT_number = avg_CT_number

class Linearity_Phantom:
    def __init__(self, dicom_image_file):
        """
        The dicom image is assumed to image a water phantom
        """
        self.dicom_file = dicom_image_file
        try:
            ds = dicom.read_file(dicom_image_file)
        except InvalidDicomError:
            #print "Please specify a valid DICOM image file!"
            return
        self.dataset = ds
        #print "dicom slope and intercept:", ds.RescaleSlope, ds.RescaleIntercept
        self.image_array = ds.pixel_array*ds.RescaleSlope+ds.RescaleIntercept
        self.pixel_spacing = ds.PixelSpacing[0]
##        pylab.imshow(self.image_array)
##        pylab.show()
        
    def get_phantom_center(self):
        xc, yc, r ,FUBU= find_CT_phantom_outer_edge(self.image_array,self.pixel_spacing, return_coors = False)
        #print "center of the phantom: ", xc, yc
        return xc, yc, r
    def get_rod_locations(self):
        x, y = find_rod_locations(self.image_array, self.dataset.PixelSpacing[0])
        #x, y = find_rod_locations(self.dicom_file, self.dataset.PixelSpacing[0])
        return x, y
    
    def get_material_CT_values(self):
        xc, yc, phantom_r = self.get_phantom_center()
        rod_locx, rod_locy = self.get_rod_locations()
        img = self.image_array

        CT_values = []
        rod_radius = ROD_RADIUS/self.dataset.PixelSpacing[0] # in pixels
        effective_radius = rod_radius * math.sqrt(PERCENTAGE_ROD) # ROI is smaller than the actual ROD size
        for i in range(len(rod_locx)): # should be 8
            CT, std = avg_CT_number(self.image_array, (rod_locx[i], rod_locy[i]), effective_radius)
            CT_values.append(CT)
            #print "CT value = %8.2f; std = %8.2f"%(CT, std)

        return CT_values
    
class DICOM:
    def __init__(self, pn, seq, im, studyID):
        self.pn = pn
        self.seq = seq
        self.im = im
        self.studyID = studyID
    def Read(self, displayImage = False):
        dcm_dir = dicom.read_dicomdir(os.path.join(self.pn, "DICOMDIR"))
        seqs = dcm_dir.DirectoryRecordSequence
        for seqInd in range(len(seqs)):
            if seqs[seqInd].DirectoryRecordType == 'STUDY' and int(seqs[seqInd].StudyID) == self.studyID:
                for i in range(seqInd, len(seqs)):
                    if seqs[i].DirectoryRecordType == 'SERIES' and int(seqs[i].SeriesNumber) == self.seq:
                        #print 'Found the Series'
                        for j in range(i, len(seqs)):
                            #print 'Examing#%d '%j, seqs[j].DirectoryRecordType
                            if seqs[j].DirectoryRecordType == 'IMAGE' and int(seqs[j].InstanceNumber) == self.im:
                                #print 'Found the Image, Seq #%d' %j
                                fname = os.path.join(self.pn, os.path.join(*seqs[j].ReferencedFileID))
                                return fname
if __name__ == '__main__':
    # first, find a suitable CT image that must include the linearity test phantom
    #pn = 'D:\\Research\\INET\\Du_Guosheng\\data\\validation_dataset\\hepingli_hospital\\4-19\\4-19\\BSoft'
    #pn = '..\\BSoft'
    # seq = 1
    # im = 1
    # studyID = 1
    # displayImage = True
    #
    # dicomIns = DICOM(pn, seq, im, studyID)
    # fname = dicomIns.Read(True)

    pn = "D:\Research\INET\Du_Guosheng\data\\11-4.8\\11-4.8"
    pn = "D:\\Research\\INET\\Du_Guosheng\\data\\19220674-GE-RGRMS-E\\19220674-GE-RGRMS-E\\2-5.0mm"
    files = os.listdir(pn)
    fnames = [os.path.join(pn, e) for e in files]
    for fname in fnames[10:11]:
        print fname
        # instantiate first
        test = Linearity_Phantom(fname)
        # then call a function to get all the CT values of the 8 rods
        x = test.get_material_CT_values()# x include the eight CT value
        # print out the CT values sorted for the lowest to the highest
        print sorted(x)
