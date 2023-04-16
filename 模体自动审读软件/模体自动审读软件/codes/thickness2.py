# -*- coding: utf-8 -*-
import scipy
from scipy.signal import butter, lfilter
from scipy.ndimage import label,filters
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

##from scipy.stats import norm#multivariate_normal
from math import ceil, sqrt

import pylab

from util import find_CT_phantom_outer_edge, canny

def Gauss(x, a, x0, sigma, b):
    return a * scipy.exp(-(x - x0)**2 / (2 * sigma**2)) + b
def fit_Gauss(x, y):
    mn = x[scipy.argmax(y)]#sum(x*y)/sum(y)
    sig = scipy.sqrt(sum(y*(x-mn)**2)/sum(y))
    if sig>0.5:
        sig = 0.5
    bg = min(y)
    try:
        popt, pcov = curve_fit(Gauss, x,y, p0=[max(y), mn, sig, bg])
        print "mn = "
        print x,y,mn
    except RuntimeError:
        return None
    return popt, pcov
def FWHM_Gauss(sigma):
    return 2.355*abs(sigma)


class SpiralBeads:
    # phantom geometry
    # key: value = spiral bead pitch: number of beads for a full 2pi circle
    # the sprial bead pitch is in millimeter.
    NUMBER_BEADS = {90: 180,
                    50: 250}

    # modified the parameter list so that only one group of beads
    #   is used for slice thickness calculation
##    def __init__(self, phantom, type_A=False, **kwargs):
    def __init__(self, phantom,
                 diameter = 166.3,    # diameter = 162.4  changed on 2018/11/30
                 pitch=90,
                 number_beads = 180,
                 **kwargs):
        self.NUMBER_BEADS[pitch]=number_beads
        self.phantom = phantom
        self.sliceThickness = self.phantom.dicom.SliceThickness

        # number of samples for the spiral beads
        self.number_samples = 5000*5
##        if type_A:
##            self.rou = 80.5 / self.phantom.dicom.PixelSpacing[0]  # for A model
##            self.ditch_kernel = 200 * self.number_samples/25000
##            self.ditch_width = 350 * self.number_samples/25000
##        else:
##            self.rou = 81.2 / self.phantom.dicom.PixelSpacing[0]  # for E model
##            self.ditch_kernel = 360 * self.number_samples/25000
##            self.ditch_width = 450 * self.number_samples/25000
        self.rou = diameter/2.0 / self.phantom.dicom.PixelSpacing[0]  # for E model
        self.pitch = pitch
        self.ditch_kernel = 360 * self.number_samples/25000
        self.ditch_width = 450 * self.number_samples/25000
        
        self.dis2radRatio = None
        self.profile = self.get_profile(False)
##        print "self.profile profile shape = ",self.profile['profile'].shape

        # This is to remove a dark bar in the same ring of the beads
        self.dark_tag = self.remove_dark_tag_in_profile()
    
    def correlate_profile_with_template(self, bead_pitch=90, full=True, profile=None):
        # to enable process other signals, instead of the profile data stored in this instance
        if profile is None:
            profile = self.profile['profile']
        # the length of the signal
        n = len(profile)
        # to avoid the edge effect
        # because the signal is circular, we can append the signals at the beginning of the end
        profile = scipy.r_[profile[-n/2:], profile, profile[:n/2]]
        # the template to do the correlation
        temp = self.generate_template(bead_pitch, full=full)
        # crop and return the correlation coefficients
        return scipy.ndimage.filters.correlate1d(profile, temp)[n/2:-n/2]
    
    def smooth_correlation_curve(self, curve, width):
        # Gaussian filter to smooth the curve
        # width should be given as the kernel size
        smoothed = scipy.ndimage.filters.gaussian_filter1d(curve, width)
        return smoothed
    
    def locate_beads(self):
        """
        """
        DEBUG = False
        profile = self.profile['profile']
        # to avoid the edge effect
        n = len(profile)
        
        if DEBUG:
            print "length of the profile:", n
            pylab.plot(profile)
            pylab.show()
        profile = scipy.r_[profile[-n/2:], profile, profile[:n/2]]
        # this is the number of samples to be removed from the filtered sequence
        #   to ensure the vlaue of the number_samples would not affect the results
        #   we compensate for different number_samples, in other words
        #   the 1000 points to removed only when the number of samples is 25000
        n_edge = int(round(1000/25000.*self.number_samples))
        profile = self.remove_profile_background(profile)[n_edge:]
        # the above method may not be necessary
        # if we can do a circular filtering

        if DEBUG:
            pylab.plot(profile)
            pylab.show()

        #combine the correlation between the profile and the two pitches
        corr90 = self.correlate_profile_with_template(bead_pitch=90, full=True, profile=profile)
        corr50 = self.correlate_profile_with_template(bead_pitch=50, full=True, profile=profile)
        corr = corr90+corr50
        if DEBUG:
            pylab.plot(corr90/corr90.max())
            pylab.plot(corr50/corr50.max())
            pylab.plot(corr/corr.max())
##            corr_diff = scipy.absolute(corr90 - corr50)
##            pylab.plot(corr_diff/corr_diff.max())
            pylab.show()
        # to locate the groupd of beads
        #  first to get the local maxima,
        #     the constant 300 is related to the number of smaples
        #     notice that the first step also eliminate the gaps between beads
        #  second, smooth the loc_max curve
        #  then get the local maxima again
        width = float(self.number_samples)/250
##        width = float(300)/25000 * self.number_samples
        loc_max = scipy.ndimage.filters.maximum_filter(corr,width)
        smoothed = self.smooth_correlation_curve(loc_max, width)
        sm_max = scipy.ndimage.filters.maximum_filter(smoothed,
                                                      float(self.phantom.dicom.SliceThickness)/50*self.number_samples)

        # remove the extra parts
        smoothed = smoothed[self.number_samples/2-n_edge:-self.number_samples/2]
        sm_max = sm_max[self.number_samples/2-n_edge:-self.number_samples/2]
##        corr = corr[self.number_samples/2-n_edge:-self.number_samples/2] # need to use it later on
        corr90 = corr90[self.number_samples/2-n_edge:-self.number_samples/2]
        corr50 = corr50[self.number_samples/2-n_edge:-self.number_samples/2]

        if DEBUG:
            pylab.plot(loc_max/loc_max.max())
            pylab.plot(smoothed/smoothed.max())
            pylab.plot(sm_max/sm_max.max())
            pylab.show()
        
        t = (smoothed.max() - smoothed.min())/10. + smoothed.min()
        indices = scipy.where(scipy.logical_and(sm_max == smoothed, smoothed > t))[0]

        # check the gaps between the indices
        #   if the gap is too small, the nearby indices should be merged as one
        if len(indices) > 1:
            # the threshold of the gap that need to be merged
            critical_v = self.phantom.dicom.SliceThickness/50*self.number_samples
            critical_v *= 2
            merged_ind = []
            flag = False
            for i in range(len(indices)):
                if not flag:
                    cur_ind = indices[i]
                    flag = True
                else:
                    if cur_ind + critical_v > indices[i]:
                        # use one index, instead of the average
                        #   since the averaged index may not have the
                        #   right correlation coefficient
                        #   (can be lower than either of the two)
##                        merged_ind.append((cur_ind + indices[i])/2)
                        merged_ind.append(indices[i])
                        flag = False
                        continue
                    else:
                        merged_ind.append(cur_ind)
                        cur_ind = indices[i]
                if i == len(indices)-1:
                    merged_ind.append(indices[i])
            indices,merged_ind = merged_ind, indices

        # to determin which one is the group of the 90-pitched beads
        max90 = scipy.ndimage.filters.maximum_filter(corr90,width)
        smth90 = self.smooth_correlation_curve(max90, width)
        max50 = scipy.ndimage.filters.maximum_filter(corr50,width)
        smth50 = self.smooth_correlation_curve(max50, width)
        if DEBUG:
            theta = self.profile['theta']
            pylab.plot(theta, smth90, 'r')
            pylab.plot(theta, smth50)
            pylab.plot(theta, smth90 - smth50)
            pylab.show()
        if len(indices) == 2:
            # the correlation coefficient should be greater if the profile and the template are matched
            if (smth90[indices[1]] < smth90[indices[0]] and \
                smth50[indices[0]] < smth50[indices[1]]):
                pass
            elif (smth90[indices[1]] > smth90[indices[0]] and \
                  smth50[indices[0]] > smth50[indices[1]]):
                indices = indices[::-1]
            elif abs(smth90[indices[0]] - smth50[indices[0]]) < t:
                indices = indices[1:]
            elif abs(smth90[indices[1]] - smth50[indices[1]]) < t:
                indices = indices[:1]
            else:
                pass
        if len(indices) <= 2:
            pass
        elif len(indices) == 3:
            # this may happen when one of the beads in a group is missing due to some reason
            print indices
            print indices[1:] - indices[:-1]
            return None
        else:
            # something wrong
            print "Could not identify the location(s) of the beads!"
            return None

        # the coverage of the beads
        t = min([t, (corr.max() + 2*corr.std())/3])
        beads = corr[self.number_samples/2-n_edge:-self.number_samples/2] > t
        spread = scipy.float32(beads)
        spread = scipy.ndimage.filters.gaussian_filter1d(spread, self.number_samples/self.NUMBER_BEADS[50]*2)
        cover_range = spread > 0
        lb, nlb = scipy.ndimage.label(cover_range)

##        print "number of regions: ", len(indices)
        # to find the corresponding profile segments that contain the groups of beads
        ranges = []
        rois = []
        # each group of indices has a profile segment
        for i in range(len(indices)):
            # get a labeled range
            roi = (lb == lb[indices[i]])
            # find the corresponding indices that not equal to zero
            roi_ind = scipy.where(roi)[0]
            # the first and last indices
##            print roi_ind[0], roi_ind[-1]
            # if the first element is in the labeled range
            #   then it is possible the current labeled range is not complete
            #   so, we need to check whether there is a ranged also labeled
            #   and the other end of the profile (it is supposed to be a circular sequence
            if (0 in roi_ind):
                uind = roi_ind[-1]
                # find the lower boundary of the range
                try:
                    lind = scipy.where(lb == 3)[0][0]
                except:
                    print "nlb = %d"%nlb
                    if nlb < 3:
                        lind = 0
                # notice that in this case, the value of lind may be
                #   greater than that of uind
                ranges.append([lind, uind])
                # make the segment complete by combining the two pieces together
                roi = scipy.logical_or(lb == lb[indices[i]], lb == 3)
                roi_ind = scipy.where(roi)[0]
            elif (self.number_samples-1 in roi_ind) and scipy.where(lb == 1)[0][0] == 0:
                lind = roi_ind[0]
                uind = scipy.where(lb == 1)[0][-1]
                ranges.append([lind, uind])
                roi = scipy.logical_or(lb == lb[indices[i]], lb == 1)
                roi_ind = scipy.where(roi)[0]
            else:
                ranges.append([roi_ind[0], roi_ind[-1]])
            rois.append(roi)
        if DEBUG:
            for roi in rois:
                pylab.plot(roi*self.profile['profile'])
                pylab.show()
        return indices, rois

    def angle_restricted_ranges(self, bead_range):
        # get the maximum pixel value
        profile = self.profile['profile']
        ind = scipy.argmax(profile)

        lb, nlb = scipy.label(bead_range)
        for i in range(1,nlb+1):
            pass

    def remove_dark_tag_in_profile(self):
        """
        there is a dark tick in the profile,
        which may cause problems when doing correlation
        """
        DEBUG = False
        #width of the tick
        #  this is determined empirically
        #  and it is why the calculation equation is weird
        width = self.ditch_width*2
        kernel = self.ditch_kernel
        profile = self.profile['profile'].copy()
        ind = scipy.argmin(profile)
        lind = ind - width/2
        uind = ind + width/2
        if lind < 0:
            org = scipy.r_[profile[lind:], profile[:uind]]
        else:
            org = profile[lind:uind]
        if DEBUG:
            print "length of 'org':", len(org), "minimum value:", org.min()
            print "indices: %s to %s"%(lind, uind)
            pylab.plot(profile)
            pylab.show()
        filtered = scipy.ndimage.filters.maximum_filter(org, kernel)# kernel = 200
        if DEBUG:
            pylab.plot(org)
            pylab.plot(filtered)
            pylab.show()
        indices = scipy.where(filtered == org)[0]
        if DEBUG:
            print "the indices where the org & maximum signals meet:", indices
        rlind = indices[0] + lind
        if DEBUG:
            print "rlind = %d"%rlind
        if rlind < 0:
            # the changed part is split to both end
            if DEBUG:
                print "beginning & ending indices in the second half of 'filtered' signal: ", -lind, indices[-1]
                print "beginning & ending indices in the original 'profile' signal: ", indices[-1]+lind, rlind
                print "beginning & ending indices in the first half of 'filtered' signal:", indices[0], -lind
            profile = scipy.r_[filtered[-lind:indices[-1]],
                               profile[indices[-1]+lind : rlind],
                               filtered[indices[0]:-lind],]
        else:
            # otherwise
            profile = scipy.r_[profile[:rlind],
                               filtered[indices[0]:indices[-1]],
                               profile[indices[-1]+lind:]]
        if DEBUG:
            print "after remove the ditch, length of profile:", len(profile), "minimum value:", profile.min()
        self.profile['profile'] = profile
        if DEBUG:
            pylab.plot(self.profile['profile'], 'g')
            pylab.plot(profile, 'r')
            pylab.show()
        return [lind, uind, org]

    def angle2coor(self, angle, rho, center_coor, as_index=False):
        # convert polar angle to cartesian coordinates
        #   This controls where the starting point is in the CT phantom image
        #   currently, the zero angle is located at the right-most point of a circle
        yc, xc = center_coor
        x = xc + rho * scipy.cos(angle)
        y = yc + rho * scipy.sin(angle)
        # because the coordinates may be used to do subpixel sampling
        # here, the coordinates can be float values
        # but you have an option to convert them to image pixel indices
        if as_index:
            return (scipy.uint16(scipy.round_(y,0)),
                    scipy.uint16(scipy.round_(x,0)))
        else:
            return (y, x)
        
    def get_profile(self, displayImage = False):
        # rename variables for convenience
        pa = self.phantom.dicom.pixel_array
        rad = self.rou
        xr, yr = self.phantom.center_x, self.phantom.center_y
        # need a profile to get degMax (position of the brightest bead)
        thetaPlt = scipy.linspace(-scipy.pi , scipy.pi, self.number_samples)
        profile = []
        # in the beginning, the precision of the model is not good
        # therefore, used to have average over different radii
        # but the current phantom is good enough
        # therefore, no need to do averaging any more

        # to romve possible inaccuracy in bead-mounting
        #    and to romve the air gaps in the profile
        #    using the maximum to replace the mean 06/19/2018
##        offsets = [0]#[-1, 0, 1]
        offsets = [-1, 0, 1]
        for off in offsets:
            y, x = self.angle2coor(thetaPlt, rad+off, (yr, xr), as_index=False)
            # scipy provides below function to do interpolation
            # using spline interpolation with the order of 3
            # when an edge is encountered, the "wrap" mode is used
            pf = scipy.ndimage.map_coordinates(pa, scipy.vstack((y,x)),
                                               order=3, mode='wrap')
            profile.append(pf)
        profile = scipy.array(profile)
        if displayImage:
            print "the shape of the profile:", profile.shape
            for i in range(profile.shape[0]):
                pylab.plot(profile[i])
            pylab.show()
##        profile = profile.mean(axis=0)
        profile = profile.max(axis=0)
##        print "profile shape =", profile.shape
                
        if displayImage:      
            pylab.figure()
            pylab.imshow(pa, interpolation="nearest", cmap='gray', origin='lower')
            pylab.plot(x, y, 'g.',markersize=1, linewidth=1)
            pylab.show()
            #print "profile mean = %s"%(profile.mean())
            pylab.plot(thetaPlt, profile)#scipy.ndimage.filters.median_filter(profile,5))
##            b,a = butter(1, 0.05, btype='high')
##            measure = lfilter(b,a, profile)
##            pylab.plot(thetaPlt, profile - measure)

            profile = scipy.ndimage.filters.median_filter(scipy.ndimage.filters.median_filter(profile,10),10)
            pylab.plot([thetaPlt.min(), thetaPlt.max()], [(profile.mean()*4 + profile.max())/5]*2)
            pylab.plot([thetaPlt.min(), thetaPlt.max()], [(profile.mean()*2 + profile.max())/3]*2)
            pylab.plot([thetaPlt.min(), thetaPlt.max()], [(profile.mean()*1 + profile.max()*2)/3]*2)
            inds = scipy.where(profile > (profile.mean()*2 + profile.max())/3)[0]
            print inds[-1], inds[0], inds[-1]-inds[0]
            print thetaPlt[inds[-1]], thetaPlt[inds[0]], thetaPlt[inds[-1]]-thetaPlt[inds[0]],\
                (thetaPlt[inds[-1]]-thetaPlt[inds[0]])/scipy.pi*50
            pylab.show()
        return {'theta': thetaPlt, 'profile': profile}

    def generate_template(self, bead_pitch=90, full=True, phase_shift=False):
        """
        generate template for the template matching method
        """
        # 90mm: w=180, 50mm: w=250
        w = self.NUMBER_BEADS.get(bead_pitch, None)
        # rename the variables for convenience and clarity
        number_samples = self.number_samples
        nominal_thickness = self.phantom.dicom.SliceThickness
        nominal_angle_percentage = nominal_thickness / float(bead_pitch)
        # number of sample points covering the beads for the nominal thickness
        number_points = int(number_samples * nominal_angle_percentage )# * 1.1)
        # if it contains partial cycles, make it full cycles
        # this is to ensure the correlation of random noise is close to zero or the mean
        number_points = ceil(number_points / (float(number_samples)/w)) * (float(number_samples)/w)
        number_points = int(number_points)

        # get the angle range for template generation
        thetaPlt = self.profile['theta'][:number_points]
        # sinusoidal function with or without phase shift
        if phase_shift:
            template = scipy.sin(w * thetaPlt-scipy.pi/2.)
        else:
            template = scipy.sin(w * thetaPlt)#-scipy.pi/2.)
        # if use only one period
        if not full:
            template = template[:number_samples/w+1]
        return template

    def get_thickness(self, pitch, bead_profile):
        # in case we need to debug the program
        # we may turn it on
        DEBUG = False
        num_beads = self.NUMBER_BEADS[pitch]
        # rename the variable
        target_roi = bead_profile
        # the bead profile need to be extracted
        #   irrelevant profile was set to zero
        indices = scipy.where(target_roi != 0)[0]
        # crop the profile segment of interest
        inds, inde = indices.min(), indices.max()
        # in case the segment of interest is split into two parts
        #   at the ends of the profile
        #   then shift the profile to get the whole segment
        if inds == 0 and inde == self.number_samples-1:
            roi_temp = target_roi*self.profile['profile']
            roi_temp = scipy.r_[roi_temp[-len(roi_temp)/2:],
                                roi_temp[:len(roi_temp)/2]]
            indices = scipy.where(roi_temp != 0)[0]
            inds, inde = indices.min(), indices.max()
            roi = roi_temp[inds:inde+1]
            # set a flag to indicate that the profile needs to be shifted
            shifted = True
        else:
            # otherwise
            roi = self.profile['profile'][inds:inde+1]
            shifted = False
        # get the peaks of each bead profile
        # first, remove the baseline
        diff = roi - scipy.ndimage.minimum_filter1d(roi, int(self.number_samples/num_beads*1.3))
        # then, get the local maxima
        max_curve = scipy.ndimage.maximum_filter1d(diff, int(self.number_samples/num_beads*1.35))
        # find the locations of the maxima
        #   I am not sure if this can be done using the scipy maximum_filter1d
        #   have no time to try it out
        #   need to do it when time allows
        #   that filter may get the unique maximum poiot for each peak
        #   current method is not perfect
        indx = scipy.where(max_curve == diff)[0]
        if DEBUG:
            pylab.plot(diff)
            pylab.plot(max_curve)
            pylab.plot(indx, max_curve[indx], 'ro')
            pylab.show()
        # remove adjacent maximum points
        #   this may not be necessary if each peak has one maximum point
        uindx = []
        flag = False
        cur_ind = indx[0]
        fst_ind = indx[0]
        # the idea is to check whether the location indices are continuous
        #   to ensure this method absorbs small bumps, a threshold (20 points here)
        #   is set to ignore maxima points whose distance is within 20 points
        for i in range(1,len(indx)):
            if cur_ind + 20 >= indx[i]: # to remove local maxima within 20 points
                # continue
                cur_ind = indx[i]
            else:
                # find a unique index
                uindx.append((cur_ind+fst_ind)/2)
                fst_ind = indx[i]
            cur_ind = indx[i]
        # now one bead peach has one unique index
        indx = scipy.array(uindx)
        if DEBUG:
            print "the indices of the bead peaks:"
            print indx
            print "the corresponding angles:"
            print self.profile['theta'][indx + inds]
        # now, we want to have the angles of each bead peak
        # in case the profile was shifted (the beads are around angle zero)
        if shifted:
            theta = self.profile['theta']
            theta = scipy.r_[theta[-len(theta)/2:], theta[:len(theta)/2]]
            theta = theta[indx + inds]
        else:
            # otherwise
            theta = self.profile['theta'][indx + inds]
        # with the angles and the expected radius
        # we ccccan get the pixels that contain the beads
        ys, xs = self.angle2coor(theta, self.rou,
                                 (self.phantom.center_y, self.phantom.center_x),
                                 as_index = True)
        # in a 3x3 neigborberhood, find the local maximum as the bead peak
        #   this is to compensate the inaccuracy in bead-mounting
        values = []
        for i, coor in enumerate(zip(ys, xs)):
            y, x = coor
            off = 1
            max_pv = (self.phantom.dicom.pixel_array[y-off:y+off+1, x-off:x+off+1]).max()
            values.append(max_pv)
    ##        off = 3 # this is for 5X5 neighborhood
    ##        pylab.imshow(self.phantom.dicom.pixel_array[y-off:y+off+1, x-off:x+off+1],
    ##                     cmap='gray', interpolation='nearest', origin='lower')
    ##        pylab.show()
    ##        if i > 10:
    ##            break
        thickness = []
        # fit a Gaussian curve to the bead peak, with background removed
        try:
            re = fit_Gauss(indx, diff[indx])
            if re is not None:
                popt, pcov = re
                if DEBUG:
                    print "estimated curve parameters:"
                    print "amplitude, mean, sigma, baseline"
                    print popt
                    print pcov
                # this can be treated as failure in curve fitting
                if pcov[2,2] < 0 or (sqrt(pcov[2,2]) > abs(popt[2])):
                    print "curve fitting for bead peaks (difference) probably not right!!!!!!!!!!!!!!"
        ##            print popt
        ##            print pcov
                else:
                    thick = FWHM_Gauss(popt[-2])/self.number_samples*pitch
                    thickness.append(thick)
                    if DEBUG:
                        print "Thickness = %s"%thick
                if DEBUG:
                    xx = scipy.linspace(indx[0], indx[-1], num=1000)
                    pylab.plot(xx, Gauss(xx, *popt))
        except:
            print "probably failed in curve fitting (Gaussian curve for bead peaks without background)!"
            print "you need to check the program to determine what went wrong!"
        # fit a Gaussian curve to the bead peak, with the background
        try:
            re = fit_Gauss(indx, values)
            if re is not None:
                popt, pcov = re
                if DEBUG:
                    print "estimated curve parameters:"
                    print "amplitude, mean, sigma, baseline"
                    print popt
                    print pcov
                if pcov[2,2] < 0 or (sqrt(pcov[2,2]) > abs(popt[2])):
                    print "curve fitting for bead peaks (original) probably not right!!!!!!!!!!!!!!"
        ##            print popt
        ##            print pcov
                else:
                    thick = FWHM_Gauss(popt[-2])/self.number_samples*pitch
                    thickness.append(thick)
                    if DEBUG:
                        print "Thickness = %s"%thick
                if DEBUG:
                    pylab.plot(roi)
                    pylab.plot(diff)
                    pylab.plot(indx, values,'o')
                    pylab.plot(indx, diff[indx],'.')
                    pylab.plot(xx, Gauss(xx, *popt))
                    pylab.show()
        except:
            print "probably failed in curve fitting (Gaussian curve for bead peaks with background)!"
            print "you need to check the program to determine what went wrong!"
        if len(thickness) != 0:
            return scipy.mean(thickness)
        else:
            return None
    def get_lthickness(self, profile,bc=0.625):
        DEBUG = True
        pitch = self.pitch
        spacing = self.phantom.dicom.PixelSpacing[0]
        #针对不同的标称值设定中值滤波器的参数和阈值的相对位置
        if bc <=0.55:
            a = 5
            k = 55/98.#141/223.
        elif bc <= 0.625:
            a =7
            k=127/223.
        elif bc<=1.1:
            a = 70
            k=55/108.
        elif bc <= 1.25:
            a = 150
            k=55/119.
        elif bc <= 2.2:
            a = 270
            k = 55/112.
        elif bc <= 5:
            a = 300
            k = 55/108.
        elif bc <= 5.5:
            a = 370
            k = 55/113.
        else:
            a = 400
            k = 55/115.

        pro = scipy.ndimage.median_filter(profile['profile'], int(a/spacing*0.44))#a
        mean = pro.mean()
        maxv = pro.max()
##        print maxv-mean
        if (maxv-mean) <= 60:#20200509,100
            print maxv-mean
            return None
        threshold = k*(mean + maxv)#k
##        threshold =  (pro.mean()*2 + pro.max())/3
        inds = scipy.where(pro > threshold)[0]
        #print threshold,inds
        print "arg:"
        print scipy.argmax(pro),len(pro)/2
        if inds[0] == 0:
            pro = scipy.ndimage.shift(pro, int(len(pro)/2), mode='wrap')
            inds = scipy.where(pro > threshold)[0]
        tht = profile['theta']
        
        span = (tht[inds[-1]] - tht[inds[0]])/scipy.pi*(pitch/2)
##        (thetaPlt[inds[-1]]-thetaPlt[inds[0]])/scipy.pi*50
        print "thickness = %s"%span
        thickness = span
#        FUBUMOTI = 1
        if abs(scipy.argmax(pro)-len(pro)/2)>len(pro)/5:
            pro = scipy.ndimage.shift(pro, int(len(pro)/2-scipy.argmax(pro)), mode='wrap')
        pro[scipy.where(pro<pro.mean())]=pro.mean()
        print pro.mean()
        if self.phantom.FUBU:
##            width = float(self.number_samples)/250
##            sigma = scipy.std(pro)
            popt,pcov = fit_Gauss(tht,pro)
##            print popt
            pro2 = Gauss(tht,*popt)
##            pro2 = norm.pdf(tht,tht.mean(),sigma)#self.smooth_correlation_curve(pro, width)
##            print pro2,pro
##            area = scipy.special.erf(pro2)
            threshold2 = 0.5*(pro2.max()+pro2.min())
            inds2 = scipy.where(pro2 > threshold2)[0]
            self.thickness2 = (tht[inds2[-1]] - tht[inds2[0]])/scipy.pi*(pitch/2)
            self.area2 = sum((pro2[inds2[0]:inds2[-1]]-pro2.min())/pro2.max())
            self.pro_max = pro2.max()
##            area,err = scipy.integrate.quad(Gauss,tht[inds2[0]] , tht[inds2[-1]],args = popt)#(
            print "area:"+str(self.area2)+"thickness2:"+str(self.thickness2)+"max:"+str(pro2.max())#"area:"+str(area)+
        if DEBUG:
            pylab.plot( pro)
            pylab.plot(pro2)
##            pylab.plot(profile['profile'])
            pylab.plot(scipy.ones(len(pro))*mean)
            pylab.plot(scipy.ones(len(pro))*maxv)
            pylab.plot(scipy.ones(len(pro))*threshold)
            pylab.show()
        return thickness
    
    def count_beads(self, bead_signal, show_plots=False):
        """
        """
        radius = 5
        inds = argrelextrema(bead_signal, scipy.less, order=radius)[0]
        print inds
        
        f = interp1d(inds, bead_signal[inds], kind='cubic')
        newx = scipy.linspace(inds[0], inds[-1], inds[-1]-inds[0]+1)
        newx = scipy.uint16(newx)
        baseline = f(newx)
        profile= bead_signal[newx] - baseline

        lb, nlb = label(profile > profile.max()/3)

        if show_plots:
            pylab.plot(baseline)
            pylab.plot(newx, bead_signal[newx])
            pylab.plot(newx, profile)
            pylab.plot([newx[0], newx[-1]], [profile.max()/3]*2)
            pylab.show()
        return nlb

    def remove_profile_background(self, signal):
        """
        remove the DC and low-frequency component
        """
        nsg = signal - signal.mean()
        b, a = butter(1, 0.005, btype = 'high')
        filtered = lfilter(b, a, signal)
        return filtered

    def extract_beads(self, bead_pitch=90, show_image=False):
        """
        template matching method
        """
        # move the fluctuation and zero-mean the profile
        measure = self.remove_profile_background(self.profile['profile'])
        
        templates = []
        corrs = []
        isbeads = []
        thicknesses = []
        signals = []
        for bead_pitch in (90 ,50):
            # create the template and zero-mean it
            template = self.generate_template(bead_pitch)
##            pylab.plot(template)
##            pylab.show()
            template -= template.mean()
            
            templates.append(template)
            
            corr = scipy.ndimage.filters.correlate(measure, template)
##            pylab.plot(measure/measure.max())
##            pylab.plot(corr/corr.max())
##            pylab.show()
            thr = (corr.max() + corr.std()*0)/2.
            isbead = corr > thr
            # remove the beads already be identified as 90mm-pitched beads
            if len(isbeads) > 0:
                isbead[scipy.where(isbeads[0] > 0)] = 0

            isbead = scipy.ndimage.filters.gaussian_filter1d(scipy.float32(isbead),
                                                             self.number_samples/125)
            
##            # remove the beads already be identified as 90mm-pitched beads
##            if len(isbeads) > 0:
##                isbead[scipy.where(isbeads[0] > 0)] = 0
            lb, nlb = scipy.ndimage.label(isbead)
            
            # if could not find any beads
            if nlb == 0:
                continue

            isbead *= (lb == lb[scipy.argmax(corr)])
            pylab.plot(corr/corr.max())
            pylab.plot(isbead/isbead.max())
            pylab.plot(measure/measure.max())
            pylab.plot(self.profile['profile']/self.profile['profile'].max())
            pylab.show()
            print isbead[0], isbead[-1]
            
            # if the region is at the end of the profile
            if isbead[0] > 0 or isbead[-1] > 0:
                print "now in the 'roll' branch"
                # shift the profile
                # need to repeat the process from the beginning
                # roll the profile
                sg = self.profile['profile']
                measure = self.remove_profile_background(scipy.roll(sg,sg.size/2))
                
                corr = scipy.ndimage.filters.correlate(measure, template)
                thr = (corr.max() + corr.std()*0)/2.
                isbead = corr > thr
                # remove the beads already be identified as 90mm-pitched beads
                if len(isbeads) > 0:
                    # have to roll the previous siganl by the same amount to match it
                    isbead[scipy.where(scipy.roll(isbeads[0],sg.size/2) > 0)] = 0
                    
                isbead = scipy.ndimage.filters.gaussian_filter1d(scipy.float32(isbead),
                                                                 self.number_samples/125)
                
                lb, nlb = scipy.ndimage.label(isbead)
                isbead *= (lb == lb[scipy.argmax(corr)])
                pylab.plot(corr/corr.max())
                pylab.plot(isbead/isbead.max())
                pylab.plot(measure/measure.max())
                pylab.plot(scipy.roll(sg, sg.size/2)/sg.max())
                pylab.show()

##            pylab.plot(measure/measure.max())
##            pylab.plot(isbead/isbead.max())
##            pylab.plot(corr/corr.max())
##            pylab.show()
            

            n = len(scipy.where(isbead > isbead.max()*.75)[0])
            thickness = float(n)/self.number_samples*bead_pitch
            thicknesses.append(thickness)
            isbeads.append(isbead)
            sgn = measure[scipy.where(isbead > 0)]
            num_beads = self.count_beads(sgn, True)
            signals.append(sgn)
            corrs.append(corr)

            sd = measure[scipy.where(isbead == 0)].std()
            lb, nlb = scipy.ndimage.label(scipy.absolute(sgn) > sd*2)
            if bead_pitch == 90:
                print "thicknes at 90mm is ", (num_beads-1)/2.
            else:
                print "thickness at 50mm is ",(num_beads-1)/5.

##        if ((isbeads[0])*(isbeads[1])).sum() != 0:
##            thicknesses.pop(-1)
        print thicknesses

        
        
##        pylab.imshow(self.phantom.dicom.pixel_array, cmap="gray")
##        pylab.show()
##        _ = pylab.plot(measure/measure.max())
##        _ = pylab.plot(isbeads[0]/isbeads[0].max())
##        _ = pylab.plot(isbeads[1]/isbeads[0].max())

##        pylab.plot(signals[0])
##
##        from scipy.signal import argrelextrema
##        radius = 2
##        inds = argrelextrema(signals[0], scipy.less, order=radius)[0]
##        from scipy.interpolate import interp1d
##        #print len(inds), len(signals[0][inds])
##        f = interp1d(inds, signals[0][inds], kind='cubic')
##        newx = scipy.linspace(inds[0], inds[-1], inds[-1]-inds[0]+1)
##        newx = scipy.uint16(newx)
##        baseline = f(newx)
##        pylab.plot(newx, signals[0][newx] - baseline)
##        
###        pylab.plot(scipy.ndimage.filters.correlate(signals[0], signals[0]))
##        if len(signals) > 1:
##            pylab.plot(signals[1])
###            pylab.plot(scipy.ndimage.filters.correlate(signals[1], signals[1]))
##        pylab.show()
        return thicknesses

SECTION_TYPE=[0,1]
GEOMETRY = {
    0: [# diameters
        161, # mm, the outer diameter
        110, # mm, the diameter of the circle where the 8 linearity rods locate
         15, # mm, the linearity rod diameter
         15, # mm, the MTF wire rod diameter
          3, # mm, the diameter of the 4 geometric distortion holes

        # distances
         90, # mm, pitch of the spiral beads
         32, # mm, the length of the hole modules
         10, # mm, the depth of the hole modules
          0, # mm, the distance from the center to the geometric distrotion holes, UNKNOWN
         30, # mm, the distance from the center to the MTF wire rod center
        ],
    1: [#
        161,#145,
        100,
         15,
         15,
          3,
        #
         70, # mm, pitch of the spiral beads
         32, #?
         10, #?
          0, # UNKNOWN
         25,
        ],
    2: [#
        161,#113, # ?
         90,
         12,
         12,
          3,
        #
         60, # mm, planned pitch of the spiral beads
         32, #?
         10, #?
          0, #UNKNOWN
          0, #UNKNOWN
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
                dcm = dicom.read_file(dcm_img, force=True)
            except:
                print "Not a dicom file: %s"%dcm_img
                return False
        else:
            dcm = dcm_img
        self.dicom = dcm
        self.image = dcm.pixel_array*dcm.RescaleSlope+dcm.RescaleIntercept

        self.section_type = self.get_section_type()

        re = find_CT_phantom_outer_edge(self.image,dcm.PixelSpacing[0], return_coors = True)
        self.center_x = re[0]
        self.center_y = re[1]
        self.outer_radius = re[2]
        self.outer_coor_xs = re[3]
        self.outer_coor_ys = re[4]
        self.FUBU = re[5]

        # find the structure
        if self.section_type == SECTION_TYPE[1]:
            self.determine_size()

    def get_section_type(self):
        """
        determine whether the phantom is the water or the comprehensive section

        since the water section has simpler structure, the edge pixels are less
        therefore, the number of edge pixels is used to tell difference
        """
        edges = canny(self.image, sigma = 2.0,
                      low_threshold=50, high_threshold=100)
        
        if edges.sum() < 7500:
            return SECTION_TYPE[0]
        else:
            return SECTION_TYPE[1]

    def determine_size(self):
        self.find_MTF_wire_rod()
        #self.find_rod_locations()

    def find_MTF_wire_rod(self):
        """
        the design of phantom can be characterized by the distance between
        the phantom center and the MTF wire rod center
        """
        DEBUG=False
        re = [self.center_x, self.center_y, self.outer_radius,
              self.outer_coor_xs, self.outer_coor_ys]
        xc, yc, r, xe, ye = re

        h,w = self.dicom.pixel_array.shape
        mask = scipy.mgrid[0:h, 0:w]
        # detect in this region to see where the wire is
        detection_dist = 40 #mm to cover the MTF wire rod
        detection_dist /= self.dicom.PixelSpacing[0]
        dist_map = scipy.hypot((mask[0] - yc), (mask[1] - xc))

        detection_zone = dist_map < detection_dist

        # to determine how to smooth the image
        #   with a high SNR, smaller kernel may be used
        std = scipy.std(self.image[scipy.where(detection_zone)])
        try:
            kernel = self.dicom.ConvolutionKernel
        except:
            kernel = None

        if kernel == "BONE" or std > 40:
            sigmaV = 3
        else:
            sigmaV = 1
        
        #print "using sigma = %s, std = %s"%(sigmaV, self.image[scipy.where(detection_zone)].std())
        edge = canny(self.image, sigma=sigmaV,
                     low_threshold=10, high_threshold=100)
        edge *= detection_zone

        # to find the largest region
        #   which can be assumed to be the wire rod
        lb, nlb = label(edge == 0)
        #print nlb
        if nlb == 1:
            # could not detect the MTF wire rod
            print "Could not detect the MTF wire rod!"
            return
        hist, le = scipy.histogram(lb, bins=range(2, nlb+2))
        ind = scipy.argsort(hist)[-1] + 2
        rod = lb == ind

        # the distance between the center of the MTF wire and the center of the phantom
        rodyc, rodxc = [scipy.mean(e) for e in scipy.where(rod)]
        dist_cc = scipy.hypot(xc - rodxc, yc - rodyc)
        dist_cc_mm = dist_cc*self.dicom.PixelSpacing[0]
        #print "distance between the MTF rod and the center:", dist_cc_mm
        
        if DEBUG:
            import pylab
            #pylab.imshow(lb)
            #pylab.show()
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

####################################################################################
# utility functions
#
def check_locating_beads(pname, delay=1000):
    """
    to draw dails pointing to the groups of beads
    for files under a given directory
    openCV display scheme is used to automatically show the dials
    for different files

    example:
    pname = "D://Research//INET//Du_Guosheng//data//060118//E//20180530//13175153//5" # 1.25mm
    check_locating_beads(pname, 300)
    """
    import cv2, dicom, os
    files = os.listdir(pname)
    print "number of files: %d"%len(files)

    for i in range(8, len(files)):#len(files)):#min(10, len(files))):
        print files[i]
        dcm = dicom.read_file(os.path.join(pname, files[i]))
##        print "%s = %s"%("thickness", dcm.SliceThickness),
##        print "%s = %s"%("pixelsize", dcm.PixelSpacing[0]),
##        print "%s = %s"%("kernel", dcm.ConvolutionKernel),
##        print "%s = %s"%("filter", dcm.FilterType)
        # generate an instance
        phan = CT_phantom(dcm)
        sb = SpiralBeads(phan)
        # get the locations of the beads
        branges, rois = sb.locate_beads()

        # display the image and show dials pointing to the beads
        img = scipy.float32(dcm.pixel_array.copy())
        img = scipy.uint8((img-img.min())/(img.max() - img.min()) * 255)
        for i, ind in enumerate(branges):
            ratio = (.5, .75)[i]
            theta = sb.profile['theta'][ind]
            # ensure length of the dials is appropriate
            x = scipy.arange(200)/200.*scipy.cos(theta)*phan.outer_radius*ratio+phan.center_x
            y = phan.center_y+scipy.tan(theta)*(x-phan.center_x)
            # draw the dial on the image
            cv2.line(img, (int(x[0]), int(y[0])), (int(x[-1]), int(y[-1])), 0xff0000)
        cv2.imshow('0', img)
        if cv2.waitKey(delay) & 0xff == 27:
            break
    cv2.destroyAllWindows()
def get_group_profiles(pname):
    """
    To collect multiple profiles of a series of files
    This can be used to demonstrate the overlap of the beads
    the FWHM may not be the best choice for the thickness
    example:
    pname = "D://Research//INET//Du_Guosheng//data//060118//E//20180530//13175153//6" # 2.5mm
    profiles = get_group_profiles(pname)
    for i in range(len(profiles)):
        _ = pylab.plot(profiles[i])
    pylab.show()
    """
    files = os.listdir(pname)
    profiles = []
    for i in range(0, 8):
        dcm = dicom.read_file(os.path.join(pname, files[i]))
        phan = CT_phantom(dcm)
        sb = SpiralBeads(phan)
        profiles.append(sb.profile['profile'])
    profiles = scipy.array(profiles)
    return profiles
def get_strip_values(array, rhos, center):
    """
    This is to demonstrate the partial volume effect on the edge of the phantom
    pixels that are near the square hole modules have pixel values lower than the normal ones
    example:
    pname = "D://Research//INET//Du_Guosheng//data//060118//E//20180530//13175153//7" # 10mm
    files = os.listdir(pname)
    dcm = dicom.read_file(os.path.join(pname, files[0]))
    phantom = CT_phantom(dcm)
    arr = get_strip_values(dcm.pixel_array,
                           scipy.linspace(-10, 10, 1000)+80.5,
                           (phantom.center_y, phantom.center_x))
    _=pylab.imshow(arr[:,:],cmap='gray', origin='lower');pylab.show()
    """
    theta = scipy.linspace(-scipy.pi, scipy.pi, 25000)
    yc, xc = center
    re = []
    for rho in rhos:
        ys = yc + scipy.cos(theta)*rho
        xs = xc + scipy.sin(theta)*rho
        pv = scipy.ndimage.map_coordinates(array, scipy.vstack([ys, xs]))
        re.append(pv)
    return scipy.array(re)
    
####################################################################################
if __name__ == "__main__":
    import os
    import pydicom as dicom
##    pname = "D://Research//INET//Du_Guosheng//data//060118//E//20180530//13175153//2" # 5mm H_SOFT kernel
##    pname = "D://Research//INET//Du_Guosheng//data//060118//E//20180530//13175153//3" # 5mm
    pname = "D://Research//INET//Du_Guosheng//data//060118//E//20180530//13175153//4" # 0.625mm
##    pname = "D://Research//INET//Du_Guosheng//data//060118//E//20180530//13175153//5" # 1.25mm
    pname = "D://Research//INET//Du_Guosheng//data//060118//E//20180530//13175153//6" # 2.5mm
##    pname = "D://Research//INET//Du_Guosheng//data//060118//E//20180530//13175153//7" # 10mm
    pname = "D://Research//INET//Du_Guosheng//data//060118//A//20180530//13202626//4" # 0.625mm
    pname = "D://Research//INET//Du_Guosheng//data//060118//A//20180530//13202626//5" # 1.25mm
    pname = "D://Research//INET//Du_Guosheng//data//060118//A//20180530//13202626//6" # 2.5mm

    pname = "D:\\Research\\INET\\Du_Guosheng\\data\\19220674-GE-RGRMS-E\\19220674-GE-RGRMS-E\\2-5.0mm"
    pname = "D:\\Research\\INET\\Du_Guosheng\\data\\19220674-GE-RGRMS-E\\19220674-GE-RGRMS-E\\4-1.25mm"#257
    pname = "D:\\Research\\INET\\Du_Guosheng\\data\\19220674-GE-RGRMS-E\\19220674-GE-RGRMS-E\\3-0.625mm"
    name = "D:\\pythonct\\py_codenew\\DI20190613\\15194997\\"
    fname = "C:\\Users\\hasee-pc\\Desktop\\dcmp\\U0000038"#87277803"
    fname = u"E://others//肿瘤医院//A\A\\U\\Z17"#"Z\\Z01"
    fname = u"E://others//肿瘤医院//A\A\\U\\Z22"
    fname = u"E://others//肿瘤医院//A\B\\D\\Z01"
    dcm= dicom.read_file(fname)                  
    phantom = CT_phantom(dcm)
    spiralbeads = SpiralBeads(phantom, diameter=75, pitch=90,number_beads=180)#diameter=75, pitch=90,number_beads=180
    profile = spiralbeads.get_profile(displayImage=True)
    thickness = spiralbeads.get_lthickness(profile,bc = dcm.SliceThickness)
    print thickness,spiralbeads.thickness2
    """
    index ="3-0.55mmH-qq"#3-0.55mmH-qq" 
    pname = name+index

    files = os.listdir(pname)
    thickness_list = []
    #files = ["U0000074", "U0000075", "U0000076"]

    import xlwt
    wb = xlwt.Workbook(encoding = 'utf-8')
    ws = wb.add_sheet('wy_ws1')
    ws.write(0,0,'分组')
    ws.write(0,1,'名称')
    ws.write(0,2,'标称')
    ws.write(0,3,'实测')
    ws.write(0,4,'误差')
    i=1

    
    for f in files:
        fname = os.path.join(pname, f)
        print fname
        dcm = dicom.read_file(fname)
        #pylab.imshow(dcm.pixel_array[160:-160, 160:-160])
        #pylab.show()
        print "%s = %s"%("thickness", dcm.SliceThickness),
        print "%s = %s"%("pixelsize", dcm.PixelSpacing[0]),
        print "%s = %s"%("kernel", dcm.ConvolutionKernel),
        print "%s = %s"%("filter", dcm.FilterType),
        print "size = ", dcm.Rows, dcm.Columns
        phantom = CT_phantom(dcm)

##        spiralbeads = SpiralBeads(phantom, diameter=162.4, pitch=90,number_beads=180)

        spiralbeads = SpiralBeads(phantom, diameter=75, pitch=90,number_beads=180)#原先写的50
        profile = spiralbeads.get_profile(displayImage=False)
        thickness = spiralbeads.get_lthickness(profile,bc = dcm.SliceThickness )
##        if thickness:

##            err=(thickness-float(dcm.SliceThickness))/float(dcm.SliceThickness)*100
##
##            ws.write(i,0,index)
##            ws.write(i,1,f)
##            ws.write(i,2,dcm.SliceThickness)
##            ws.write(i,3,thickness)
##            ws.write(i,4,str(err)+"%")
##            i=i+1
        continue
          
        if thickness:
            err=(thickness-float(dcm.SliceThickness))/float(dcm.SliceThickness)*100
####
            ws.write(i,0,index)
            ws.write(i,1,f)
            ws.write(i,2,dcm.SliceThickness)
            ws.write(i,3,thickness)
            ws.write(i,4,str(err)+"%")
            i=i+1
    #wb.save("data10625.xls")

                  
##        pro = scipy.ndimage.median_filter(profile['profile'], 257)
##        mean = pro.mean()
##        maxv = pro.max()
##        threshold = (mean + maxv)/2.
##        inds = scipy.where(pro > threshold)[0]
##        if inds[0] == 0:
##            pro = scipy.ndimage.shift(pro, int(len(pro)/2), mode='wrap')
##            inds = scipy.where(pro > threshold)[0]
##        tht = profile['theta']
##        span = (tht[inds[-1]] - tht[inds[0]])/scipy.pi*45
##        print "thickness = %s"%span
##        pylab.plot( pro)
##        pylab.plot(profile['profile'])
##        pylab.plot(scipy.ones(len(pro))*mean)
##        pylab.plot(scipy.ones(len(pro))*maxv)
##        pylab.show()

    '''        
        spiralbeads = SpiralBeads(phantom, diameter=166.3, pitch=90,number_beads=180)
        indices, profile_segments = spiralbeads.locate_beads()
        # use only the 90-pitched beads
        for i in range(1):#len(profile_segments)):
            pitch = [90, 50][i]
            segment = profile_segments[i]
            thickness = spiralbeads.get_thickness(pitch, segment)
        if thickness is None:
            print "cannot estimate the slice thickness!"
        else:
            print "The measured slice thickness is %f"%thickness
            thickness_list.append(thickness)
    pylab.boxplot(thickness_list)
    pylab.show()
    '''
    """
      
