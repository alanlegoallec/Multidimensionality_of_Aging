# importing required modules
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from zipfile import ZipFile
import shutil
import scipy.misc
import math
import re
import cv2 as cv
import pydicom
from pydicom.data import get_testdata_files
from tqdm import tqdm
from scipy.signal import savgol_filter

data_path = '/n/groups/patel/uk_biobank/project_52887_41230/OCT_21017/'
output_path = '/n/groups/patel/Alan/Aging/Medical_Images/images/Eyes/OCT/'

class BadShape(Exception):
    """
    Error raised when loaded data don't have the right shape.
    """
    pass

class BadImage(Exception):
    """
    Error raised when there an image must be ignored.
    """
    pass

def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

class ImgAnalysis:
    def __init__(self,ar):
        self.ar = ar
    
    def compute_upper_border_argmax(self, ar):
        """
        Compute upper border of an OCT image. 
        :return : 1D-array (curve).
        """
#         self.denoised_ar = cv.fastNlMeansDenoising(ar, 10, 7, 21)
#         edges = cv.Canny(self.denoised_ar,100,200)
#         self.denoised_ar = cv.fastNlMeansDenoising(ar, 30, 7, 3)
#         self.edges = cv.Canny(self.denoised_ar,290,310)
        self.denoised_ar = cv.fastNlMeansDenoising(ar, h=30, templateWindowSize=7, searchWindowSize=7)
        self.edges = cv.Canny(self.denoised_ar,30,150)
        return smoothTriangle(savgol_filter(np.argmax(self.edges, axis=0),65,1),15)
    
    def compute_upper_border(self, ar):
        """
        Compute upper border of an OCT image. 
        :return : 1D-array (curve).
        """
        self.denoised_ar = cv.fastNlMeansDenoising(ar, h=30, templateWindowSize=7, searchWindowSize=7)
        self.edges = cv.Canny(self.denoised_ar,15,100)#30,150
        h,w = self.edges.shape
        self.border_avg = []
        self.border_median = []
        for k in range(w):
            white_pixels = np.nonzero(self.edges[:,k])[0]
            if len(white_pixels) == 0:
                pass
            else:
                self.border_avg.append(int(np.average(white_pixels)))
                self.border_median.append(int(np.median(white_pixels)))
        if len(self.border_avg) < 400:
            raise BadImage
        return [smoothTriangle(savgol_filter(np.array(self.border_avg),65,1),35),
                smoothTriangle(savgol_filter(np.array(self.border_median),65,1),35),
                smoothTriangle(savgol_filter(np.argmax(self.edges, axis=0),65,1),35)] #15
    
    @staticmethod
    def curvature(curve):
        """
        Compute a curve curvature.
        """
        curve_2D = np.array([np.arange(len(curve)), curve])
        curve_2D = curve_2D.transpose()
        dx_dt = np.gradient(curve_2D[:, 0])
        dy_dt = np.gradient(curve_2D[:, 1])
        velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        tangent = np.array([1/ds_dt] * 2).transpose() * velocity
        tangent_x = tangent[:, 0]
        tangent_y = tangent[:, 1]
        deriv_tangent_x = np.gradient(tangent_x)
        deriv_tangent_y = np.gradient(tangent_y)
        dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
        length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
        normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt
        d2s_dt2 = np.gradient(ds_dt)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        return curvature

    def compute_curvature_max(self):
        """
        Computation of maximum curvature of an OCT image (array) upper border.
        This maximum curvature must correspond to the fovea.
        """
        try:
            self.curves = self.compute_upper_border(self.ar[:,:,0])
            self.curvatures_ = [self.curvature(c) for c in self.curves]
            self.smoothed_curvatures_ = [smoothTriangle(c, 5) for c in self.curvatures_]
            self.borne_inf = 100
            self.borne_sup = 400
            # computing maximums for the 3 curves and keeping the min the maximums
            self.curv_index = np.argmin([np.max(c[self.borne_inf:self.borne_sup]) for c in self.smoothed_curvatures_])
            self.max_curv_index = self.borne_inf + np.argmax(self.smoothed_curvatures_[self.curv_index][self.borne_inf:self.borne_sup])
            return self.smoothed_curvatures_[self.curv_index][self.max_curv_index]
        except BadImage:
            return 0
    
    def crop_image(self):
        """
        Crop image and return it.
        """
        h,w = self.ar.shape[:2]
        new_h = 500
        # median_value: value (axis 0) around which cropping is centered
        median_value = int(np.median(self.compute_upper_border(self.ar[:,:,0])[1]))
        if median_value < new_h//2:
            return self.ar[:new_h,:,:]
        else:
            if median_value < h - new_h//2:
                return self.ar[median_value - new_h//2 : median_value + new_h//2,:,:]
            else:
                return self.ar[h - new_h:,:,:]

    @staticmethod
    def plot2(data1, data2, label1="", label2=""):
        t = np.arange(len(data1))

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        #ax1.set_xlabel('time')
        ax1.set_ylabel(label1, color=color)
        ax1.plot(t, data1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel(label2, color=color)  # we already handled the x-label with ax1
        ax2.plot(t, data2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
    
    def plots(self):
        plt.figure()
        plt.imshow(self.ar)
        plt.plot(np.arange(len(self.curves[self.curv_index])),self.curves[self.curv_index])
        plt.plot([self.max_curv_index, self.max_curv_index], [100, 400])
        
        c = np.array(self.smoothed_curvatures_[self.curv_index])
        c[:50] = 0
        c[450:] = 0
        self.plot2(self.curves[self.curv_index], c, 'curve', 'curvature')
        
    def plot_curves(self):
        """
        Plot average, median and argmax curves.
        """
        plt.figure()
        plt.imshow(self.ar)
        for curve in self.curves:
            plt.plot(curve)
        plt.legend(['Average curve', 'Median curve', 'Argmax curve'])
        plt.show()
        

class OCT:
    # only one shot per participant
    def __init__(self, file, ext = '.jpg'):
        """
        :param file: string, ID _ data_ID _ instance _ test_number (+ potentially .zip), e.g. 2016212_20227_2_0 (+ potentially .zip)
        """
        self.file = os.path.splitext(file)[0]
        self.data_path = '/n/groups/patel/uk_biobank/project_52887_41230/OCT_21017/'
        self.output_path = '/n/groups/patel/Alan/Aging/Medical_Images/images/Eyes/OCT/'
        self.unzip_folder = os.path.join(self.output_path, self.file)
        self.ext = ext
        self.img_computed = False # True if image already computed
        self.dicoms = None
        self.extract_info_from_file()
        self.sides = {
            '21017': 'left',
            '21018': 'right'
        }
        
    def extract_info_from_file(self):
        """
        Extract each information from file name: patient_id, field_id, instance and shot.
        """
        self.patient_id = re.findall('(\d+)_\d+_\d+_\d+', self.file)[0]
        self.field_id = re.findall('\d+_(\d+)_\d+_\d+', self.file)[0]
        self.instance = re.findall('\d+_\d+_(\d+)_\d+', self.file)[0]
        self.shot = re.findall('\d+_\d+_\d+_(\d+)', self.file)[0]
    
    def unzip(self):
        """
        Extract information from zip file.
        """
        try:
            os.mkdir(self.unzip_folder)
        except FileExistsError:
            pass
        with ZipFile(os.path.join(self.data_path, self.file + '.zip'), 'r') as zip:
            # printing all the contents of the zip file
            zip.extractall(self.unzip_folder)

    def extract_info_from_unzip(self):
        """
        Extract relevant information from unzipped files.
        """
        # store all .png files in a list
        self.files = [ f for f in np.sort(os.listdir(self.unzip_folder)) if re.search('.png$', f)]
        # load images
        self.arrays = []
        self.img_nb = [ int(re.findall('image\d+_(\d+).png', f)[0]) for f in self.files]
        for k in np.argsort(self.img_nb):
            self.arrays.append(cv.imread(os.path.join(self.unzip_folder, self.files[k])))
        self.shape_check()
        
    def shape_check(self):
        """
        Check shapes of arrays.
        """
        for ar in self.arrays:
            if ar.shape != (650, 512, 3):
                raise BadShape
                
    def compute_img(self, unzip=True, progress_bar=True):
        """
        Select right image among all the images present in each sample.
        mcc = 'maximum curvature curve'
        """
        if unzip:
            self.unzip()
        self.extract_info_from_unzip()
        self.mcc = np.zeros((len(self.arrays),))
        # first and last images looked at
        self.first_img = 30
        self.last_img = 105
        if progress_bar:
            for k,ar in enumerate(tqdm(self.arrays[self.first_img:self.last_img])):
                self.mcc[self.first_img + k] = ImgAnalysis(ar).compute_curvature_max()
        else:
            for k,ar in enumerate(self.arrays[self.first_img:self.last_img]):
                self.mcc[self.first_img + k] = ImgAnalysis(ar).compute_curvature_max()
        self.smoothed_mcc = savgol_filter(self.mcc,5,1)
        # argmax of smoothed mcc
        self.argmax_smoothed_mcc = np.argmax(self.smoothed_mcc)
        # self.selected_img: real argmax curvature => chosen image
        #self.max = self.first_img+self.argmax_smoothed_mcc - 10 + np.argmax(self.mcc[self.argmax_smoothed_mcc-10:self.argmax_smoothed_mcc + 10])
        self.selected_img =  self.argmax_smoothed_mcc
        self.img = ImgAnalysis(self.arrays[self.selected_img]).crop_image()
        
        # flip image if necessary
        if self.sides[self.field_id] == 'right': 
            self.img = np.flip(self.img, axis=1)
        
    def plot(self, selected_img=True, img_nb=None):
        if img_nb is not None:
            plt.figure()
            plt.imshow(self.arrays[img_nb])
            plt.show()
        elif selected_img:
            plt.figure()
            plt.imshow(self.img)
            plt.show()
        else:
            for k, ar in enumerate(self.arrays):
                plt.figure()
                print(k)
                plt.imshow(ar)
                plt.show()
    
    def plotMaxSearch(self):
        """
        Plot graph explaining how the 'real' maximum curvature is searched.
        """
        plt.figure()
        # curves
        plt.plot(self.mcc)
        plt.plot(self.smoothed_mcc)
        # confidence interval
        val = self.smoothed_mcc[self.argmax_smoothed_mcc]
        plt.plot([self.argmax_smoothed_mcc-10, self.argmax_smoothed_mcc+10], [val, val], c='r')
        # real mc max
        plt.plot([self.selected_img],[self.smoothed_mcc[self.selected_img]], marker ='o', c='g')
        #legend
        plt.legend(['raw mc curve', 'smoothed mc curve', 'confidence interval', 'selected maximum'])
    
    def save_img(self):
        """
        Save image in the right folder.
        """
        if self.shot == '1':
            print(self.file)
            try:
                # compute images
                self.compute_img()
                # save img
                cv.imwrite(os.path.join(self.output_path, 'raw', self.sides[self.field_id], self.patient_id + '_' + self.instance + '.jpg'), self.img)

            except BadShape:
                pass
            shutil.rmtree(self.unzip_folder)