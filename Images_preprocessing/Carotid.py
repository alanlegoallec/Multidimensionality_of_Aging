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
import cv2
import pydicom
from skimage import exposure
from tqdm import tqdm

data_path = '/n/groups/patel/uk_biobank/project_52887_41230/carotid_20223/'
output_path = '/n/groups/patel/Alan/Aging/Medical_Images/images/Carotid/'

class BadShape(Exception):
    """
    Error raised when loaded data don't have the right shape.
    """
    pass

class NoFile(Exception):
    """
    Error raised when there is not the image we look for.
    """
    pass

class Carotid:
    # only one shot per participant
    def __init__(self, file, ext = '.jpg'):
        """
        :param file: string, ID _ data_ID _ instance _ test_number (+ potentially .zip), e.g. 2016212_20227_2_0 (+ potentially .zip)
        """
        self.file = os.path.splitext(file)[0]
        self.data_path = '/n/groups/patel/uk_biobank/project_52887_41230/carotid_20223/'
        self.output_path = '/n/groups/patel/Alan/Aging/Medical_Images/images/Carotid/'
        self.unzip_folder = os.path.join(self.output_path, self.file)
        self.side = {'20222':'left', '20223':'right'}
        self.ext = ext
        self.img_computed = False # True if image already computed
        self.dicoms = None
        self.extract_info_from_file()
        
        # name of images
        self.shortaxis = 'shortaxis'
        self.longaxis = 'longaxis'
        self.CIMT150 = 'CIMT150'
        self.CIMT120 = 'CIMT120'
        self.RGB = 'mixedRGB'
        
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
        # store all .dcm names in a list
        self.files = [ f for f in np.sort(os.listdir(self.unzip_folder)) if re.search('.dcm$', f)]
        # store all .dcm files in a list
        self.dicoms = [ pydicom.filereader.dcmread(os.path.join(self.unzip_folder,f)) for f in self.files ]
        # store all Protocol Names in a list
        self.AcquisitionTimes = [ int(self.ItemToValue(dcm.get_item('AcquisitionTime'))) for dcm in self.dicoms]
        
        # dict of arrays
        self.arrays = dict()
        for k in range(len(self.files)):
            self.arrays[self.compute_img_name(self.get_array(k))] = self.get_array(k)
            if len(self.arrays) == 4:
                break
        
    def ItemToValue(self, item):
        try:
            return item.value.decode("utf-8")
        except:
            return item.value

    def get_array(self, img_nb):
        """
        Return array corresponding to image n°img_nb.
        """
        return self.dicoms[img_nb].pixel_array
    
    def plot_imgs(self, img_nb = None, raw = False):
        """
        Plot images. If img_nb is not None, plot the image number img_nb in self.dicoms.
        
        :param order: iterable, order in which images are displayed
        """
        
        if img_nb is not None:
            plt.figure()
            plt.imshow(self.dicoms[img_nb].pixel_array, cmap=plt.cm.bone)
            plt.show()
        elif raw:
            for dcm in self.dicoms:
                plt.figure()
                plt.imshow(dcm.pixel_array, cmap=plt.cm.bone)
                plt.show() 
        else:
            for name in self.arrays:
                plt.figure()
                print(name)
                plt.imshow(self.arrays[name], cmap=plt.cm.bone)
                plt.show() 
    
    @staticmethod
    def closest_state(img,position,thresholds):
        """
        img: 3D array
        position: list of ints: top,bottom,left,right
        thresholds: list of ints
        """
        top, bottom, left, right = position
        avg = np.average(img[top:bottom,left:right,:])
        if np.abs(avg - thresholds[0]) < np.abs(avg - thresholds[1]):
            return False
        else:
            return True
    
    def compute_img_name(self,img):
        """
        Return 'name' (right classification) of an image.
        """
        if self.side[self.field_id] == 'left':
            colors = {'upper':[350,360,75,85], 'lower':[410,420,75,85], 'thresholds':[80,150]} #top,bottom,left,right
            check_symbol = {'upper':[357,379,50,70], 'lower':[426,448,50,70], 'thresholds':[100,200]}
            angles = {'small':[440,450,840,850], 'large':[500,510,780,790], 'thresholds':[94,146]}
        else:
            colors = {'upper':[160,170,70,80], 'lower':[230,240,70,80], 'thresholds':[80,150]} #top,bottom,left,right
            check_symbol = {'upper':[168,188,106,126], 'lower':[237,257,106,126], 'thresholds':[100,200]}
            angles = {'small':[500,510,990,1000], 'large':[440,450,930,940], 'thresholds':[94,146]}
        if self.closest_state(img, colors['lower'], colors['thresholds']):
            if self.closest_state(img, angles['small'], angles['thresholds']):
                if self.side[self.field_id] == 'left':
                    return self.CIMT150 #'CIMT210'
                else:
                    return self.CIMT120
            elif self.closest_state(img, angles['large'], angles['thresholds']):
                if self.side[self.field_id] == 'left':
                    return self.CIMT120 #'CIMT240'
                else:
                    return self.CIMT150
            else:
                return None
        elif self.closest_state(img, colors['upper'], colors['thresholds']):
            if self.closest_state(img, check_symbol['upper'], check_symbol['thresholds']):
                return self.longaxis
            else:
                return self.shortaxis
        else:
            return None
    
    def crop_imgs(self):
        """
        Crop images to keep only useful information.
        """
        left = 300
        right = 736
        top = 113
        bottom = 618
        for name in self.arrays:
            self.arrays[name] = self.arrays[name][top:bottom,left:right,:]
    
    def RGB_computation(self):
        """
        Create RGB image from the CIMTs and longaxis images.
        """
        
        def reformat_ar(ar):
            return np.expand_dims(np.uint8(np.average(ar, axis=-1)),axis=-1)
        
        if self.CIMT120 in self.arrays and self.CIMT150 in self.arrays and self.longaxis in self.arrays:
            self.arrays[self.RGB] = np.concatenate([reformat_ar(self.arrays[self.CIMT120]),
                                             reformat_ar(self.arrays[self.CIMT150]),
                                             reformat_ar(self.arrays[self.longaxis])], axis=-1)
    
    def compute_imgs(self, unzip=True):
        """
        Load and process images.
        """
        if unzip:
            self.unzip()
        self.extract_info_from_unzip()
        self.crop_imgs()
        self.RGB_computation()
        
    def save_imgs(self):
        """
        Save image in the right folder.
        """
        for name in self.arrays:
            path = os.path.join(self.output_path, '202223','main', name, self.side[self.field_id],\
                                self.patient_id + '_' + self.instance + self.ext)
            cv2.imwrite(path, self.arrays[name])
        shutil.rmtree(self.unzip_folder)