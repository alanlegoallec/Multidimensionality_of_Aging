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
from pydicom.data import get_testdata_files
from skimage import exposure

data_path = '/n/groups/patel/uk_biobank/project_52887_41230/pancreas_20259/'
output_path = '/n/groups/patel/Alan/Aging/Medical_Images/images/Pancreas/20259/main/'

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

class Pancreas:
    # only one shot per participant
    def __init__(self, file, ext = '.jpg'):
        """
        :param file: string, ID _ data_ID _ instance _ test_number (+ potentially .zip), e.g. 2016212_20227_2_0 (+ potentially .zip)
        """
        self.file = os.path.splitext(file)[0]
        self.data_path = '/n/groups/patel/uk_biobank/project_52887_41230/pancreas_20259/'
        self.output_path = '/n/groups/patel/Alan/Aging/Medical_Images/images/Pancreas/20259/main/'
        self.unzip_folder = os.path.join(self.output_path, self.file)
        self.ext = ext
        self.img_computed = False # True if image already computed
        self.dicoms = None
        self.extract_info_from_file()
        
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
        self.ProtocolNames = self.ItemToValues('ProtocolName')
        self.PerformedProcedureStepDescription = self.ItemToValues('PerformedProcedureStepDescription')
        self.InstanceNumber = self.ItemToValues('InstanceNumber')
        self.SeriesDescription = self.ItemToValues('SeriesDescription')
        
    def extract_dcom14(self):
        # store all .dcm names in a list
        self.files = [ f for f in np.sort(os.listdir(self.unzip_folder)) if re.search('.dcm$', f)]
        self.dicoms = [ pydicom.filereader.dcmread(os.path.join(self.unzip_folder,self.files[14])) ]
        self.SeriesDescription = self.ItemToValues('SeriesDescription')
        
    def ItemToValues(self, item):
        if self.dicoms is None:
            raise Exception('self.dicoms has not been instantiated.')
        return [ self.dicoms[k].get_item(item).value.decode("utf-8") for k in range(len(self.dicoms)) ]
        
    def get_ProtocolNames(self):
        return self.ProtocolNames
    
    def get_pixels(self):
        """
        Return list images
        """
        return [ self.dicoms[k].pixel_array for k in range(len(self.dicoms))]
    
    def NameToPixels(self, name):
        """
        Return array corresponding to a name.
        TO COMPLETE FOR FULL BODY
        """
        indexes =  [i for i, x in enumerate(self.ProtocolNames) if x == name]
        return self.dicoms[indexes[0]].pixel_array
    
    def get_array(self, img_nb):
        """
        Return array corresponding to image n°img_nb.
        """
        return self.dicoms[img_nb].pixel_array
    
    def plot_imgs(self, img_nb = None, order = None):
        """
        Plot images. If img_nb is not None, plot the image number img_nb in self.dicoms.
        
        :param order: iterable, order in which images are displayed
        """
        
        if img_nb is not None:
            plt.figure()
            plt.imshow(self.dicoms[img_nb].pixel_array, cmap=plt.cm.bone)
            plt.show()
        elif order is not None:
            for k in order:
                dcm = self.dicoms[k]
                plt.figure()
                plt.imshow(dcm.pixel_array, cmap=plt.cm.bone)
                plt.show() 
        else:
            for dcm in self.dicoms:
                plt.figure()
                plt.imshow(dcm.pixel_array, cmap=plt.cm.bone)
                plt.show() 
    
    def img_index(self):
        """
        Return index of the selected img.
        """
        for index in range(len(self.SeriesDescription)):
            if 'T1MAP' in self.SeriesDescription[index]:
                return(index)
        raise NoFile

    def check_shape(self):
        """
        Check the shape of the selected image.
        If this shape is not right, it raises the error BadShape
        """
        if np.array(self.img_ar).shape != (288, 384):
            raise BadShape
            
    def compute_right_img(self, unzip = True):
        """
        Compute right image.
        """
        if unzip:
            self.unzip()
        try:
            self.extract_dcom14()
            self.img_ar = self.get_array(self.img_index())
        except:
            self.extract_info_from_unzip()
            self.img_ar = self.get_array(self.img_index())
        self.check_shape()
        # crop the image
        self.img_ar = self.img_ar[:,:350]
        self.img_ar_enhanced = exposure.equalize_adapthist(np.copy(self.img_ar), clip_limit=0.03) # contrast enhancement
        # preprocess array for jpg convertion
        def convert_for_jpg(img):
            img = img * (255/np.max(img))
            img = img.astype(int)
            return img
            
        self.img_ar = convert_for_jpg(self.img_ar)
        self.img_ar_enhanced = convert_for_jpg(self.img_ar_enhanced)
    
    def save_img(self):
        """
        Save images in the right folder.
        """
        try:
            # compute images
            self.compute_right_img()
            # save raw img
            cv2.imwrite(os.path.join(self.output_path, 'raw', self.patient_id + '_' + self.instance + '.jpg'), self.img_ar)
            # save img with enhanced contrast
            cv2.imwrite(os.path.join(self.output_path, 'contrast', self.patient_id + '_' + self.instance + '.jpg'), self.img_ar_enhanced)
        except BadShape:
            pass
        except NoFile:
            pass
        shutil.rmtree( self.unzip_folder )