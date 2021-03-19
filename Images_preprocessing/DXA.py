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
from tqdm import tqdm

import pydicom
from pydicom.data import get_testdata_files

data_path = '/n/groups/patel/uk_biobank/project_52887_41230/DXA_20158/'
output_path = '/n/groups/patel/Alan/Aging/Medical_Images/images/'

class BadShape(Exception):
    """
    Error raised when loaded data don't have the right shape.
    """
    pass

class BadImage(Exception):
    """
    Error raised when bad features are detected.
    """
    pass

class DXA:
    # only one shot per participant
    def __init__(self, file, ext = '.jpg'):
        """
        :param file: string, ID _ data_ID _ instance _ test_number (+ potentially .zip), e.g. 2016212_20227_2_0 (+ potentially .zip)
        """
        self.file = os.path.splitext(file)[0]
        self.data_path = '/n/groups/patel/uk_biobank/project_52887_41230/DXA_20158/'
        self.output_path = '/n/groups/patel/Alan/Aging/Medical_Images/images/'
        self.unzip_folder = os.path.join(self.output_path, self.file)
        self.ext = ext
        self.img_computed = False # True if image already computed
        #self.field_to_dir = {'21015': 'left', '21016': 'right'}
        self.extract_info_from_file()
        
        # resizing sizes
        self.skeleton_img_name = 'Total Body Skeleton'
        self.RGB_img_name = 'Total Body RGB'
        self.resizing_sizes = {'AP Spine': (724, 720), 'Left Femur': (626, 680), 'Left Ortho Knee': (851, 700),\
         'LVA': (1513, 684), 'Right Femur': (626, 680), 'Right Ortho Knee': (851, 700), 'Total Body': (811, 272),\
         'Total Body 2': (811, 272), self.skeleton_img_name: (811, 272)}
        
        # saving paths
        self.saving_paths = {
            'AP Spine':os.path.join(self.output_path,'Spine','201581','coronal','raw'), # coronal
            'LVA': os.path.join(self.output_path,'Spine','201581','sagittal','raw'),
            'Left Femur': os.path.join(self.output_path, 'Hip', '201582','main', 'raw','left'),
            'Right Femur': os.path.join(self.output_path, 'Hip', '201582','main', 'raw','right'),
            'Left Ortho Knee': os.path.join(self.output_path, 'Knee', '201583', 'main', 'raw', 'left'),
            'Right Ortho Knee': os.path.join(self.output_path, 'Knee', '201583', 'main', 'raw', 'right'),
            'Total Body': os.path.join(self.output_path, 'FullBody', '201580', 'main', 'figure'),
            'Total Body 2': os.path.join(self.output_path, 'FullBody', '201580', 'main', 'flesh'),
            self.skeleton_img_name: os.path.join(self.output_path, 'FullBody', '201580', 'main', 'skeleton'),
            self.RGB_img_name: os.path.join(self.output_path, 'FullBody', '201580', 'main', 'mixed')            
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
        # store all .dcm names in a list
        self.files = [ f for f in np.sort(os.listdir(self.unzip_folder)) if re.search('.dcm$', f)]
        # store all .dcm files in a list
        self.dicoms = [ pydicom.filereader.dcmread(os.path.join(self.unzip_folder,f)) for f in self.files ]
        # store all Protocol Names in a list ; same info as PerformedProcedureStepDescription
        self.ProtocolNames = [ self.dicoms[k].get_item('ProtocolName').value.decode("utf-8").strip() for k in range(len(self.dicoms)) ]
        try :
            i = self.ProtocolNames.index('Total Body', self.ProtocolNames.index('Total Body')) # 2nd occurence
            self.ProtocolNames[i] = 'Total Body 2'
        except:
            pass
        
        # store arrays in a dictionnary
        self.arrays = dict()
        for k,name in enumerate(self.ProtocolNames):
            self.arrays[name] = self.dicoms[k].pixel_array
        
    def ItemToValues(self, item):
        if self.dicoms is None:
            raise Exception('self.dicoms has not been instantiated.')
        try:
            return [ self.dicoms[k].get_item(item).value.decode("utf-8") for k in range(len(self.dicoms)) ]
        except AttributeError:
            return [ self.dicoms[k].get_item(item).value for k in range(len(self.dicoms)) ]
    
    def get_pixels(self):
        """
        Return list images
        """
        return [ self.dicoms[k].pixel_array for k in range(len(self.dicoms))]
    
    def NameToPixels(self, name):
        name = name.strip()
        try:
            return self.dicoms[self.ProtocolNames.index(name)].pixel_array
        except:
            print('error')
            return None
    
    def get_imgs_shapes(self):
        shapes = dict()
        flesh_total_body = True
        for name in self.ProtocolNames:
            shapes[name] = self.NameToPixels(name).shape
        return shapes
    
    @staticmethod
    def change_background(ar):
        h, w = ar.shape
        new_ar = np.copy(ar)
        binary_ar = np.where( np.logical_or(ar > 250, ar < 70), 0, 1)
        left_borders = np.sum( np.where(np.cumsum(binary_ar, axis = 1) == 0, 1, 0), axis=1 )
        right_borders = np.sum( np.where(np.cumsum(np.flip(binary_ar, axis = 1), axis = 1) == 0, 1, 0), axis=1 )
        top_borders = np.sum( np.where(np.cumsum(binary_ar, axis = 0) == 0, 1, 0), axis=0 )
        bottom_borders = np.sum( np.where(np.cumsum(np.flip(binary_ar, axis = 0), axis = 0) == 0, 1, 0), axis=0 )
        for k in range(h):
            new_ar[k,:left_borders[k]] = 0
            new_ar[k, w-right_borders[k]:] = 0
        for l in range(w):
            new_ar[:top_borders[l], l] = 0
            new_ar[h-bottom_borders[l]:, l] = 0
        return(new_ar)
    
    def skeleton_computation(self):
        """
        Process Total Body image, keeping only the skeleton on a black background.
        """
        if 'Total Body' in self.ProtocolNames:
            self.ProtocolNames.append(self.skeleton_img_name)
            self.arrays[self.skeleton_img_name] = self.change_background(self.arrays['Total Body'])
        
    def resize_arrays(self):
        """
        Resize arrays to the computed resizing_sizes.
        """
        for name in self.ProtocolNames:
            HEIGHT, WIDTH = self.resizing_sizes[name]
            self.arrays[name] = cv2.resize(self.arrays[name], (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)

    def flip_arrays(self):
        """
        Flip arrays when necessary (for left images).
        """
        for name in self.arrays:
            if 'left' in name.lower():
                self.arrays[name] = np.flip(self.arrays[name], 1)
                                         
    def RGB_computation(self):
        """
        Create RGB image from the 3 different views of the Total Body.
        """
        if 'Total Body' in self.ProtocolNames and 'Total Body 2' in self.ProtocolNames:
            self.ProtocolNames.append(self.RGB_img_name)
            self.arrays[self.RGB_img_name] = np.concatenate([np.expand_dims(self.arrays['Total Body'],axis=-1),
                                             np.expand_dims(self.arrays['Total Body 2'],axis=-1),
                                             np.expand_dims(self.arrays[self.skeleton_img_name],axis=-1)],axis=-1)
    
    def plot_imgs(self, img_name = None, raw = False):
        """
        Plot images. If img_name is not None, plot the image whose name is img_name.
        If raw is True, plot raw images from dicom files.
        """
        if raw:
            if img_name is not None:
                plt.figure()
                plt.imshow(self.dicoms[self.ProtocolNames.index(img_name)].pixel_array, cmap=plt.cm.bone)
                plt.show()
            else:
                for dcm in self.dicoms:
                    plt.figure()
                    plt.imshow(dcm.pixel_array, cmap=plt.cm.bone)
                    plt.show() 
                    
        else:
            if img_name is not None:
                plt.figure()
                plt.imshow(self.arrays[img_name], cmap=plt.cm.bone)
                plt.show()
            else:
                for name in np.sort(self.ProtocolNames):
                    print(name)
                    plt.figure()
                    plt.imshow(self.arrays[name], cmap=plt.cm.bone)
                    plt.show()

    def compute_imgs(self, unzip=True):
        """
        Load and process images.
        """
        if unzip:
            self.unzip()
        self.extract_info_from_unzip()
        self.skeleton_computation()
        self.resize_arrays()
        self.flip_arrays()
        self.RGB_computation()
    
    def save_imgs(self):
        """
        Save image in the right folder.
        """
        for name in self.arrays:
            cv2.imwrite( os.path.join(self.saving_paths[name], self.patient_id + '_' + self.instance + self.ext), self.arrays[name])
        shutil.rmtree(self.unzip_folder)
            
            