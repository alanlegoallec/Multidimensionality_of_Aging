import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import re
MI_path = '/n/groups/patel/Alan/Aging/Medical_Images/'
data_path = '/n/groups/patel/uk_biobank/project_52887_41230/EyeFundus/'
output_path = os.path.join( MI_path, 'images/EyeFundus/')


class BadShape(Exception):
    """
    Error raised when loaded data don't have the right shape.
    """
    pass


class EyeFundus:
    def __init__(self, data_path, file, output_path, ext='.jpg'):
        self.data_path = data_path
        self.file = file
        self.output_path = output_path
        self.ext = ext
        self.img_computed = False  # True if image already computed

    def extract_info_from_file(self):
        """
        Extract each information from file name: patient_id, field_id, instance and shot.
        """
        self.patient_id = re.findall('(\d+)_\d+_\d+_\d+', self.file)[0]
        self.field_id = re.findall('\d+_(\d+)_\d+_\d+', self.file)[0]
        self.instance = re.findall('\d+_\d+_(\d+)_\d+', self.file)[0]
        self.shot = re.findall('\d+_\d+_\d+_(\d+)', self.file)[0]

    def load_img(self):
        """
        Load image.
        """
        self.img = Image.open(os.path.join(self.data_path, self.file))

    def check_shape(self):
        """
        Check the shape of the loaded image.
        If this shape is not right, it raises the error BadShape
        """
        if np.array(self.img).shape != (1536, 2048, 3):
            raise BadShape

    def convert_img(self):
        """
        Convert image to the right extension.
        """
        self.img = self.img.convert('RGB')

    def compute_img(self):
        """
        Load, check shape, and convert image to the right extension.
        """
        self.load_img()
        self.check_shape()
        self.convert_img()
        self.img_computed = True

    def save_img(self):
        """
        Save image in the right folder. If self.img not already computed, call self.compute_img
        """
        self.extract_info_from_file()
        path_0 = os.path.join(self.output_path, self.field_id, self.patient_id + self.ext)
        path_1 = os.path.join(self.output_path, self.field_id + '_' + self.instance, self.patient_id + self.ext)
        if self.shot == '0':  # first shot
            if os.path.exists(path_0) or os.path.exists(path_1):
                print(self.patient_id, 'already done')
                pass
            else:
                if not self.img_computed:
                    self.compute_img()
                if self.instance == '0':
                    self.img.save(path_0)
                else:
                    self.img.save(path_1)
        else:  # newer shot
            if not self.img_computed:
                self.compute_img()
            if self.instance == '0':
                self.img.save(path_0)
            else:
                self.img.save(path_1)

