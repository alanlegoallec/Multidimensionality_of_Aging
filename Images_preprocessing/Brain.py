# import required modules
import os
import numpy as np
import nibabel as nib
import scipy.misc
import math
import re

data_path = '../../images/Brain/20227'


class BadShape(Exception):
    """
    Error raised when loaded data don't have the right shape.
    """
    pass


class brain_mri:
    """
    This class processes brain MRI data (fMRI): 4d_matrix (video of 3D brain) and/or reference scan (3d-array: 3D brain).
    It enables to extract pictures from 3D brain, computed from different slices of it.
    """

    def __init__(self, file):
        """
        :param file: string, ID _ data_ID _ instance _ test_number, e.g. 2016212_20227_2_0
        """
        self.file = file
        # general path where all extracted data are stored
        self.data_path = '../../images/Brain/20227'
        # path to reference scans
        self.reference_scan_path = os.path.join(self.data_path, 'reference_scan', file, ) + '.nii.gz'
        # path to 4d_matrix ()
        self.full_matrix_path = os.path.join(self.data_path, '4d_matrix', file) + '.nii.gz'
        # subpath to save computed images
        self.output_path = None  # will be chosen when data source is

        self.reference_scan = None
        self.full_matrix = None

        # 3d-array chosen (either reference_scan or full_matrix at a specific time)
        self.chosen_scan = None

        # cutting parameters
        self.slice_indexes = (None, None, None)  # index of middle slice for each of the 3 resulting images
        self.delta = None  # gap between each slice for the same image
        self.time = None  # chosen timframe when working with 4d_matrix

        # computed images
        self.image1 = None
        self.image2 = None
        self.image3 = None

        # image extension
        self.ext = '.jpg'

    def _load_reference_scan(self):
        """
        Load reference scan stored in self.reference_scan_path, and store it in a 3d-array
        """
        self.reference_scan = nib.load(self.reference_scan_path).get_fdata()
        if self.reference_scan.shape != (88, 88, 64):
            raise BadShape('reference scan does not have the right shape')

    def _load_full_matrix(self):
        """
        Load 4d_matrix stored in self.full_matrix_path, and store it in a 4d-array
        """
        self.full_matrix = nib.load(self.full_matrix_path).get_fdata()
        if self.full_matrix.shape != (88, 88, 64, 490):
            raise BadShape('4d_matrix does not have the right shape')

    def _pad_to_square(self, ar):
        """
        Pad a 2d-array to make it square.
        :param ar: 2d-array
        """
        d1, d2 = ar.shape
        pad1 = abs(d2 - d1) // 2
        pad2 = math.ceil(abs(d2 - d1) / 2)  # round up, to get: pad1 + pad2 = d1 - d2
        if d1 < d2:
            return np.pad(ar, ((pad1, pad2), (0, 0)), mode='constant', constant_values=0)
        if d2 < d1:
            return np.pad(ar, ((0, 0), (pad1, pad2)), mode='constant', constant_values=0)
        return ar

    def _get_slices(self, axis=0):
        """
        Compute slices for a given axis and a given index with a gap delta between each slice.
        :param ar: 3d-array
        :param axis: int, axis along which the slice is done
        Return 3 padded slices
        """
        index = self.slice_indexes[axis]
        if axis == 0:
            return (self._pad_to_square(self.chosen_scan[index - self.delta, :, :]),
                    self._pad_to_square(self.chosen_scan[index, :, :]),
                    self._pad_to_square(self.chosen_scan[index + self.delta, :, :]))
        elif axis == 1:
            return (self._pad_to_square(self.chosen_scan[:, index - self.delta, :]),
                    self._pad_to_square(self.chosen_scan[:, index, :]),
                    self._pad_to_square(self.chosen_scan[:, index + self.delta, :]))
        else:  # axis == 2
            return (self._pad_to_square(self.chosen_scan[:, :, index - self.delta]),
                    self._pad_to_square(self.chosen_scan[:, :, index]),
                    self._pad_to_square(self.chosen_scan[:, :, index + self.delta]))

    def _slices_to_image(self, s1, s2, s3):
        """
        Concatenate in the same array slices.
        :param s1: 2d-array
        :param s2: 2d-array
        :param s3: 2d-array
        Return PIL image
        """
        return (scipy.misc.toimage(np.concatenate([np.expand_dims(s1, axis=0),
                                                   np.expand_dims(s2, axis=0),
                                                   np.expand_dims(s3, axis=0)], axis=0)))

    def _compute_images(self):
        """
        Compute images from 3D-arrays.
        """

        # 1st step: get 3 slices for each image
        S1 = self._get_slices(0)
        S2 = self._get_slices(1)
        S3 = self._get_slices(2)

        # 2nd step: compute the 3 images
        self.image1 = self._slices_to_image(S1[0], S1[1], S1[2]).rotate(90)
        self.image2 = self._slices_to_image(S2[0], S2[1], S2[2]).rotate(90)
        self.image3 = self._slices_to_image(S3[0], S3[1], S3[2]).rotate(-90)

    def compute_images_from_ref_scan(self, i1=None, i2=None, i3=None, delta=5):
        """
        Compute images from reference scan.

        :param i1: int, index of middle slice for image 1
        :param i2: int, index of middle slice for image 2
        :param i3: int, index of middle slice for image 3
        :param delta: int, gap between each slice for the same image
        """
        # load reference scan and define it as the chosen one
        self._load_reference_scan()
        self.chosen_scan = self.reference_scan

        # update output_path
        self.output_path = os.path.join('from_ref_scan', re.findall('(\d+)_\d+_\d+_\d+', self.file)[0] + self.ext)

        # update class parameters
        it1, it2, it3 = np.array(self.chosen_scan.shape) // 2
        self.slice_indexes = (i1 or it1, i2 or it2, i3 or it3)
        self.delta = delta

        # compute images
        self._compute_images()

    def compute_images_from_4d_matrix(self, time=None, i1=None, i2=None, i3=None, delta=5):
        """
        Compute images from reference scan.

        :param i1: int, index of middle slice for image 1
        :param i2: int, index of middle slice for image 2
        :param i3: int, index of middle slice for image 3
        :param delta: int, gap between each slice for the same image
        """
        # load 4d_matrix and define it as the chosen one
        self._load_full_matrix()
        self.time = time or self.full_matrix.shape[-1] // 2
        self.chosen_scan = self.full_matrix[:, :, :, self.time]

        # update output_path
        self.output_path = os.path.join('from_4d_matrix', re.findall('(\d+)_\d+_\d+_\d+', self.file)[0] + self.ext)

        # update class parameters
        it1, it2, it3 = np.array(self.chosen_scan.shape) // 2
        self.slice_indexes = (i1 or it1, i2 or it2, i3 or it3)
        self.delta = delta

        # compute images
        self._compute_images()

    def get_images(self):
        """
        Return computed images from brain MRI.
        :return : tuple of 3 PIL square images
        """
        return (self.image1, self.image2, self.image3)

    def save_images(self):
        """
        Save images to proper folders.
        """
        self.image1.save(os.path.join(self.data_path, 'slice1', self.output_path))
        self.image2.save(os.path.join(self.data_path, 'slice2', self.output_path))
        self.image3.save(os.path.join(self.data_path, 'slice3', self.output_path))
