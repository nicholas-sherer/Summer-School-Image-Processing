# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:48:25 2016

@author: Nicholas Sherer
"""

import skimage.io as skio
import skimage.filters as skf
import skimage.feature as skfe
import skimage.morphology as skmo
import skimage.measure as skme
import skimage.segmentation as skseg
import numpy as np


class ImageCollectionFolder(object):
    """
    This is just a convenient way to reuse glob patterns for forming image
    collections.
    """

    def __init__(self, filepath, filename, filetype='.tif', channel_dict=None):
        self.image_names = filepath+filename
        self.filetype = filetype
        if channel_dict is None:
            self.channel_dict = {'Brightfield': '1', '514TIRF': '2'}
        else:
            self.channel_dict = channel_dict

    def channelCollection(self, channel, z):
        globend = 'xy*z' + str(z) + 'c' + self.channel_dict[channel]
        fullglob = self.image_names + globend + self.filetype
        return skio.ImageCollection(fullglob)


def findBackground(image_collection):
    conc_array = image_collection.concatenate()
    return np.mean(conc_array, axis=0)


def normalizeCollection(image_collection):
    background = findBackground(image_collection)
    normed_collection = []
    for image in image_collection:
        normed_collection.append(image / background)
    return normed_collection


def stdvProjection(image_collection):
    conc_array = image_collection.concatenate()
    return np.std(conc_array, axis=0)


def binaryMask(image):
    threshold = skf.threshold_li(image)
    return image > threshold


def testTriThreshold(image, low_threshold, high_threshold):
    low_mask = image < low_threshold
    high_mask = image > high_threshold
    mid_mask = np.logical_not(low_mask + high_mask)
    lm_n = np.sum(low_mask)
    hm_n = np.sum(high_mask)
    mm_n = np.sum(mid_mask)
