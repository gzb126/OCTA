import cv2
import os
import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import imageio
import numpy as np
import vtk
from PIL import Image


def ImageDecode(root):
    in_path = root + 'E233/6SJ3B6375L62AZ4BM9QEI4BV9NLAFM0D127IOLS4X2ZU.EX.DCM'
    out_path = './imgfiles/'
    dcm_data = pydicom.dcmread(in_path)
    if 'PixelData' in dcm_data:
        image = dcm_data.pixel_array
        image = (image / image.max()) * 255
        for i in range(image.shape[2]):
            img = image[:,:,i]
            img = Image.fromarray(img.astype('uint8'))
            img.save(out_path + str(i) + '.jpg')


if __name__ == '__main__':
    # root = r'\\10.10.93.215\公共空间\读片中心\项目图像\糖络宁\糖络宁项目 ZEISS OCT OCTA图像\成都中医大银海眼科医院\OCTA访视/'
    root = r'F:\2_PycharmWorks\OCTAReadProject\imgfiles\DataFiles/'
    ImageDecode(root)