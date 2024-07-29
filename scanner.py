#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:34:50 2020

@author: abhishekbhat
"""
import numpy as np
import cv2
import imutils
import math
from deskew import determine_skew

def rotate(
        image: np.ndarray
) -> np.ndarray:
    '''
    rotates the image according to the angle determined by "determine_deskew"
    '''
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=(0,0,0))

def scanner(img): 

    image = cv2.imread(img)

    ratio = image.shape[0]/500

    resized = imutils.resize(image, height = 500)
    # cv2.imshow("resized", resized)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",gray)
    # cv2.waitKey(0)

    equal=cv2.equalizeHist(gray)
    # cv2.imshow("equal",equal)
    # cv2.waitKey(0)
    
    blurred = cv2.medianBlur(equal,15) # better when the card is further
    #blurred = cv2.GaussianBlur(equal, (5, 5), 0) # better when the card is closer, detects small text as well but also picks up background noise
    # cv2.imshow("blurred",blurred)
    # cv2.waitKey(0)

    edge = cv2.Canny(blurred, 50, 200,L2gradient = True)
    # cv2.imshow("Edge Image", edge)
    # cv2.waitKey(0)

    # Find Contours and save the largest ones
    cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    largest_area = 0
    bounding_box = None
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if area > largest_area:
            largest_area = area
            bounding_box = (x, y, w, h)

    # cv2.imshow("boundrect", resized)
    # cv2.waitKey(0)

    if bounding_box is not None:
        x, y, w, h = bounding_box
        cropped_image = image[int((y)*ratio):int((y + h )*ratio), int((x)*ratio):int((x + w )*ratio)]
        # cv2.imshow("cropped", cropped_image)
        # cv2.waitKey(0)
        rotated=rotate(cropped_image)
        # cv2.imshow("rotated",rotated)
        # cv2.waitKey(0)

    return rotated
