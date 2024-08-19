import math
import cv2
import numpy as np

'''
this python file includes many function that we will use for image preprocessing
'''

def surface(points):
    '''
    returns the surface of the rectangle formed by 4 points
    '''
    if len(points) != 4:
        raise ValueError("Exactly four points are required")
    # Sort points in clockwise or counterclockwise order
    points = sorted(points, key=lambda p: (p[0], p[1]))
    centroid = (sum(p[0] for p in points) / 4, sum(p[1] for p in points) / 4)
    points = sorted(points, key=lambda p: (math.atan2(p[1] - centroid[1], p[0] - centroid[0])))

    # Calculate area using Shoelace formula
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    area = 0.5 * abs(
        x_coords[0]*y_coords[1] + x_coords[1]*y_coords[2] + x_coords[2]*y_coords[3] + x_coords[3]*y_coords[0]
        - y_coords[0]*x_coords[1] - y_coords[1]*x_coords[2] - y_coords[2]*x_coords[3] - y_coords[3]*x_coords[0]
    )
    return area


def largest_rect(result):
    '''
    returns the coordinates of the vertices of the largest rectangle detected

    result : the output of paddleocr
    '''
    max_surface=0
    max_points=None
    for line in result:
        surf=surface(line[0])
        if surf>max_surface: 
            max_surface=surf
            max_points=line[0]
    return max_points


def isVerticle(points):
    '''
    determines if the rectangle is verticle, meaning if the height is bigger than the width

    points: coordinates of the vertices of the rectangle
    '''
    p1=points[0]
    p3=points[2]
    x=p3[0]-p1[0]
    y=p3[1]-p1[1]

    if x<y: return True
    else: return False


def rotate_90(img):
    '''
    rotates the image by 90 degrees without cutting the image or extending it
    '''
    rotated_img = cv2.transpose(img)
    rotated_img = cv2.flip(rotated_img, flipCode=1) 
    return rotated_img


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated_image

