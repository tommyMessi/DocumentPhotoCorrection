import cv2
import numpy as np
import os

# class Perspector(object):
#     def __init__(self):
#         pass

def transform(points, image):
    h,w,c = image.shape
    pts = np.array(points)
    pts1 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    print(pts)
    print(pts1)
    M = cv2.getPerspectiveTransform(pts, pts1)
    dst = cv2.warpPerspective(image, M, (w,h))
    return dst
