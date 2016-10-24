import cv2
import numpy as np
from numpy import argwhere
import sys
from scipy import spatial
from skimage.feature import hog
from skimage import data, color, exposure
import matplotlib.pyplot as plt

def boundingBox(A):
    if np.mean(A) == 0:
        return A.copy()

    B = argwhere(A)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1

    Atrim = A[ystart:ystop, xstart:xstop]

    return Atrim.copy()

def calculateComponents(mat1, mat2):
    # Find max blob
    mat1Component = boundingBox(mat1)
    mat2Component = boundingBox(mat2)

    # Rescale if different sizes
    if mat1Component.shape[0] * mat1Component.shape[1] > mat2Component.shape[0] * mat2Component.shape[1]:
        mat2Component = cv2.resize(mat2Component, (mat1Component.shape[1], mat1Component.shape[0]))
    elif mat1Component.shape[0] * mat1Component.shape[1] < mat2Component.shape[0] * mat2Component.shape[1]:
        mat1Component = cv2.resize(mat1Component, (mat2Component.shape[1], mat2Component.shape[0]))

    '''
    cv2.imshow("Flappy", np.vstack((mat1Component, mat2Component)))
    cv2.waitKey(0)
    '''
    return mat1Component, mat2Component

def calculateDistance(s, g):
    state_image, goal_image = calculateComponents(s, g)
    cellSize = .1 * state_image.shape[0]

    
    state_fd = hog(state_image, orientations=4, pixels_per_cell=(cellSize, cellSize),
                cells_per_block=(1, 1), visualise=False)

    goal_fd = hog(goal_image, orientations=4, pixels_per_cell=(cellSize, cellSize),
                cells_per_block=(1, 1), visualise=False)
    

    cv2.imshow("HOG state", state_fd)
    cv2.imshow("HOG goal", goal_fd)
    
    return np.linalg.norm(state_fd - goal_fd)
