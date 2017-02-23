from calibrate_camera import CalibrationUtils
from utils import Utils
from roi_utils import Roi
from moviepy.editor import VideoFileClip
import sys
import numpy as np

import matplotlib.pyplot as plt

print("Run with args: ")
print(sys.argv)
print()

# initialization
mtx, dist = CalibrationUtils.load_calibration()
clip = VideoFileClip(sys.argv[1])
X,Y,Z = clip.w, clip.h, 3

roishape = np.array([[[0, Y-30], 
                      [int(X/2)-100, Y-int(Y/2.7)], 
                      [int(X/2)+100, Y-int(Y/2.7)], 
                      [X,Y-30]]], 
                      dtype=np.int32)

roi_dst_shape = np.array([[[0, Y-30], 
                           [0, 0], 
                           [X, 0], 
                           [X,Y-30]]], 
                           dtype=np.int32)

def process_image(image):
    # undistort
    undistorted_image = CalibrationUtils.undistort_img(image, mtx, dist)

    # select region
    roi = Roi.select_roi(undistorted_image, roishape)
    #Utils.compare_before_after(image, roi)

    # transform
    transformed_roi = Roi.transform(roi, roishape.reshape(4,2), roi_dst_shape)
    #Utils.compare_before_after(roi, transformed_roi)
    
    # image threshold
    threshold = Roi.threshold(transformed_roi)
    #Utils.compare_before_after(transformed_roi, threshold)

    # TODO - detect lane
    #import pdb
    #pdb.set_trace()
    histogram = np.sum(threshold[int(threshold.shape[0]/2):,:], axis=0)
    plt.plot(histogram)
    plt.show()

    histogram = np.sum(threshold[:int(threshold.shape[0]/2),:], axis=0)
    plt.plot(histogram)
    plt.show()

    # TODO - calc curvature

    return undistorted_image



video_output = "out_" + sys.argv[1]
frame = clip.fl_image(process_image)
frame.write_videofile(video_output, audio=False)

