from calibrate_camera import CalibrationUtils
from utils import Utils
from roi_utils import Roi
from lane import Lane
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
left_fit_arr = []
right_fit_arr = []


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
    #Utils.display_image(image, "Undistorted road image")
    
    undistorted_image = CalibrationUtils.undistort_img(image, mtx, dist)
    #Utils.compare_before_after(image, undistorted_image, "Raw Image", "Undistorted Image")

    # select region
    #roi = Roi.select_roi(undistorted_image, roishape)
    #Utils.compare_before_after(undistorted_image, roi, "Undistorted Image", "ROI")

    # transform
    transformed_roi = Roi.transform(undistorted_image, roishape.reshape(4,2), roi_dst_shape)
    #Utils.compare_before_after(undistorted_image, transformed_roi, "Undistorted", "Wraped ROI")
    
    # image threshold
    threshold = Roi.threshold(transformed_roi)
    #Utils.compare_before_after(transformed_roi, threshold, "Wraped ROI", "Threshold Image")

    # detect lane
    global left_fit_arr
    global right_fit_arr
    left_fit_arr, right_fit_arr = Lane.detect(threshold, left_fit_arr, right_fit_arr)
    left_fit, right_fit = np.mean(left_fit_arr, axis=0), np.mean(right_fit_arr, axis=0)
    #result = Lane.visualize_detection(threshold, left_fit, right_fit)

    # calc curvature
    left_curve, right_curve = Lane.curvature(left_fit, right_fit)
    final_curve = (left_curve + right_curve) / 2

    # draw lane
    left_fit, right_fit = left_fit_arr[-1], right_fit_arr[-1]
    with_lane = Utils.draw_lane(undistorted_image, threshold, left_fit, right_fit, roi_dst_shape, roishape.reshape(4,2))
    #Utils.compare_before_after(undistorted_image, with_lane)

    with_lane = Utils.draw_curve(with_lane, final_curve)
    #Utils.compare_before_after(undistorted_image, with_lane, "Undistorted Image", "Undist. Image with lane")
    #Utils.display_image(with_lane, "Final Image")
    return with_lane



video_output = "out_" + sys.argv[1]
frame = clip.fl_image(process_image)
frame.write_videofile(video_output, audio=False)

