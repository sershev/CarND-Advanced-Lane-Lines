from calibrate_camera import CalibrationUtils
from utils import Utils
from roi_utils import Roi
from moviepy.editor import VideoFileClip
import sys
import numpy as np

print("Run with args: ")
print(sys.argv)
print()

# initialization
mtx, dist = CalibrationUtils.load_calibration()
clip = VideoFileClip(sys.argv[1])
X,Y,Z = clip.w, clip.h, 3

roishape = np.array([[[0, Y-10], 
                      [int(X/2)-100, Y-int(Y/2.5)], 
                      [int(X/2)+100, Y-int(Y/2.5)], 
                      [X,Y-10]]], 
                      dtype=np.int32)

roi_dst_shape = np.array([[[int(X/4), Y-10], 
                           [int(X/2)-400, 0], 
                           [int(X/2)+400, 0], 
                           [X-int(X/4),Y-10]]], 
                           dtype=np.int32)

def process_image(image):
    # undistort
    undistorted_image = CalibrationUtils.undistort_img(image, mtx, dist)

    # select region
    roi = Roi.select_roi(undistorted_image, roishape)
    #Utils.compare_before_after(image, roi)

    # transform
    transformed_roi = Roi.transform(roi, roishape.reshape(4,2), roi_dst_shape)
    Utils.compare_before_after(roi, transformed_roi)
    
    # TODO - color gradient threshold
    threshold = Roi.threshold(transformed_roi)
    Utils.compare_before_after(roi, threshold)

    # TODO - detect lane

    # TODO - calc curvature

    return undistorted_image



video_output = "out_" + sys.argv[1]
frame = clip.fl_image(process_image)
frame.write_videofile(video_output, audio=False)

