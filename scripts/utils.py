from calibrate_camera import CalibrationUtils
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np

class Utils:

    @staticmethod
    def compare_before_after(img1, img2, title1 = "Before", title2 = "After"):
        fig = plt.figure()
        a=fig.add_subplot(1,2,1)
        imgplot = plt.imshow(img1)
        a.set_title(title1)
        a=fig.add_subplot(1,2,2)
        imgplot = plt.imshow(img2)
        a.set_title(title2)
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(0, 0, 640, 360)
        plt.show()

    @staticmethod
    def display_image(img1, title1 = "Image"):
        fig = plt.figure()
        a=fig.add_subplot(1,1,1)
        imgplot = plt.imshow(img1)
        a.set_title(title1)
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(0, 0, 640, 360)
        plt.show()


    @staticmethod
    def test_camera_calibration():
        mtx,dist = CalibrationUtils.load_calibration()
        img = cv2.imread("./camera_cal/calibration1.jpg")
        img2 = CalibrationUtils.undistort_img(img, mtx, dist)
        CalibrationUtils.compare_before_after(img,img2)

    @staticmethod
    def draw_lane(undist, warped, left_fit, right_fit, dst, src):
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )

        src, dst = np.float32(src), np.float32(dst)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (255,255, 0))
        
        lane_displacement = (((np.min(left_fitx) + np.max(right_fitx)) - undist.shape[1])/2) * 3.7/700
        if (lane_displacement > 0):
            curve_string = "The car is {:1.1f}m left from lane center.".format(lane_displacement)
        elif(lane_displacement < 0):
            curve_string = "The car is {:1.1f}m right from lane center.".format(lane_displacement*(-1))
        else:
            curve_string = "The car is in lane center.".format(lane_displacement)

        Minv = cv2.getPerspectiveTransform(dst, src)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
        cv2.putText(newwarp, curve_string, (300, 80),  cv2.FONT_HERSHEY_DUPLEX, 1, (255,255, 0))
        #cv2.circle(newwarp, (int(np.max(left_fitx)), newwarp.shape[0]), 10, (255,0,255))
        #cv2.circle(newwarp, (int(np.max(right_fitx)), newwarp.shape[0]), 10, (255,0,255))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)
        return result


    def draw_curve(image, curve):

        tmp = np.zeros_like(image)
        curve_string = "The curvature is: {:4.1f}m".format(curve)
        cv2.putText(tmp, curve_string, (300, 50),  cv2.FONT_HERSHEY_DUPLEX, 1, (255,255, 0))
        result = cv2.addWeighted(image, 1, tmp, 0.6, 0)
        return result