import numpy as np
import cv2



class Roi(object):

    @staticmethod
    def select_roi(img, vertices):
        """
        Select region of interes based on provided verticles/points.
        """

        mask = np.zeros_like(img)   

        ignore_mask_color = (255,) * 3

        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def transform(img, src_points, dst_points):
        """
        Changes the perspective of the image from src_points do dst_points.
        """
        #import pdb
        #pdb.set_trace()
        src_points, dst_points = np.float32(src_points), np.float32(dst_points)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img, M, img.shape[0:2][::-1])
        return warped

    def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        """
        Apply binary thresholding to combination of image
        S-Channel and sobel filter in X-direction for L-Channel.
        """
        img = np.copy(img)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:,:,1]
        s_channel = hsv[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        #hot sobel border fix
        sobelx[:,:10] = 0
        sobelx[:,-10:] = 0
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        binary_combined = np.maximum( sxbinary, s_binary )
        return binary_combined



    