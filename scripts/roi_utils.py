import numpy as np
import cv2



class Roi(object):

    @staticmethod
    def select_roi(img, vertices):
        """
        Select region of interes
        """

        mask = np.zeros_like(img)   

        ignore_mask_color = (255,) * 3

        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def transform(img, src_points, dst_points):
        #import pdb
        #pdb.set_trace()
        src_points, dst_points = np.float32(src_points), np.float32(dst_points)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img, M, img.shape[0:2][::-1])
        return warped

    def threshold(image):
        return image



    