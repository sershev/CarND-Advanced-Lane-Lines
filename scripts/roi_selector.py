import numpy as np
import cv2



class RoiSelector(object):

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



    