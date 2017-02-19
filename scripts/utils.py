from calibrate_camera import CalibrationUtils
import matplotlib.pyplot as plt
import cv2

class Utils:

    @staticmethod
    def compare_before_after(img1, img2):
        fig = plt.figure()
        a=fig.add_subplot(1,2,1)
        imgplot = plt.imshow(img1)
        a.set_title('Before')
        a=fig.add_subplot(1,2,2)
        imgplot = plt.imshow(img2)
        a.set_title('After')
        plt.show()

    @staticmethod
    def test_camera_calibration():
        mtx,dist = CalibrationUtils.load_calibration()
        img = cv2.imread("./camera_cal/calibration1.jpg")
        img2 = CalibrationUtils.undistort_img(img, mtx, dist)
        CalibrationUtils.compare_before_after(img,img2)