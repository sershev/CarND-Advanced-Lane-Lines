from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle
import warnings


class CalibrationUtils(object):

    @staticmethod
    def calibrate(cols=9, rows=6, path="./camera_cal/", debug=False):
        """
        Calibrate the camera based on images in camera_cal dir.
        """
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((cols*rows,3), np.float32)
        objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        search_str = path+'*.jpg'
        images = glob.glob(search_str)
        print("Found {0} images for calibration.".format(len(images)))

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (cols,rows), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners)

                if (debug):
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (cols,rows), corners,ret)
                    window_name = "calib image"
                    cv2.imshow('calib image',img)
                    cv2.moveWindow("calib image", 10, 50);
                    cv2.waitKey()
                    cv2.destroyAllWindows()

            else:
                print("No chessboard corners in {0} found!!!".format(fname))

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        return mtx, dist


    @staticmethod
    def save_calibration(mtx, dist_coeff, file_path="calibration_info"):
        payload = {
            "matrix": mtx,
            "dist_coeff": dist_coeff,
        }
        with open('calibration_info', 'wb') as outfile:
            pickle.dump(payload, outfile)



    @staticmethod
    def load_calibration(file_path="calibration_info" ):
        """
        Load existing serialized camera matrix and destortion coefficients
        or recalculate them if there is no file to load.
        """
        info_file = Path(file_path)
        if info_file.is_file():
            with open(file_path, 'rb') as infile:
                payload = pickle.load(infile)
                mtx = payload['matrix']
                dist_coeff = payload['dist_coeff']
                return mtx, dist_coeff
        else:
            warnings.warn("File {} do not exist! Try to recalibrate".format(file_path), UserWarning)
            mtx, dist_coeff = CalibrationUtils.calibrate()
            CalibrationUtils.save_calibration(mtx, dist_coeff, file_path)
            return mtx, dist_coeff

    @staticmethod
    def undistort_img(img, mtx, dist, debug=False):
        """
        Apply camera matrix and destoriton cefficients to undestort an image. 
        """
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        if (debug):
            window_name = "Undistorted Image"
            cv2.imshow('Undistorted Image', undist)
            cv2.moveWindow("Undistorted Image", 10, 50);
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return undist

    