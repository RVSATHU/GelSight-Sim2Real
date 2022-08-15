import cv2
import glob
import os

import numpy as np
from utils import try_make_dir
from calibration import calibration_customized, Calibration
from quantitative_depth import handle_object

shapes = ["triangle", "prism", "sphere", "large_sphere"]

if __name__ == "__main__":
    ball_radius = 2.92
    ref_img_real_path = './quantitative_real_gan/background_real.png'
    calibration_real_path = './quantitative_real_gan/'
    calibration_img_real_path_list = glob.glob(os.path.join(calibration_real_path, "calib_*_real.png"))
    calibration_real_file = "./real/calib_res.txt"
    calibration_real_table_path = './real/cali_table/'

    ref_img_gan_path = './quantitative_real_gan/background.png'
    calibration_gan_path = './quantitative_real_gan/'
    calibration_img_gan_path_list = glob.glob(os.path.join(calibration_real_path, "calib_*[0-9].png"))
    calibration_gan_file = "./sim/calib_res.txt"
    calibration_gan_table_path = './sim/cali_table/'

    try_make_dir(calibration_gan_table_path)
    try_make_dir(calibration_real_table_path)

    # press Esc to proceed
    calibration_customized(ball_radius, ref_img_real_path, calibration_img_real_path_list, calibration_real_file,
                           calibration_real_table_path, direct_pass=False)
    calibration_customized(ball_radius, ref_img_gan_path, calibration_img_gan_path_list, calibration_gan_file,
                           calibration_gan_table_path, direct_pass=False)

    # read calibration table
    table_real = np.load(os.path.join(calibration_real_table_path, 'cali_table_smooth.npy'))
    table_gan = np.load(os.path.join(calibration_gan_table_path, 'cali_table_smooth.npy'))

    cali = Calibration()

    # preprocessing of ref_img, mostly from GelSlim repo
    ref_img_real = cv2.imread(ref_img_real_path)
    ref_img_gan = cv2.imread(ref_img_gan_path)

    ref_blur_real = cv2.GaussianBlur(ref_img_real.astype(np.float32), (3, 3), 0) + 1
    blur_inverse_real = 1 + ((np.mean(ref_blur_real) / ref_blur_real) - 1) * 2

    ref_blur_gan = cv2.GaussianBlur(ref_img_gan.astype(np.float32), (3, 3), 0) + 1
    blur_inverse_gan = 1 + ((np.mean(ref_blur_gan) / ref_blur_gan) - 1) * 2

    # depth calculation and evaluation
    percentage_list = []
    for shape in shapes:
        percentage_list += handle_object(shape, './quantitative_real_gan/', cali, table_real, table_gan, ref_blur_real,
                                         ref_blur_gan,
                                         blur_inverse_real, blur_inverse_gan, save=True, show=True)

    print(percentage_list)
    print('AVG percentage improved: ', np.mean(np.array(percentage_list)))
