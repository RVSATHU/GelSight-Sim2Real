# -*- coding: utf-8 -*-
import glob, cv2
import math
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from fast_poisson import fast_poisson
from utils import generate_ball, get_current_path


def matching_v2(test_img, ref_blur, cali, table, blur_inverse):
    """get gradient values from LUT"""

    diff_temp1 = test_img - ref_blur
    diff_temp2 = diff_temp1 * blur_inverse

    diff_temp2[:, :, 0] = (diff_temp2[:, :, 0] - cali.zero_point[0]) / cali.abs_range[0]
    diff_temp2[:, :, 1] = (diff_temp2[:, :, 1] - cali.zero_point[1]) / cali.abs_range[1]
    diff_temp2[:, :, 2] = (diff_temp2[:, :, 2] - cali.zero_point[2]) / cali.abs_range[2]
    diff_temp3 = np.clip(diff_temp2, 0, 0.999)
    diff = (diff_temp3 * cali.bin_num).astype(int)

    grad_img = table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]

    return grad_img


def handle_object(_shape, _target_path, _cali, _table_real, _table_gan, _ref_blur_real, _ref_blur_gan,
                  _blur_inverse_real, _blur_inverse_gan,
                  save=False, show=True):
    img_list = glob.glob(_target_path + _shape + "_[0-9]*_real.png")

    percentage_improved_list = []
    for real_img_path in img_list:
        filepath, tmpfilename = os.path.split(real_img_path)
        sim_img_path = os.path.join(filepath, tmpfilename.replace('_real', ''))

        disp_img_real = cv2.imread(real_img_path).copy()
        disp_img_gan = cv2.imread(sim_img_path).copy()

        test_img_real = cv2.imread(real_img_path)
        test_img_gan = cv2.imread(sim_img_path)

        grad_img_real = matching_v2(test_img_real, _ref_blur_real, _cali, _table_real, _blur_inverse_real)
        grad_img_gan = matching_v2(test_img_gan, _ref_blur_gan, _cali, _table_gan, _blur_inverse_gan)

        depth_real = fast_poisson(grad_img_real[:, :, 0], grad_img_real[:, :, 1])
        depth_real[depth_real < 0] = 0

        depth_gan = fast_poisson(grad_img_gan[:, :, 0], grad_img_gan[:, :, 1])
        depth_gan[depth_gan < 0] = 0

        mask = np.zeros_like(depth_real)
        points = np.array([[161, 137], [212, 132], [250, 193]], dtype=np.int32)
        points = points.reshape((-1, 1, 2))

        gt_depth_value = float(tmpfilename.replace(_shape, '').split('_')[1])

        if _shape == "cylinder":
            mask = cv2.circle(mask, (208, 166), 43, (255, 255, 255), -1, cv2.LINE_AA).astype(np.uint8)
            gt_depth = gt_depth_value
        elif _shape == "prism":
            mask = cv2.rectangle(mask, (172, 114,), (172 + 70, 114 + 106), (255, 255, 255), -1, cv2.LINE_AA).astype(
                np.uint8)
            gt_depth = gt_depth_value
        elif _shape == "triangle":
            mask = cv2.fillPoly(mask, [points], color=(255, 255, 255), ).astype(np.uint8)
            gt_depth = gt_depth_value
        elif _shape == "sphere":
            gt_depth, r = generate_ball(2.92, 208, 166, 2.92 - gt_depth_value)
            mask = cv2.circle(mask, (208, 166), int(r * 0.85), (255, 255, 255), -1, cv2.LINE_AA).astype(np.uint8)
        elif _shape == "large_sphere":
            gt_depth, r = generate_ball(9.85 / 2, 203, 167, 9.85 / 2 - gt_depth_value)
            mask = cv2.circle(mask, (203, 167), int(r * 0.85), (255, 255, 255), -1, cv2.LINE_AA).astype(np.uint8)

        grad_x, grad_y = np.gradient(mask)
        grad_total = np.sqrt(grad_x ** 2 + grad_y ** 2)
        mask_edge = np.zeros_like(mask)
        mask_edge[grad_total > 10] = 255
        mask_edge[grad_total <= 10] = 0
        mask_edge = mask_edge.astype(np.uint8)

        rms_error_real = math.sqrt(np.mean((depth_real - gt_depth)[mask > 127] ** 2))  # np.mean(depth_real[mask > 127])
        rms_error_gan = math.sqrt(np.mean((depth_gan - gt_depth)[mask > 127] ** 2))

        percentage_improved = (rms_error_real - rms_error_gan) / rms_error_real * 100
        percentage_improved_list.append(percentage_improved)

        max_depth_real = np.max(depth_real)
        max_depth_gan = np.max(depth_gan)
        max_all = max(max_depth_real, max_depth_gan)
        fig = plt.figure(figsize=(6, 7))

        plt.subplot(221)
        disp_img_real = disp_img_real + cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 8
        disp_img_real = np.clip(disp_img_real, 0, 255).astype(np.uint8)
        plt.title("Real")
        plt.imshow(cv2.cvtColor(disp_img_real, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(222)
        disp_img_gan = disp_img_gan + cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 8
        disp_img_gan = np.clip(disp_img_gan, 0, 255).astype(np.uint8)
        plt.title("Real2Sim")
        plt.imshow(cv2.cvtColor(disp_img_gan, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(223)
        depth_real[mask_edge > 0] = 0
        plt.imshow(depth_real, vmin=0, vmax=max_all)
        # plt.colorbar(orientation='horizontal', fraction=0.045, pad=0.1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("RMSE:{:.3f} mm".format(rms_error_real))
        plt.title("Depth from Real")

        plt.subplot(224)
        depth_gan[mask_edge > 0] = 0
        plt.imshow(depth_gan, vmin=0, vmax=max_all)

        plt.xticks([])
        plt.yticks([])
        plt.xlabel("RMSE:{:.3f} mm".format(rms_error_gan))
        plt.title("Depth from Real2Sim")

        plt.subplots_adjust(bottom=0.12, left=0.05, right=0.95, top=0.9)
        cax = plt.axes([0.05, 0.05, 0.9, 0.03])
        plt.colorbar(cax=cax, orientation='horizontal')

        plt.suptitle("Object: {}, GT Depth: {:.2f} mm".format(_shape, gt_depth_value))
        if save:
            if not os.path.isdir(os.path.join(get_current_path(), "result")):
                os.mkdir(os.path.join(get_current_path(), "result"))
            plt.savefig(os.path.join(get_current_path(), "result", tmpfilename.replace('_real', '')), dpi=250)
        if show:
            plt.show()

    return percentage_improved_list
