# -*- coding:utf8 -*-
import cv2
import numpy as np
from scipy import signal
import os
import time


class Calibration:
    def __init__(self, ball_radius=3, pix2mm=5.9259259259e-2, blur_size_up=13,
                 blur_size_down=7, blur_size_diff=5):
        self.blur_size_up = blur_size_up
        self.blur_size_down = blur_size_down
        self.blur_size_diff = blur_size_diff

        self.ball_radius = ball_radius
        self.pix2mm = pix2mm

        self.red_range = [-135, 135]
        self.green_range = [-135, 135]
        self.blue_range = [-135, 135]
        self.zero_point = [self.blue_range[0], self.green_range[0], self.red_range[0]]
        self.abs_range = [self.blue_range[1] - self.blue_range[0], self.green_range[1] - self.green_range[0],
                          self.red_range[1] - self.red_range[0]]
        self.bin_num = [135, 135, 135]

    def contact_detection_trim(self, raw_image, center, radius, ref, marker_mask, direct_pass=True):
        """manually trim the circle position and size、
           w/s/a/d for position，m/n for size
           press Esc to proceed to next image"""

        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0)
        diff_img = np.max(np.abs(raw_image.astype(np.float32) - blur), axis=2)

        key = -1
        x = center[0]
        y = center[1]
        while key != 27 and not direct_pass:
            center = (int(x), int(y))
            radius = int(radius)
            im2show = cv2.circle(np.array(raw_image), center, radius, (0, 40, 0), 2)
            cv2.imshow('w/a/s/d to move, m/n to change size, esc to continue', im2show.astype(np.uint8))
            key = cv2.waitKey(0)
            if key == 119:
                y -= 1
            elif key == 115:
                y += 1
            elif key == 97:
                x -= 1
            elif key == 100:
                x += 1
            elif key == 109:
                radius += 1
            elif key == 110:
                radius -= 1

        contact_mask = np.zeros_like(marker_mask)
        cv2.circle(contact_mask, center, radius, (1), -1)
        # contact_mask = contact_mask * (1 - marker_mask)

        return contact_mask, center, radius

    def get_gradient_v2(self, img, ref, center, radius_p, valid_mask, table, table_account):
        """calculate gradient values and store them into table"""

        ball_radius_p = self.ball_radius / self.pix2mm
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0) + 1
        blur_inverse = 1 + ((np.mean(blur) / blur) - 1) * 2
        # img_smooth = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
        diff_temp1 = img - blur
        diff_temp2 = diff_temp1 * blur_inverse

        diff_temp2[:, :, 0] = (diff_temp2[:, :, 0] - self.zero_point[0]) / self.abs_range[0]
        # zeropoint -90, lookscale 180
        diff_temp2[:, :, 1] = (diff_temp2[:, :, 1] - self.zero_point[1]) / self.abs_range[1]
        diff_temp2[:, :, 2] = (diff_temp2[:, :, 2] - self.zero_point[2]) / self.abs_range[2]
        diff_temp3 = np.clip(diff_temp2, 0, 0.999)
        diff = (diff_temp3 * self.bin_num).astype(int)
        pixels_valid = diff[valid_mask > 0]

        x = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y = np.linspace(0, img.shape[1] - 1, img.shape[1])
        xv, yv = np.meshgrid(y, x)
        xv = xv - center[0]
        yv = yv - center[1]
        rv = np.sqrt(xv ** 2 + yv ** 2)

        radius_p = min(radius_p, ball_radius_p - 1)
        mask = (rv < radius_p)
        mask_small = (rv < radius_p - 1)  # ？？？

        temp = ((xv * mask) ** 2 + (yv * mask) ** 2) * self.pix2mm ** 2
        height_map = (np.sqrt(self.ball_radius ** 2 - temp) * mask - np.sqrt(
            self.ball_radius ** 2 - (radius_p * self.pix2mm) ** 2)) * mask
        height_map[np.isnan(height_map)] = 0

        gx_num = signal.convolve2d(height_map, np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]), boundary='symm',
                                   mode='same') * mask_small
        gy_num = signal.convolve2d(height_map, np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]).T, boundary='symm',
                                   mode='same') * mask_small
        gradxseq = gx_num[valid_mask > 0]
        gradyseq = gy_num[valid_mask > 0]

        for i in range(gradxseq.shape[0]):
            b, g, r = pixels_valid[i, 0], pixels_valid[i, 1], pixels_valid[i, 2]
            if table_account[b, g, r] < 1.:
                table[b, g, r, 0] = gradxseq[i]
                table[b, g, r, 1] = gradyseq[i]
                table_account[b, g, r] += 1
            else:
                # average
                table[b, g, r, 0] = (table[b, g, r, 0] * table_account[b, g, r] + gradxseq[i]) / (
                        table_account[b, g, r] + 1)
                table[b, g, r, 1] = (table[b, g, r, 1] * table_account[b, g, r] + gradyseq[i]) / (
                        table_account[b, g, r] + 1)
                table_account[b, g, r] += 1

        return table, table_account

    def smooth_table(self, table, count_map):
        """对标定的table中没有值的部分进行填充"""

        y, x, z = np.meshgrid(np.linspace(0, self.bin_num[0] - 1, self.bin_num[0]),
                              np.linspace(0, self.bin_num[1] - 1, self.bin_num[1]),
                              np.linspace(0, self.bin_num[2] - 1, self.bin_num[2]))

        unfill_x = x[count_map < 1].astype(int)
        unfill_y = y[count_map < 1].astype(int)
        unfill_z = z[count_map < 1].astype(int)
        fill_x = x[count_map > 0].astype(int)
        fill_y = y[count_map > 0].astype(int)
        fill_z = z[count_map > 0].astype(int)

        fill_gradients = table[fill_x, fill_y, fill_z, :]
        table_new = np.array(table)
        temp_num = unfill_x.shape[0]
        for i in range(temp_num):
            if i > 0 and i % 10000 == 0:
                print('\n[INFO] Generating {}/{}\n'.format(i, temp_num))
            distance = (unfill_x[i] - fill_x) ** 2 + (unfill_y[i] - fill_y) ** 2 + (unfill_z[i] - fill_z) ** 2
            if np.min(distance) < 20:
                index = np.argmin(distance)
                table_new[unfill_x[i], unfill_y[i], unfill_z[i], :] = fill_gradients[index, :]

        return table_new


def calibration_customized(ball_radius, ref_img_path, calib_img_path_list, calib_file, calib_table_path,
                           direct_pass=False):
    cali = Calibration(ball_radius)

    ref_img = cv2.imread(ref_img_path)
    # ref_img = imp.crop_image(ref_img, pad_V, pad_H)

    # initialize lookup_table
    table = np.zeros((*cali.bin_num, 2))
    table_account = np.zeros(cali.bin_num)

    cv2.namedWindow('w/a/s/d to move, m/n to change size, esc to continue', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('w/a/s/d to move, m/n to change size, esc to continue', (1000, 1000))

    def parse_calib_res_file(file):
        with open(file, 'r') as f:
            lines = f.readlines()
            calib_res = {}
            for line in lines:
                line = line.strip().split(' ')
                name = line[0]
                center_x = int(line[1].strip('(').strip(','))
                center_y = int(line[2].strip(')'))
                center = (center_x, center_y)
                radius = int(line[3])

                calib_res[name] = (center, radius)
        return calib_res

    if os.path.exists(calib_file):
        calib_res = parse_calib_res_file(calib_file)
    else:
        calib_res = {}

    for calib_img_path in calib_img_path_list:
        filepath, name = os.path.split(calib_img_path)
        img = cv2.imread(calib_img_path)
        # img = imp.crop_image(img, pad_V, pad_H)

        marker_mask = np.zeros_like(img[:, :, 0])
        regular_marker_mask = np.zeros_like(img[:, :, 0])

        if name in calib_res.keys():
            center, radius_p = calib_res[name]
            print("Load calibrated data for {} from file.".format(name))
            print(center, radius_p)
        else:
            diff_img = np.max(np.abs(img.astype(np.float32) - ref_img.astype(np.float32)), axis=2)
            contact_mask = (diff_img > 50).astype(np.uint8) * (1 - marker_mask)
            contours, hierarchy = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            sorted_areas = np.sort(areas)
            if len(sorted_areas) > 0:
                cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour
                (x, y), radius_p = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius_p = int(radius_p)
            else:
                center = (100, 100)
                radius_p = 40
            calib_res[name] = (center, radius_p)

        try:
            valid_mask, center, radius_p = cali.contact_detection_trim(img, center, radius_p, ref_img,
                                                                       regular_marker_mask, direct_pass=direct_pass)
            table, table_account = cali.get_gradient_v2(img, ref_img, center, radius_p, valid_mask, table,
                                                        table_account)
            calib_res[name] = (center, radius_p)

        except Exception as e:
            print(e)
            # os.remove(cali_path+name)

    with open(calib_file, 'w') as file:
        for name in calib_res.keys():
            center, radius = calib_res[name]
            file.write(name + " " + str(center) + " " + str(radius) + "\n")

    cv2.destroyAllWindows()

    print('\n[INFO] Calibration table is generating\n')
    start = time.time()

    np.save(os.path.join(calib_table_path, 'cali_table_raw.npy'), table)
    np.save(os.path.join(calib_table_path, 'count_map.npy'), table_account)
    table = np.load(os.path.join(calib_table_path, 'cali_table_raw.npy'))
    table_account = np.load(os.path.join(calib_table_path, 'count_map.npy'))
    table_smooth = cali.smooth_table(table, table_account)
    np.save(os.path.join(calib_table_path, 'cali_table_smooth.npy'), table_smooth)
    end = time.time()
    print('[INFO] Calibration table is generated')
    print('[INFO] Time cost : {:.2f}min'.format((end - start) / 60))
