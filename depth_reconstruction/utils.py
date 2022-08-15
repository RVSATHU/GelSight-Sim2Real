import os
import sys
import numpy as np
import math


def generate_ball(R, center_p_x, center_p_y, center_z, pix2mm=5.9259259259e-2):
    w = 320
    h = 320

    x = center_p_x * pix2mm
    y = center_p_y * pix2mm

    depth = np.zeros((h, w)).astype(np.float32)
    # Generate ball contact
    for i in range(320):
        p_x = i * pix2mm
        for j in range(320):
            p_y = j * pix2mm
            dx = p_x - x
            dy = p_y - y
            dz = center_z
            tmp = R ** 2 - dx ** 2 - dy ** 2
            if tmp > dz ** 2:
                depth[j, i] = - center_z + math.sqrt(tmp)

    return depth, math.sqrt(np.clip(R ** 2 - dz ** 2, 0, 1e20))/pix2mm


def try_make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_current_path():
    paths = sys.path
    current_file = os.path.basename(__file__)
    for path in paths:
        try:
            if current_file in os.listdir(path):
                return path
                break
        except Exception as e:
            pass
            # print(e)

