import time

import numpy as np
import cv2
import numpy as np
import scipy.ndimage.filters as fi
import math


def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def gaussian_noise(image, sigma):
    row, col = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def derivative(mat, direction):
    assert (direction == 'x' or direction == 'y'), "The derivative direction must be 'x' or 'y'"
    kernel = None
    if direction == 'x':
        kernel = [[-1.0, 0.0, 1.0]]
    elif direction == 'y':
        kernel = [[-1.0], [0.0], [1.0]]
    kernel = np.array(kernel, dtype=np.float64)
    return cv2.filter2D(mat, -1, kernel) / 2.0


def tangent(mat):
    dx = derivative(mat, 'x')
    dy = derivative(mat, 'y')
    img_shape = np.shape(mat)
    _1 = np.repeat([1.0], img_shape[0] * img_shape[1]).reshape(img_shape).astype(dx.dtype)
    unormalized = cv2.merge((-dx, -dy, _1))
    norms = np.linalg.norm(unormalized, axis=2)
    return (unormalized / np.repeat(norms[:, :, np.newaxis], 3, axis=2))


def solid_color_img(color, size):
    image = np.zeros(size + (3,), np.float64)
    image[:] = color
    return image


def add_overlay(rgb, alpha, color):
    s = np.shape(alpha)

    opacity3 = np.repeat(alpha, 3).reshape((s[0], s[1], 3))  # * 10.0 # 透明度复制到3个通道上

    overlay = solid_color_img(color, s)  # 单色的图像

    foreground = opacity3 * overlay  # 加透明度衰减，alpha越大，color的占比越大
    background = rgb.astype(np.float64)
    #background = (1.0 - opacity3) * rgb.astype(np.float64)  # 为什么要在原图上加个比例？？？？
    res = background + foreground

    res[res > 255.0] = 255.0
    res[res < 0.0] = 0.0
    res = res.astype(np.uint8)

    return res


class PhongShadingRenderer:

    def __init__(self, **config):
        self.light_sources = config['light_sources']
        self.with_background = config['with_background'] if 'with_background' in config else False
        self.background = cv2.imread(config['background_img'])
        self.px2m_ratio = config['px2m_ratio']
        self.elastomer_thickness = config['elastomer_thickness']
        self.max_depth = config['max_depth']

        self.default_ks = 0.15
        self.default_kd = 0.5
        self.default_alpha = 5

        self.ka = config['ka'] or 0.8

        self.enable_depth_texture = config['enable_depth_texture'] if 'enable_depth_texture' in config else False
        self.texture_sigma = config['texture_sigma'] if 'texture_sigma' in config else 0.000001
        self.t = config['t'] if 't' in config else 3
        self.sigma = config['sigma'] if 'sigma' in config else 7
        self.kernel_size = config['sigma'] if 'sigma' in config else 21

        self.min_depth = self.max_depth - self.elastomer_thickness
        self.background_color = self._get_background_color()


    def _get_background_color(self):
        depth = np.zeros((10, 10))
        background_rendered = self._generate(depth, noise=False)
        return background_rendered.mean(axis=(0, 1))

    def _phong_illumination(self, depth, T, light_dir, kd, ks, alpha):

        dot = np.dot(T, np.array(light_dir)).astype(np.float64)  # 各点法向与光线来源方向的点积
        diffuse_l = dot * kd  # 散射光强度
        diffuse_l[diffuse_l < 0] = 0.0

        dot3 = np.repeat(dot[:, :, np.newaxis], 3, axis=2)

        R = 2.0 * dot3 * T - light_dir  # 镜面反射光的方向
        V = [0.0, 0.0, 1.0]  # 观察的方向

        spec_l = np.power(np.dot(R, V), alpha) * ks  # 镜面反射光强度

        return diffuse_l + spec_l

    def _generate(self, target_depth, noise=False):

        if noise:
            textured_elastomer_depth = gaussian_noise(target_depth, self.texture_sigma)  # 添加噪声
        else:
            textured_elastomer_depth = target_depth.copy()

        out = np.zeros((target_depth.shape[0], target_depth.shape[1], 3))

        self.depth = target_depth
        depth = (self.depth * 1000).copy()
        (h, w) = depth.shape
        x = np.linspace(0, (w - 1) * self.px2m_ratio * 1000, num=w)
        y = np.linspace(0, (h - 1) * self.px2m_ratio * 1000, num=h)

        xx, yy = np.meshgrid(x, y)
        self.points = np.dstack((xx, yy, depth))  # 保证是右手系

        T = tangent(textured_elastomer_depth / self.px2m_ratio)  # 计算深度法向

        # show_normalized_img('tangent', T)
        for light in self.light_sources.values():  # 对每一个光源
            ks = light['ks'] if 'ks' in light else self.default_ks  # 镜面反射系数
            kd = light['kd'] if 'kd' in light else self.default_kd  # 散射系数
            alpha = light['alpha'] if 'alpha' in light else self.default_alpha  # 镜面反射中的指数
            out = add_overlay(out, self._phong_illumination(target_depth, T, light['position'], kd, ks, alpha),
                              light['color'])  # 这里把light的position直接作为来源方向！也就是说position其实是指光的方向！

        return out

    def generate(self, target_depth, return_depth=False):
        out = self._generate(target_depth, noise=self.enable_depth_texture)
        if self.with_background:
            diff = (out.astype(np.float32) - solid_color_img(self.background_color, out.shape[:2])) * 1
            kernel = gkern2(5, 2)
            diff = cv2.filter2D(diff, -1, kernel)

            # Combine the simulated difference image with real background image
            result = self.ka * self.background.copy()

            result = np.clip((diff + result), 0, 255).astype(np.uint8)
        else:
            result = out
        if return_depth:
            return result, target_depth
        else:
            return result
