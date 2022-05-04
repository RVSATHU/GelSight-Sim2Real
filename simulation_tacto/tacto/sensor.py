# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import warnings
import collections
from dataclasses import dataclass

import cv2
import numpy as np
import pybullet as p
import trimesh
import scipy.ndimage.filters as fi

from urdfpy import URDF

from .renderer import Renderer

logger = logging.getLogger(__name__)


def _get_default_config(filename):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)


def get_digit_config_path():
    return _get_default_config("config_digit.yml")


def get_omnitact_config_path():
    return _get_default_config("config_omnitact.yml")


def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def DoG(depth, kernel_size_small, sigma_small, kernel_size_big, sigma_big, t):
    kernel_small = gkern2(kernel_size_small, sigma_small)
    kernel_big = gkern2(kernel_size_big, sigma_big)

    filtered_small = depth.copy()
    filtered_big = depth.copy()

    for i in range(2 * t):
        filtered_small_tmp = cv2.filter2D(filtered_small, -1, kernel_small)
        filtered_small = np.maximum(filtered_small_tmp, depth)

    for i in range(t):
        filtered_big_tmp = cv2.filter2D(filtered_big, -1, kernel_big)
        filtered_big = np.maximum(filtered_big_tmp, depth)

    _DoG = 2 * filtered_small - filtered_big

    kernel_last = gkern2(5, 3)
    _DoG = cv2.filter2D(_DoG, -1, kernel_last)
    return _DoG

@dataclass
class Link:
    obj_id: int  # pybullet ID
    link_id: int  # pybullet link ID (-1 means base)
    cid: int  # physicsClientId

    def get_pose(self):
        if self.link_id < 0:
            # get the base pose if link ID < 0
            position, orientation = p.getBasePositionAndOrientation(
                self.obj_id, physicsClientId=self.cid
            )
        else:
            # get the link pose if link ID >= 0
            position, orientation = p.getLinkState(
                self.obj_id, self.link_id, physicsClientId=self.cid
            )[:2]

        orientation = p.getEulerFromQuaternion(orientation, physicsClientId=self.cid)
        return position, orientation


class Sensor:
    def __init__(
        self,
        width=120,
        height=160,
        config_path=get_digit_config_path(),
        visualize_gui=True,
        show_depth=True,
        zrange=0.002,
        cid=0,
    ):
        """

        :param width: scalar
        :param height: scalar
        :param background: image
        :param visualize_gui: Bool
        :param show_depth: Bool
        :param config_path:
        :param cid: Int
        """
        self.cid = cid
        self.renderer = Renderer(width, height, config_path)

        self.visualize_gui = visualize_gui
        self.show_depth = show_depth
        self.zrange = zrange

        self.cameras = {}
        self.nb_cam = 0
        self.objects = {}
        self.object_poses = {}
        self.normal_forces = {}
        self._static = None

    @property
    def height(self):
        return self.renderer.height

    @property
    def width(self):
        return self.renderer.width

    @property
    def background(self):
        return self.renderer.background

    def add_camera(self, obj_id, link_ids):
        """
        Add camera into tacto

        self.cameras format: {
            "cam0": Link,
            "cam1": Link,
            ...
        }
        """
        if not isinstance(link_ids, collections.abc.Sequence):
            link_ids = [link_ids]

        for link_id in link_ids:
            cam_name = "cam" + str(self.nb_cam)
            self.cameras[cam_name] = Link(obj_id, link_id, self.cid)
            self.nb_cam += 1

    def add_object(self, urdf_fn, obj_id, globalScaling=1.0):
        # Load urdf file by urdfpy
        robot = URDF.load(urdf_fn)

        for link_id, link in enumerate(robot.links):
            if len(link.visuals) == 0:
                continue
            link_id = link_id - 1
            # Get each links
            visual = link.visuals[0]
            obj_trimesh = visual.geometry.meshes[0]

            # Set mesh color to default (remove texture)
            obj_trimesh.visual = trimesh.visual.ColorVisuals()

            # Set initial origin (pybullet pose already considered initial origin position, not orientation)
            pose = visual.origin

            # Scale if it is mesh object (e.g. STL, OBJ file)
            mesh = visual.geometry.mesh
            if mesh is not None and mesh.scale is not None:
                S = np.eye(4, dtype=np.float64)
                S[:3, :3] = np.diag(mesh.scale)
                pose = pose.dot(S)

            # Apply interial origin if applicable
            inertial = link.inertial
            if inertial is not None and inertial.origin is not None:
                pose = np.linalg.inv(inertial.origin).dot(pose)

            # Set global scaling
            pose = np.diag([globalScaling] * 3 + [1]).dot(pose)

            obj_trimesh = obj_trimesh.apply_transform(pose)
            obj_name = "{}_{}".format(obj_id, link_id)

            self.objects[obj_name] = Link(obj_id, link_id, self.cid)
            position, orientation = self.objects[obj_name].get_pose()

            # Add object in pyrender
            self.renderer.add_object(
                obj_trimesh,
                obj_name,
                position=position,  # [-0.015, 0, 0.0235],
                orientation=orientation,  # [0, 0, 0],
            )

    def add_body(self, body):
        self.add_object(
            body.urdf_path, body.id, globalScaling=body.global_scaling or 1.0
        )

    def loadURDF(self, *args, **kwargs):
        warnings.warn(
            "\33[33mSensor.loadURDF is deprecated. Please use body = "
            "pybulletX.Body(...) and Sensor.add_body(body) instead\33[0m."
        )
        """
        Load the object urdf to pybullet and tacto simulator.
        The tacto simulator will create the same scene in OpenGL for faster rendering
        """
        urdf_fn = args[0]
        globalScaling = kwargs.get("globalScaling", 1.0)

        # Add to pybullet
        obj_id = p.loadURDF(physicsClientId=self.cid, *args, **kwargs)

        # Add to tacto simulator scene
        self.add_object(urdf_fn, obj_id, globalScaling=globalScaling)

        return obj_id

    def update(self):
        warnings.warn(
            "\33[33mSensor.update is deprecated and renamed to ._update_object_poses()"
            ", which will be called automatically in .render()\33[0m"
        )

    def _update_object_poses(self):
        """
        Update the pose of each objects registered in tacto simulator
        """
        for obj_name in self.objects.keys():
            self.object_poses[obj_name] = self.objects[obj_name].get_pose()

    def get_force(self, cam_name):
        # Load contact force

        obj_id = self.cameras[cam_name].obj_id
        link_id = self.cameras[cam_name].link_id

        pts = p.getContactPoints(
            bodyA=obj_id, linkIndexA=link_id, physicsClientId=self.cid
        )

        # accumulate forces from 0. using defaultdict of float
        self.normal_forces[cam_name] = collections.defaultdict(float)

        for pt in pts:
            body_id_b = pt[2]
            link_id_b = pt[4]

            obj_name = "{}_{}".format(body_id_b, link_id_b)

            # ignore contacts we don't care (those not in self.objects)
            if obj_name not in self.objects:
                continue

            # Accumulate normal forces
            self.normal_forces[cam_name][obj_name] += pt[9]

        return self.normal_forces[cam_name]

    @property
    def static(self):
        if self._static is None:
            depths = [np.zeros_like(d0) for d0 in self.renderer.depth0]
            colors = []
            for depth in self.renderer.depth0:
                color, _ = self.renderer.render_from_depth(depth = depth)
                colors.append(color)
            self._static = (colors, depths)

        return self._static

    def _render_static(self):
        colors, depths = self.static
        colors = [self.renderer._add_noise(color) for color in colors]
        return colors, depths

    def render(self):
        """
        Render tacto images from each camera's view.
        """

        self._update_object_poses()

        colors = []
        depths = []

        for i in range(self.nb_cam):
            cam_name = "cam" + str(i)

            # get the contact normal forces
            normal_forces = self.get_force(cam_name)
            contact = False

            if normal_forces:
                contact = True
                position, orientation = self.cameras[cam_name].get_pose()
                self.renderer.update_camera_pose(position, orientation)
                depth = self.renderer.render(self.object_poses, normal_forces)  # depth map in camera's coordinate

                for j in range(len(depth)):
                    depth[j] = self.renderer.depth0[j] - depth[j]  # extract the deformation to perform filtering
                    depth[j] = DoG(depth[j], 5, 3, 7, 7, 3)
                    depth[j] = self.renderer.depth0[j] - depth[j]

                color, depth = self.renderer.render_from_depth(depth[0])
                depth = [depth]
                for j in range(len(depth)):
                    depth[j] = self.renderer.depth0[j] - depth[j]
                color = [color]

            else:
                color, depth = self._render_static()

            colors += color
            depths += depth
        return colors, depths, contact

    def _depth_to_color(self, depth):
        gray = (np.clip(depth / self.zrange, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def updateGUI(self, colors, depths):
        """
        Update images for visualization
        """
        if not self.visualize_gui:
            return

        # concatenate colors horizontally (axis=1)
        color = np.concatenate(colors, axis=1)

        if self.show_depth:
            # concatenate depths horizontally (axis=1)
            depth = np.concatenate(list(map(self._depth_to_color, depths)), axis=1)

            # concatenate the resulting two images vertically (axis=0)
            color_n_depth = np.concatenate([color, depth], axis=0)

            cv2.imshow("color and depth", color_n_depth)
        else:
            cv2.imshow("color", color)

        cv2.waitKey(1)
