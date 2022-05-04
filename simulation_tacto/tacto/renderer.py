# This source code is modified based on Tacto (by Facebook Inc.) and https://github.com/danfergo/gelsight_simulation.

"""
Set backend platform for OpenGL render (pyrender.OffscreenRenderer)
- Pyglet, the same engine that runs the pyrender viewer. This requires an active
  display manager, so you can’t run it on a headless server. This is the default option.
- OSMesa, a software renderer. require extra install OSMesa.
  (https://pyrender.readthedocs.io/en/latest/install/index.html#installing-osmesa)
- EGL, which allows for GPU-accelerated rendering without a display manager.
  Requires NVIDIA’s drivers.

The handle for EGL is egl (preferred, require NVIDIA driver),
The handle for OSMesa is osmesa.
Default is pyglet, which requires active window
"""

# import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"
import copy
import logging

import numpy as np
import cv2
import pyrender
import trimesh
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

import pybullet as p
from .phong_shading import PhongShadingRenderer
import math

logger = logging.getLogger(__name__)


def euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0]):
    q = p.getQuaternionFromEuler(angles)
    r = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)

    pose = np.eye(4)
    pose[:3, 3] = translation
    pose[:3, :3] = r
    return pose


class Renderer:
    def __init__(self, width, height, config_path):
        """

        :param width: scalar
        :param height: scalar
        :param background: image
        :param config_path:
        """
        self._width = width
        self._height = height

        logger.info("Loading configuration from: %s" % config_path)
        self.conf = OmegaConf.load(config_path)

        self.force_enabled = (
                self.conf.sensor.force is not None and self.conf.sensor.force.enable
        )

        if self.force_enabled:
            self.min_force = self.conf.sensor.force.range_force[0]
            self.max_force = self.conf.sensor.force.range_force[1]
            self.max_deformation = self.conf.sensor.force.max_deformation

        self.phong_shading_renderer = PhongShadingRenderer(**self.conf.sensor)
        self._init_pyrender()

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def _init_pyrender(self):
        """
        Initialize pyrender
        """
        # Create scene for pybullet sync
        self.scene = pyrender.Scene()

        """
        objects format:
            {obj_name: pyrender node}
        """
        self.object_nodes = {}
        self.current_object_nodes = {}

        self._init_gel()
        self._init_camera()

        self.r = pyrender.OffscreenRenderer(self.width, self.height)

        depths = self.render(object_poses=None)
        colors = []
        for depth in depths:
            color, _ = self.render_from_depth(depth)
            colors.append(color)
        self.depth0 = depths
        self._background_sim = colors


    def _init_gel(self):
        """
        Add gel surface in the scene
        """
        # Create gel surface (flat/curve surface based on config file)
        gel_trimesh = self._generate_gel_trimesh()

        mesh_gel = pyrender.Mesh.from_trimesh(gel_trimesh, smooth=False)
        self.gel_pose0 = np.eye(4)
        self.gel_node = pyrender.Node(mesh=mesh_gel, matrix=self.gel_pose0)
        self.scene.add_node(self.gel_node)

    def _generate_gel_trimesh(self):

        # Load config
        g = self.conf.sensor.gel
        origin = g.origin

        X0, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = g.width, g.height

        if hasattr(g, "mesh") and g.mesh is not None:
            gel_trimesh = trimesh.load(g.mesh)

            # scale up for clearer indentation
            matrix = np.eye(4)
            matrix[[0, 1, 2], [0, 1, 2]] = 1.02
            gel_trimesh = gel_trimesh.apply_transform(matrix)

        elif not g.curvature:
            # Flat gel surface
            gel_trimesh = trimesh.Trimesh(
                vertices=[
                    [X0, Y0 + W / 2, Z0 + H / 2],
                    [X0, Y0 + W / 2, Z0 - H / 2],
                    [X0, Y0 - W / 2, Z0 - H / 2],
                    [X0, Y0 - W / 2, Z0 + H / 2],
                ],
                faces=[[0, 1, 2], [2, 3, 0]],
            )
        else:
            # Curved gel surface
            N = g.countW
            M = int(N * H / W)
            R = g.R
            zrange = g.curvatureMax

            y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
            z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
            yy, zz = np.meshgrid(y, z)

            h = R - np.maximum(0, R ** 2 - (yy - Y0) ** 2 - (zz - Z0) ** 2) ** 0.5
            xx = X0 - zrange * h / h.max()

            gel_trimesh = self._generate_trimesh_from_depth(xx)

        return gel_trimesh

    def _generate_trimesh_from_depth(self, depth):
        # Load config
        g = self.conf.sensor.gel
        origin = g.origin

        _, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = g.width, g.height

        N = depth.shape[1]
        M = depth.shape[0]

        # Create grid mesh
        vertices = []
        faces = []

        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)

        # Vertex format: [x, y, z]
        vertices = np.zeros([N * M, 3])

        # Add x, y, z position to vertex
        vertices[:, 0] = depth.reshape([-1])
        vertices[:, 1] = yy.reshape([-1])
        vertices[:, 2] = zz.reshape([-1])

        # Create faces

        faces = np.zeros([(N - 1) * (M - 1) * 6], dtype=np.uint)

        # calculate id for each vertex: (i, j) => i * m + j
        xid = np.arange(N)
        yid = np.arange(M)
        yyid, xxid = np.meshgrid(xid, yid)
        ids = yyid[:-1, :-1].reshape([-1]) + xxid[:-1, :-1].reshape([-1]) * N

        # create upper triangle
        faces[::6] = ids  # (i, j)
        faces[1::6] = ids + N  # (i+1, j)
        faces[2::6] = ids + 1  # (i, j+1)

        # create lower triangle
        faces[3::6] = ids + 1  # (i, j+1)
        faces[4::6] = ids + N  # (i+1, j)
        faces[5::6] = ids + N + 1  # (i+1, j+1)

        faces = faces.reshape([-1, 3])
        gel_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        return gel_trimesh

    def _init_camera(self):
        """
        Set up camera
        """

        self.camera_nodes = []
        self.camera_zero_poses = []

        conf_cam = self.conf.sensor.camera
        self.nb_cam = len(conf_cam)

        for i in range(self.nb_cam):
            cami = conf_cam[i]

            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(cami.yfov), znear=cami.znear,
            )
            camera_zero_pose = euler2matrix(
                angles=np.deg2rad(cami.orientation), translation=cami.position,
            )
            self.camera_zero_poses.append(camera_zero_pose)

            # Add camera node into scene
            camera_node = pyrender.Node(camera=camera, matrix=camera_zero_pose)
            self.scene.add_node(camera_node)
            self.camera_nodes.append(camera_node)

    def add_object(
            self, objTrimesh, obj_name, position=[0, 0, 0], orientation=[0, 0, 0]
    ):
        """
        Add object into the scene
        """

        mesh = pyrender.Mesh.from_trimesh(objTrimesh)
        pose = euler2matrix(angles=orientation, translation=position)
        obj_node = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(obj_node)

        self.object_nodes[obj_name] = obj_node
        self.current_object_nodes[obj_name] = obj_node

    def update_camera_pose(self, position, orientation):
        """
        Update sensor pose (including camera, lighting, and gel surface)
        """

        pose = euler2matrix(angles=orientation, translation=position)

        # Update camera
        for i in range(self.nb_cam):
            camera_pose = pose.dot(self.camera_zero_poses[i])
            self.camera_nodes[i].matrix = camera_pose

        # Update gel
        gel_pose = pose.dot(self.gel_pose0)
        self.gel_node.matrix = gel_pose

    def update_object_pose(self, obj_name, position, orientation):
        """
        orientation: euler angles
        """

        node = self.object_nodes[obj_name]
        pose = euler2matrix(angles=orientation, translation=position)
        self.scene.set_pose(node, pose=pose)

    def _add_noise(self, color):
        """
        Add Gaussian noise to the RGB image
        :param color:
        :return:
        """
        # Add noise to the RGB image
        mean = self.conf.sensor.noise.color.mean
        std = self.conf.sensor.noise.color.std

        if mean != 0 or std != 0:
            noise = np.random.normal(mean, std, color.shape)  # Gaussian noise
            color = np.clip(color + noise, 0, 255).astype(
                np.uint8
            )  # Add noise and clip

        return color


    def adjust_with_force(
            self, camera_pos, camera_ori, normal_forces, object_poses,
    ):
        """
        Adjust object pose with normal force feedback
        The larger the normal force, the larger indentation
        Currently linear adjustment from force to shift distance
        It can be replaced by non-linear adjustment with calibration from real sensor
        """
        existing_obj_names = list(self.current_object_nodes.keys())
        for obj_name in existing_obj_names:
            # Remove object from scene if not in contact
            if obj_name not in normal_forces:
                self.scene.remove_node(self.current_object_nodes[obj_name])
                self.current_object_nodes.pop(obj_name)

        # Add/Update the objects' poses the scene if in contact
        for obj_name in normal_forces:
            if obj_name not in object_poses:
                continue
            obj_pos, objOri = object_poses[obj_name]

            # Add the object node to the scene
            if obj_name not in self.current_object_nodes:
                node = self.object_nodes[obj_name]
                self.scene.add_node(node)
                self.current_object_nodes[obj_name] = node

            if self.force_enabled:
                offset = -1.0
                if obj_name in normal_forces:
                    offset = (
                            min(self.max_force, normal_forces[obj_name]) / self.max_force
                    )

                # Calculate pose changes based on normal force
                camera_pos = np.array(camera_pos)
                obj_pos = np.array(obj_pos)

                direction = camera_pos - obj_pos
                direction = direction / (np.sum(direction ** 2) ** 0.5 + 1e-6)
                obj_pos = obj_pos + offset * self.max_deformation * direction

            self.update_object_pose(obj_name, obj_pos, objOri)

    def _post_process(self, color, depth, camera_index, noise=True, calibration=True):
        if calibration:
            color = self._calibrate(color, camera_index)
        if noise:
            color = self._add_noise(color)
        return color, depth

    def render(
            self, object_poses=None, normal_forces=None,
    ):
        """

        :param object_poses:
        :param normal_forces:
        :param noise:
        :return:
        """
        depths = []

        for i in range(self.nb_cam):
            # Set the main camera node for rendering
            self.scene.main_camera_node = self.camera_nodes[i]

            # Adjust contact based on force
            if object_poses is not None and normal_forces is not None:
                # Get camera pose for adjusting object pose
                camera_pose = self.camera_nodes[i].matrix
                camera_pos = camera_pose[:3, 3].T
                camera_ori = R.from_matrix(camera_pose[:3, :3]).as_quat()

                self.adjust_with_force(
                    camera_pos, camera_ori, normal_forces, object_poses,
                )

            depth = self.r.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

            depths.append(depth)
        return depths

    def render_from_depth(
            self, depth
    ):
        rgb = self.phong_shading_renderer.generate(depth)
        return rgb, depth
