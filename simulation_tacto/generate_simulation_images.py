import logging
import math
import os

import cv2
import hydra
import pybullet as p
import tacto  # Import TACTO

import pybulletX as px
import time

import numpy as np

max_offset_planar = 0.0055

max_offset_roll_default = 0.1  # about 6 degrees. for some objects, we use half of the angle.
max_offset_pitch_default = 0.1
max_offset_yaw = 3.1
num_force = 3  # for each pose of each object, run simulation with 3 levels of forces
min_force = 15
max_force = 20

folder = "18class"  # folder to place the generated images
num_of_each_class = 12  # number of images to generate for each class

log = logging.getLogger(__name__)
"""
all 21 classes
'moon', 'cylinder', 'random', 'parallel_lines', 'line', 'triangle', 'dots', 'cylinder_shell', 
'large_sphere', 'flat_slab', 'cylinder_side', 'hexagon', 'wave', 'torus', 'prism', 'small_sphere', 'cone', 'pacman', 
'curved_face', 'dot_in', 'crossed_lines'
but 'curved_face', 'flat_slab' and 'crossed_lines' are excluded (explained in the paper).
"""
object_has_small_roll_and_pitch = {'moon': False, 'cylinder': False, 'random': False, 'parallel_lines': True,
                                   'line': True, 'triangle': False, 'dots': True, 'cylinder_shell': False,
                                   'large_sphere': True, 'cylinder_side': True, 'hexagon': False, 'wave': True,
                                   'torus': True, 'prism': False, 'small_sphere': True, 'cone': True, 'pacman': False,
                                   'dot_in': True}


def try_make_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)
        return True
    else:
        return False


@hydra.main(config_path="conf", config_name="generate_simulation_images")
def main(cfg):
    # Initialize the sensor
    global max_offset_roll, max_offset_pitch
    thusight = tacto.Sensor(**cfg.tacto)
    cfg.object.urdf_path = 'objects/' + object_name + '.urdf'
    print(cfg)

    # Initialize World
    log.info("Initializing world")
    client = px.init(mode=p.GUI)
    p.setGravity(0, 0, 0)  # disable gravity to get true constraint force

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize THUsight
    thusight_body = px.Body(**cfg.thusight)
    thusight.add_camera(thusight_body.id, [-1])

    # Add object to pybullet and tacto simulator
    obj = px.Body(**cfg.object)
    thusight.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel, max_force=10)
    # we use panel, but use other constraints instead
    # panel.start()

    position_center_top = (0, 0, 0.07)  # z is 0.07, always not in contact. Move until in contact. Then change the
    # max force to generate different depth

    if object_has_small_roll_and_pitch[object_name]:
        max_offset_roll = max_offset_roll_default / 2
        max_offset_pitch = max_offset_pitch_default / 2
    else:
        max_offset_roll = max_offset_roll_default
        max_offset_pitch = max_offset_pitch_default

    constraint_cid = panel.cid  # override by code

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    index = 0
    for random_num in range(math.ceil(num_of_each_class / num_force)):
        pos = np.random.rand(3, ) * 2 * max_offset_planar - max_offset_planar
        pos[2] = 0
        pos += position_center_top

        roll = np.random.rand(1) * 2 * max_offset_roll - max_offset_roll
        pitch = np.random.rand(1) * 2 * max_offset_pitch - max_offset_pitch
        yaw = np.random.rand(1) * 2 * max_offset_yaw - max_offset_yaw
        ori = p.getQuaternionFromEuler([roll, pitch, yaw])

        p.changeConstraint(constraint_cid, pos, ori, maxForce=1)
        time.sleep(1)
        color, depth, contact = thusight.render()
        thusight.updateGUI(color, depth)

        while not contact:  # move down until contact
            pos = (pos[0], pos[1], pos[2] - 0.0001)
            p.changeConstraint(constraint_cid, pos, ori, maxForce=0.1)
            time.sleep(0.01)
            color, depth, contact = thusight.render()
            thusight.updateGUI(color, depth)

        print("{:.5f}".format(pos[2]))
        pos = (pos[0], pos[1], pos[2] - 0.005)

        for force in np.linspace(start=min_force, stop=max_force, num=num_force, endpoint=True):
            p.changeConstraint(constraint_cid, pos, ori, maxForce=force)
            time.sleep(0.01)
            color, depth, contact = thusight.render()
            thusight.updateGUI(color, depth)
            if contact:
                index += 1
                cv2.imwrite(os.path.join(folder + "/rgb/" + object_name, str(index) + '.png'),
                            color[0])
                # np.save(
                #     os.path.join(folder + "/depth/" + object_name, str(index) + '.npy'),
                #     depth[0])
                # uncomment the above if you would like to save the depth map
            else:
                print(pos)
    p.disconnect(client)


if __name__ == "__main__":
    for object_name in object_has_small_roll_and_pitch.keys():
        directory = folder + "/rgb/" + object_name
        if not try_make_dir(directory):
            continue  # avoid re-generation
        directory = folder + "/depth/" + object_name
        try_make_dir(directory)
        main()
