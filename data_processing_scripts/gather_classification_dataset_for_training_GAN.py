import os, math, glob, cv2
import shutil, random

"""
Select the training, validation or test sets to form a CycleGAN dataset
"""


def tryMakeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


labels = ['cone', 'cylinder', 'cylinder_shell', 'cylinder_side', 'dot_in', 'dots', 'hexagon', 'large_sphere',
          'line', 'moon', 'pacman', 'parallel_lines', 'prism', 'random', 'small_sphere', 'torus', 'triangle', 'wave']

split_data_folder = "SIM"
sets_to_gather = ['train', 'val', 'test']  # select the subsets to gather
merged_gan_set_path = "forGANtraining/All_B"

tryMakeDir(merged_gan_set_path)

for classification_set in sets_to_gather:
    set_path = os.path.join(split_data_folder, classification_set)
    for label in labels:
        class_path = os.path.join(set_path, label)
        files = glob.glob(os.path.join(class_path, "*.png"))
        for file in files:
            basename = os.path.basename(file)
            dst = os.path.join(merged_gan_set_path, label + "_" + basename)  # add prefix according to object name
            shutil.copy(file, dst)
