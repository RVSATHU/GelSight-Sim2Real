import os, math, glob, cv2
import shutil, random

"""
The generated simulation images are placed in folders, e.g., "moon/", "cylinder/", "prism/", etc.
The collected real images are also suggested to be placed in this way.
This script will split the images into training, validation and test sets, also placed separately in folders.
"""


def tryMakeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


trainingSetPercentage = 0.7
validationSetPercentage = 0.1
testSetPercentage = 0.2
labels = ['cone', 'cylinder', 'cylinder_shell', 'cylinder_side', 'dot_in', 'dots', 'hexagon', 'large_sphere',
          'line', 'moon', 'pacman', 'parallel_lines', 'prism', 'random', 'small_sphere', 'torus', 'triangle', 'wave']

folder_with_18classes = "../simulation_tacto/18class/rgb"
folder_to_place_train_val_and_test = "SIM"


def copy_class(target_folder, class_name):
    origin_file_path = os.path.join(folder_with_18classes, class_name)
    files = glob.glob(os.path.join(origin_file_path, "*.png"))
    file_num_total = len(files)
    print(file_num_total)
    # randomIndex = [i for i in range(0, file_num_total)]
    # random.shuffle(randomIndex)
    tryMakeDir(os.path.join(target_folder + '/train', class_name))
    tryMakeDir(os.path.join(target_folder + '/val', class_name))
    tryMakeDir(os.path.join(target_folder + '/test', class_name))

    for i in range(0, file_num_total, 1):
        src = files[i]
        basename = os.path.basename(src)
        id = int(basename.split('.')[0])

        if id <= file_num_total * trainingSetPercentage:
            dest_file_path = os.path.join(target_folder + "/train", class_name)
        elif id <= file_num_total * (trainingSetPercentage + validationSetPercentage):
            dest_file_path = os.path.join(target_folder + "/val", class_name)
        else:
            dest_file_path = os.path.join(target_folder + "/test", class_name)

        dest_file_name = basename
        dst = os.path.join(dest_file_path, dest_file_name)
        shutil.copy(src, dst)


for label in labels:
    copy_class(folder_to_place_train_val_and_test, label)
