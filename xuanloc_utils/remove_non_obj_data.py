import os
import shutil
from .common import *
from tqdm import tqdm

def read_label(label_path):
    f = open(label_path, 'r')
    lines = f.readlines()
    f.close()

    label = []
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        line = [int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        label.append(line)
    
    return label

def remove_non_obj_data(input_path, output_path):
    img_output_path = os.path.join(output_path, 'images')
    label_output_path = os.path.join(output_path, 'labels')
    create_folder(output_path)
    create_folder(img_output_path)
    create_folder(label_output_path)

    img_input_path = os.path.join(input_path, 'images')
    label_input_path = os.path.join(input_path, 'labels')

    img_names = os.listdir(img_input_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(img_input_path, img_name)
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(label_input_path, label_name)

        # check label_path exists
        if not os.path.exists(label_path):
            continue

        label = read_label(label_path)
        if len(label) > 0:
            output_img_path = os.path.join(img_output_path, img_name)
            output_label_path = os.path.join(label_output_path, label_name)

            shutil.copy(img_path, output_img_path)
            shutil.copy(label_path, output_label_path)


