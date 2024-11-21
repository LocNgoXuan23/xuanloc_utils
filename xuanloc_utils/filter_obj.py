import os
import shutil
from tqdm import tqdm
from xuanloc_utils import common

def filter_obj(input_path, output_path, c_list, force_c=False):
    imgs_output_path = os.path.join(output_path, 'images')
    labels_output_path = os.path.join(output_path, 'labels')
    common.create_folder(output_path)
    common.create_folder(imgs_output_path)
    common.create_folder(labels_output_path)

    # //////////////////////////////////////////////////////////////////
    imgs_input_path = os.path.join(input_path, 'images')
    labels_input_path = os.path.join(input_path, 'labels')
    img_names = os.listdir(imgs_input_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(imgs_input_path, img_name)
        label_path = os.path.join(labels_input_path, img_name.replace('.jpg', '.txt'))

        if os.path.exists(label_path):
            label = common.read_label_detect(label_path)
            label = [obj for obj in label if obj[0] in c_list]
            output_label_path = os.path.join(labels_output_path, img_name.replace('.jpg', '.txt'))
            common.create_label_detect(output_label_path, label, force_c=force_c)
        
        # copy image
        output_img_path = os.path.join(imgs_output_path, img_name)
        shutil.copy(img_path, output_img_path)
        