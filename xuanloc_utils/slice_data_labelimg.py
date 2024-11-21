import os
import shutil
import re
from tqdm import tqdm
from xuanloc_utils import common

def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)


def slice_data_labelimg(input_path, output_path, s):
    imgs_output_path = os.path.join(output_path, 'images')
    labels_output_path = os.path.join(output_path, 'labels')
    common.create_folder(output_path)
    common.create_folder(imgs_output_path)
    common.create_folder(labels_output_path)

    # ////////////////////////////////////////////////
    imgs_input_path = os.path.join(input_path, 'images')
    labels_input_path = os.path.join(input_path, 'labels')

    img_names = os.listdir(imgs_input_path)
    natural_sort(img_names)
    for img_name in tqdm(img_names[:s]):
    # for img_name in tqdm(img_names[s:]):
        img_path = os.path.join(imgs_input_path, img_name)
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(labels_input_path, label_name)

        output_img_path = os.path.join(imgs_output_path, img_name)
        output_label_path = os.path.join(labels_output_path, label_name)

        shutil.copy(img_path, output_img_path)

        # check label_path exist
        if os.path.exists(label_path):
            shutil.copy(label_path, output_label_path)
        
# if __name__ == '__main__':
#     input_path = '/media/xuanloc/sandisk500G/deep_learning/weapon_detection/data/multip_object_detection_data/crop_person_v1_preprocess_remove_pad'
#     output_path = '/media/xuanloc/sandisk500G/deep_learning/weapon_detection/data/multip_object_detection_data/head_data_v4'
#     s = 6500
#     slice_data(input_path, output_path, s)