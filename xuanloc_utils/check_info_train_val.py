import os
import shutil
from xuanloc_utils import common

def get_classes(labels_path):
    classes = []
    for label_name in os.listdir(labels_path):
        label_path = os.path.join(labels_path, label_name)
        label = common.read_label(label_path)
        for obj in label:
            c = obj[0]
            classes.append(c)
    classes = list(set(classes))
    classes.sort()
    return classes

def get_num_objs(labels_path, classes):
    num_objs = {}
    for c in classes:
        cnt = common.cal_num_items_in_labels(labels_path, c)
        num_objs[c] = cnt
    return num_objs


def check_info_train_val(input_path):
    train_input_path = os.path.join(input_path, 'train')
    train_imgs_input_path = os.path.join(train_input_path, 'images')
    train_labels_input_path = os.path.join(train_input_path, 'labels')
    val_input_path = os.path.join(input_path, 'val')
    val_imgs_input_path = os.path.join(val_input_path, 'images')
    val_labels_input_path = os.path.join(val_input_path, 'labels')
    
    # check num
    print('----------------------------------------------------')
    num_train_imgs = len(os.listdir(train_imgs_input_path))
    num_train_labels = len(os.listdir(train_labels_input_path))
    num_val_imgs = len(os.listdir(val_imgs_input_path))
    num_val_labels = len(os.listdir(val_labels_input_path))
    print(f'num_train_imgs = {num_train_imgs}, num_train_labels = {num_train_labels}')
    print(f'num_val_imgs = {num_val_imgs}, num_val_labels = {num_val_labels}')

    # check num classes
    print('----------------------------------------------------')
    classes_train = get_classes(train_labels_input_path)
    classes_val = get_classes(val_labels_input_path)
    print(f'classes_train = {classes_train}', f'classes_val = {classes_val}')
    print(f'num_classes_train = {len(classes_train)}, num_classes_val = {len(classes_val)}')

    # check num objects
    print('----------------------------------------------------')
    num_objs_train = get_num_objs(train_labels_input_path, classes_train)
    num_objs_val = get_num_objs(val_labels_input_path, classes_val)
    print(f'num_objs_train = {num_objs_train}')
    print(f'num_objs_val = {num_objs_val}')

    # check same items name
    train_img_names = os.listdir(train_imgs_input_path)
    train_label_names = os.listdir(train_labels_input_path)
    val_img_names = os.listdir(val_imgs_input_path)
    val_label_names = os.listdir(val_labels_input_path)
    train_img_names.sort()
    train_label_names.sort()
    val_img_names.sort()
    val_label_names.sort()

    for train_img_name, train_label_name in zip(train_img_names, train_label_names):
        assert train_img_name.replace('.jpg', '') == train_label_name.replace('.txt', ''), f'{train_img_name} != {train_label_name}'

    for val_img_name, val_label_name in zip(val_img_names, val_label_names):
        assert val_img_name.replace('.jpg', '') == val_label_name.replace('.txt', ''), f'{val_img_name} != {val_label_name}'
    print('----------------------------------------------------')

    # check overlap items name between train and val
    for train_img_name in train_img_names:
        assert train_img_name not in val_img_names, f'{train_img_name} in val_img_names'
    for train_label_name in train_label_names:
        assert train_label_name not in val_label_names, f'{train_label_name} in val_label_names'
    for val_img_name in val_img_names:
        assert val_img_name not in train_img_names, f'{val_img_name} in train_img_names'
    for val_label_name in val_label_names:
        assert val_label_name not in train_label_names, f'{val_label_name} in train_label_names'
