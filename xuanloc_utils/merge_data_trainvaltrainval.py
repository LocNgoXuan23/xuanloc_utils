import os
import shutil
from tqdm import tqdm
from xuanloc_utils import common

def merge_data_trainvaltrainval(input_paths, output_path):
    train_output_path = os.path.join(output_path, 'train')
    train_imgs_output_path = os.path.join(train_output_path, 'images')
    train_labels_output_path = os.path.join(train_output_path, 'labels')
    val_output_path = os.path.join(output_path, 'val')
    val_imgs_output_path = os.path.join(val_output_path, 'images')
    val_labels_output_path = os.path.join(val_output_path, 'labels')
    common.create_folder(output_path)
    common.create_folder(train_output_path)
    common.create_folder(train_imgs_output_path)
    common.create_folder(train_labels_output_path)
    common.create_folder(val_output_path)
    common.create_folder(val_imgs_output_path)
    common.create_folder(val_labels_output_path)

    # ///////////////////////////////////////
    for input_path in tqdm(input_paths):
        train_imgs_input_path = os.path.join(input_path, 'train', 'images')
        train_labels_input_path = os.path.join(input_path, 'train', 'labels')
        val_imgs_input_path = os.path.join(input_path, 'val', 'images')
        val_labels_input_path = os.path.join(input_path, 'val', 'labels')

        for img_name in os.listdir(train_imgs_input_path):  
            label_name = img_name.replace('.jpg', '.txt')
            img_path = os.path.join(train_imgs_input_path, img_name)
            label_path = os.path.join(train_labels_input_path, label_name)
            shutil.copy(img_path, train_imgs_output_path)
            shutil.copy(label_path, train_labels_output_path)
        
        for img_name in os.listdir(val_imgs_input_path):
            label_name = img_name.replace('.jpg', '.txt')
            img_path = os.path.join(val_imgs_input_path, img_name)
            label_path = os.path.join(val_labels_input_path, label_name)
            shutil.copy(img_path, val_imgs_output_path)
            shutil.copy(label_path, val_labels_output_path)