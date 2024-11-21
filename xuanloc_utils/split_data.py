import os
import shutil
from xuanloc_utils import common
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_data(input_path, output_path, ratio):
    train_output_path = os.path.join(output_path, 'train')
    val_output_path = os.path.join(output_path, 'val')
    img_train_output_path = os.path.join(train_output_path, 'images')
    label_train_output_path = os.path.join(train_output_path, 'labels')
    img_val_output_path = os.path.join(val_output_path, 'images')
    label_val_output_path = os.path.join(val_output_path, 'labels')
    common.create_folder(output_path)
    common.create_folder(train_output_path)
    common.create_folder(val_output_path)
    common.create_folder(img_train_output_path)
    common.create_folder(label_train_output_path)
    common.create_folder(img_val_output_path)
    common.create_folder(label_val_output_path)
    
    img_input_path = os.path.join(input_path, 'images')
    label_input_path = os.path.join(input_path, 'labels')

    img_names = os.listdir(img_input_path)
    train_img_names, val_img_names = train_test_split(img_names, test_size=ratio, random_state=42)

    for img_name in tqdm(train_img_names):
        img_path = os.path.join(img_input_path, img_name)
        label_name = img_name.replace('.jpg', '.txt').replace('.PNG', '.txt').replace('.JPG', '.txt').replace('.png', '.txt')
        label_path = os.path.join(label_input_path, label_name)

        output_img_path = os.path.join(img_train_output_path, img_name)
        output_label_path = os.path.join(label_train_output_path, label_name)

        shutil.copy(img_path, output_img_path)
        shutil.copy(label_path, output_label_path)

    for img_name in tqdm(val_img_names):
        img_path = os.path.join(img_input_path, img_name)
        label_name = img_name.replace('.jpg', '.txt').replace('.PNG', '.txt').replace('.JPG', '.txt').replace('.png', '.txt')
        label_path = os.path.join(label_input_path, label_name)

        output_img_path = os.path.join(img_val_output_path, img_name)
        output_label_path = os.path.join(label_val_output_path, label_name)

        shutil.copy(img_path, output_img_path)
        shutil.copy(label_path, output_label_path)


if __name__ == "__main__":
    input_path = '/media/xuanloc/sandisk500G/deep_learning/weapon_detection/data/multip_object_detection_data/weapon_data_v8(pedestrian)_preprocess'
    output_path = '/media/xuanloc/sandisk500G/deep_learning/weapon_detection/data/multip_object_detection_data/weapon_data_v8(pedestrian)_preprocess_split'
    ratio = 0.2

    print(input_path)
    split_data(input_path, output_path, ratio)