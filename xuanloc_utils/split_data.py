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
        
def split_data_cls(input_path, output_path, ratio):
    train_output_path = os.path.join(output_path, 'train')
    val_output_path = os.path.join(output_path, 'val')
    common.create_folder(output_path)
    common.create_folder(train_output_path)
    common.create_folder(val_output_path)

    input_img_cnt = 0
    split_img_cnt = 0

    for cls_name in os.listdir(input_path):
        cls_input_path = os.path.join(input_path, cls_name)
        if not os.path.isdir(cls_input_path):
            continue

        cls_train_output_path = os.path.join(train_output_path, cls_name)
        cls_val_output_path = os.path.join(val_output_path, cls_name)
        common.create_folder(cls_train_output_path)
        common.create_folder(cls_val_output_path)

        img_names = os.listdir(cls_input_path)
        input_img_cnt += len(img_names)

        img_names_train, img_names_val = train_test_split(img_names, shuffle=True, test_size=ratio)
        
        for img_name in tqdm(img_names_train, desc=f'Copying train images for {cls_name}'):
            src_path = os.path.join(cls_input_path, img_name)
            dst_path = os.path.join(cls_train_output_path, img_name)
            shutil.copy(src_path, dst_path)
            split_img_cnt += 1

        for img_name in tqdm(img_names_val, desc=f'Copying val images for {cls_name}'):
            src_path = os.path.join(cls_input_path, img_name)
            dst_path = os.path.join(cls_val_output_path, img_name)
            shutil.copy(src_path, dst_path)
            split_img_cnt += 1

    print(f'Input image count: {input_img_cnt}')
    print(f'Split image count: {split_img_cnt}')

    assert input_img_cnt == split_img_cnt, "Số lượng ảnh đầu vào và đầu ra không khớp"



if __name__ == "__main__":
    input_path = '/media/xuanloc/sandisk500G/deep_learning/weapon_detection/data/multip_object_detection_data/weapon_data_v8(pedestrian)_preprocess'
    output_path = '/media/xuanloc/sandisk500G/deep_learning/weapon_detection/data/multip_object_detection_data/weapon_data_v8(pedestrian)_preprocess_split'
    ratio = 0.2

    print(input_path)
    split_data(input_path, output_path, ratio)