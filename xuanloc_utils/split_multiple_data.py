import os
import shutil
from xuanloc_utils import common
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_data(input_paths, output_path, ratio):
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

    total_img_paths_train = []
    total_img_paths_val = []
    total_label_paths_train = []
    total_label_paths_val = []

    input_img_cnt = 0
    input_label_cnt = 0
    
    for input_path in input_paths:
        img_input_path = os.path.join(input_path, 'images')
        label_input_path = os.path.join(input_path, 'labels')

        input_img_cnt += len(common.get_items_from_folder(img_input_path, '.jpg')[0])
        input_label_cnt += len(common.get_items_from_folder(label_input_path, '.txt')[0])


        img_names = os.listdir(img_input_path)
        label_names = os.listdir(label_input_path)
        img_names.sort()
        label_names.sort()

        img_names_train, img_names_val, label_names_train, label_names_val = train_test_split(img_names, label_names, shuffle=True, test_size=ratio)
        
        img_paths_train = [os.path.join(img_input_path, img_name) for img_name in img_names_train]
        img_paths_val = [os.path.join(img_input_path, img_name) for img_name in img_names_val]
        label_paths_train = [os.path.join(label_input_path, label_name) for label_name in label_names_train]
        label_paths_val = [os.path.join(label_input_path, label_name) for label_name in label_names_val]


        total_img_paths_train += img_paths_train
        total_img_paths_val += img_paths_val
        total_label_paths_train += label_paths_train
        total_label_paths_val += label_paths_val

    print(f'Input image count: {input_img_cnt}, Input label count: {input_label_cnt}')
    print(f'Split image count: {len(total_img_paths_train) + len(total_img_paths_val)}, Split label count: {len(total_label_paths_train) + len(total_label_paths_val)}')

    total_img_paths_train.sort()
    total_img_paths_val.sort()
    total_label_paths_train.sort()
    total_label_paths_val.sort()

    print(len(total_img_paths_train), len(total_img_paths_val))
    print(len(total_label_paths_train), len(total_label_paths_val))

    for img_path_train, label_path_train in tqdm(zip(total_img_paths_train, total_label_paths_train), total=len(total_img_paths_train)):
        img_name = img_path_train.split('/')[-1]
        label_name = label_path_train.split('/')[-1]
        out_img_path = os.path.join(img_train_output_path, img_name)
        out_label_path = os.path.join(label_train_output_path, label_name)
        shutil.copy(img_path_train, out_img_path)
        shutil.copy(label_path_train, out_label_path)

    for img_path_val, label_path_val in tqdm(zip(total_img_paths_val, total_label_paths_val), total=len(total_img_paths_val)):
        img_name = img_path_val.split('/')[-1]
        label_name = label_path_val.split('/')[-1]
        out_img_path = os.path.join(img_val_output_path, img_name)
        out_label_path = os.path.join(label_val_output_path, label_name)
        shutil.copy(img_path_val, out_img_path)
        shutil.copy(label_path_val, out_label_path)
    
    split_img_cnt = len(common.get_items_from_folder(output_path, '.jpg')[0])
    split_label_cnt = len(common.get_items_from_folder(output_path, '.txt')[0])
    print(f'Split image count: {split_img_cnt}, Split label count: {split_label_cnt}')

    assert input_img_cnt == split_img_cnt 
    assert input_label_cnt == split_label_cnt

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
