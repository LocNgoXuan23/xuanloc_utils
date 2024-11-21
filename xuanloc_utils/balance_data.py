import os
import shutil
import random
from tqdm import tqdm
from collections import Counter
from . import common

def balance_data(input_path, output_path, mode='over'):
    common.create_folder(output_path)
    
    cls_names = os.listdir(input_path)
    if mode == 'over':
        target_name = max(cls_names, key=lambda x: len(os.listdir(os.path.join(input_path, x))))
    elif mode == 'under':
        target_name = min(cls_names, key=lambda x: len(os.listdir(os.path.join(input_path, x))))
    target_cnt = len(os.listdir(os.path.join(input_path, target_name)))
    
    print(f'Max class: {target_name} with {target_cnt} images')
    for cls_name in cls_names:
        if cls_name == target_name:
            # copy all images in max class to output path
            cls_input_path = os.path.join(input_path, cls_name)
            cls_output_path = os.path.join(output_path, cls_name)
            common.create_folder(cls_output_path)
            
            img_names = os.listdir(cls_input_path)
            for img_name in tqdm(img_names, desc=f'Copying {cls_name} images'):
                shutil.copy(os.path.join(cls_input_path, img_name), os.path.join(cls_output_path, img_name))
        else:
            
            if mode == 'over':
                dup_times = target_cnt // len(os.listdir(os.path.join(input_path, cls_name)))
                remain_cnt = target_cnt - dup_times * len(os.listdir(os.path.join(input_path, cls_name)))
                
                cls_input_path = os.path.join(input_path, cls_name)
                cls_output_path = os.path.join(output_path, cls_name)
                common.create_folder(cls_output_path)

                img_names = os.listdir(cls_input_path)
                for i in range(dup_times):
                    for img_name in tqdm(img_names, desc=f'Duplicating times {i} for {cls_name} images'):
                        dupl_name = f'{img_name[:-4]}_dup{i}.jpg'
                        shutil.copy(os.path.join(cls_input_path, img_name), os.path.join(cls_output_path, dupl_name))

                for i in tqdm(range(remain_cnt), desc=f'Adding remaining {remain_cnt} images for {cls_name}'):
                    random_img_name = random.choice(img_names)
                    dupl_name = f'{random_img_name[:-4]}_duprand{i}.jpg'
                    shutil.copy(os.path.join(cls_input_path, random_img_name), os.path.join(cls_output_path, dupl_name))
            
            elif mode == 'under':
                cls_input_path = os.path.join(input_path, cls_name)
                cls_output_path = os.path.join(output_path, cls_name)
                common.create_folder(cls_output_path)
                
                img_names = os.listdir(cls_input_path)
                random.shuffle(img_names)
                for img_name in tqdm(img_names[:target_cnt], desc=f'Copying {cls_name} images'):
                    shutil.copy(os.path.join(cls_input_path, img_name), os.path.join(cls_output_path, img_name))
                    
    print(f'Balanced data saved to {output_path}')
    
    # verify balance
    verify_balance(output_path)
                
def verify_balance(output_path):
    class_counts = Counter()

    # Đếm số lượng ảnh trong mỗi lớp
    for class_name in os.listdir(output_path):
        class_path = os.path.join(output_path, class_name)
        if os.path.isdir(class_path):
            image_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = image_count

    # Kiểm tra xem tất cả các lớp có cùng số lượng ảnh không
    is_balanced = len(set(class_counts.values())) == 1

    print("Số lượng ảnh trong mỗi lớp:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    if is_balanced:
        print("\nDữ liệu đã được cân bằng chính xác.")
    else:
        print("\nDữ liệu chưa được cân bằng chính xác.")

    return is_balanced

if __name__ == '__main__':
    mode = 'under'  
    input_path = '/home/xuanlocserver/workingspace/ACB_project/new_approach_project/data/merged_data(demohub&ACB)_gap10_v2(for_smoking)(mouth_in_face)_preprocess_cls'
    output_path = f'/home/xuanlocserver/workingspace/ACB_project/new_approach_project/data/merged_data(demohub&ACB)_gap10_v2(for_smoking)(mouth_in_face)_preprocess_cls(balanced_{mode})'
    balance_data(input_path, output_path, mode)
    verify_balance(output_path)
    
    mode = 'over'  
    input_path = '/home/xuanlocserver/workingspace/ACB_project/new_approach_project/data/merged_data(demohub&ACB)_gap10_v2(for_smoking)(mouth_in_face)_preprocess_cls'
    output_path = f'/home/xuanlocserver/workingspace/ACB_project/new_approach_project/data/merged_data(demohub&ACB)_gap10_v2(for_smoking)(mouth_in_face)_preprocess_cls(balanced_{mode})'
    balance_data(input_path, output_path, mode)
    verify_balance(output_path)
