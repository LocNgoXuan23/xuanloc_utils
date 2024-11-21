import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import albumentations as A
from ..common import create_folder

def blur_aug(img):
    max_imgsz = max(img.shape)
    
    # create blur factor
    motion_blur_factor = int(max_imgsz / 4096 * (2 * np.random.randint(10, 20) + 1)) # 21 - 41
    motion_blur_factor = motion_blur_factor if motion_blur_factor % 2 else motion_blur_factor + 1
    
    gaussian_blur_factor = int(max_imgsz / 4096 * (2 * np.random.randint(10, 20) + 1)) # 21 - 41
    gaussian_blur_factor = gaussian_blur_factor if gaussian_blur_factor % 2 else gaussian_blur_factor + 1
    
    defocus_factor = int(max_imgsz / 4096 * (2 * np.random.randint(3, 5) + 1)) # 6 - 11
    defocus_factor = defocus_factor if defocus_factor % 2 else defocus_factor + 1
    
    # create aug blur list
    aug_blur_list = [
        A.MotionBlur(blur_limit=(7 * motion_blur_factor, 7 * motion_blur_factor), allow_shifted=False, p=1),
        A.GaussianBlur(blur_limit=(3 * gaussian_blur_factor, 7 * gaussian_blur_factor), p=1),
        A.Defocus(radius=(3 * defocus_factor, 7 * defocus_factor), alias_blur=(0.1 * defocus_factor, 0.5 * defocus_factor), p=1)
    ]
    
    # random aug blur
    random_aug_blur_idx = np.random.randint(0, len(aug_blur_list))
    
    # create transform
    transform = A.Compose([
        aug_blur_list[random_aug_blur_idx],
        A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), p=0.5),
    ])
    
    # apply transform
    transformed_img = transform(image=img)['image']
    
    return transformed_img

def aug_data_detect(input_path, output_path, label_ext='json'):
    input_imgs_path = os.path.join(input_path, 'images')
    input_labels_path = os.path.join(input_path, 'labels')
    
    output_imgs_path = os.path.join(output_path, 'images')
    output_labels_path = os.path.join(output_path, 'labels')
    create_folder(output_path)
    create_folder(output_imgs_path)
    create_folder(output_labels_path)
    
    img_names = os.listdir(input_imgs_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(input_imgs_path, img_name)
        
        label_name = img_name[:-4] + '.' + label_ext
        label_path = os.path.join(input_labels_path, label_name)
        
        # aug data
        img = cv2.imread(img_path)
        transformed_img = blur_aug(img) 
        
        # save raw to output
        output_img_path = os.path.join(output_imgs_path, img_name)
        shutil.copy(img_path, output_img_path)
        output_label_path = os.path.join(output_labels_path, label_name)
        shutil.copy(label_path, output_label_path)
        
        # save aug to output
        aug_name = img_name[:-4] + '_aug'
        aug_img_name = aug_name + '.jpg'
        aug_img_path = os.path.join(output_imgs_path, aug_img_name)
        cv2.imwrite(aug_img_path, transformed_img)
        
        aug_label_name = aug_name + '.' + label_ext
        aug_label_path = os.path.join(output_labels_path, aug_label_name)
        shutil.copy(label_path, aug_label_path)
        