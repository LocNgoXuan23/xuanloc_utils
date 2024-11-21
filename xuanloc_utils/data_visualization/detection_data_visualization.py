import os
import re
import cv2
import shutil
from tqdm import tqdm
from ..box_annotator import BoxAnnotator
from ..common import create_folder, read_label_detect

def get_class_names(classes_path):
    with open(classes_path, 'r') as f:
        class_names = f.read().strip().split('\n')
    return class_names

def detection_data_visualization(input_path, output_path):
    box_annotator = BoxAnnotator()
    create_folder(output_path)
    
    input_imgs_path = os.path.join(input_path, 'images')
    input_labels_path = os.path.join(input_path, 'labels')
    classes_path = os.path.join(input_path, 'classes.txt')
    if os.path.exists(classes_path):
        class_names = get_class_names(classes_path)
    else:
        class_names = None
    
    img_names = os.listdir(input_imgs_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(input_imgs_path, img_name)
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        
        label_name = img_name.replace('.jpg', '.txt').replace('.PNG', '.txt').replace('.JPG', '.txt').replace('.png', '.txt')
        label_path = os.path.join(input_labels_path, label_name)
        label = read_label_detect(label_path)
        
        if len(label) == 0:
            continue
        
        for obj in label:
            c, x1, y1, x2, y2 = obj
            x1, y1, x2, y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
            img = box_annotator.annotate(
                img,
                box=[x1, y1, x2, y2],
                text=class_names[c] if class_names else str(c),
                c=c
            )
            
        output_img_path = os.path.join(output_path, img_name)
        cv2.imwrite(output_img_path, img)