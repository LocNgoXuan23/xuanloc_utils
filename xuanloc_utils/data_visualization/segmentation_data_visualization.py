import os
import cv2
from tqdm import tqdm
from ..box_annotator import BoxAnnotator
from ..common import create_folder, read_label_segment_labelme, read_label_segment_yolo

def segmentation_data_visualization_labelme(input_path, output_path):
    box_annotator = BoxAnnotator()
    create_folder(output_path)
    
    input_imgs_path = os.path.join(input_path, 'images')
    input_labels_path = os.path.join(input_path, 'labels')
    
    img_names = os.listdir(input_imgs_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(input_imgs_path, img_name)
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        
        label_name = img_name.replace('.jpg', '.json').replace('.PNG', '.json').replace('.JPG', '.json').replace('.png', '.json')
        label_path = os.path.join(input_labels_path, label_name)
        label = read_label_segment_labelme(label_path)
        for obj in label:
            c, poly, _ = obj
            for i in range(len(poly)):
                poly[i] = [int(poly[i][0]*W), int(poly[i][1]*H)]
            
            img = box_annotator.annotate(
                img=img,
                mask=poly,
                text=c,
                c=hash(c) % 256
            )
            
        output_img_path = os.path.join(output_path, img_name)
        cv2.imwrite(output_img_path, img)
        
def segmentation_data_visualization_yolo(input_path, output_path):
    box_annotator = BoxAnnotator()
    create_folder(output_path)
    
    input_imgs_path = os.path.join(input_path, 'images')
    input_labels_path = os.path.join(input_path, 'labels')
    
    img_names = os.listdir(input_imgs_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(input_imgs_path, img_name)
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        
        label_name = img_name.replace('.jpg', '.txt').replace('.PNG', '.txt').replace('.JPG', '.txt').replace('.png', '.txt')
        label_path = os.path.join(input_labels_path, label_name)
        # label = read_label_segment_labelme(label_path)
        label = read_label_segment_yolo(label_path)
        
        for obj in label:
            c, poly = obj
            for i in range(len(poly)):
                poly[i] = [int(poly[i][0]*W), int(poly[i][1]*H)]
            
            img = box_annotator.annotate(
                img=img,
                mask=poly,
                c=c
            )
            
        output_img_path = os.path.join(output_path, img_name)
        cv2.imwrite(output_img_path, img)