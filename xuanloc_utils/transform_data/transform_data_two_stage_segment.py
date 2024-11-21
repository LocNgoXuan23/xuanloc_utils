import os
import shutil
import cv2
import json
from tqdm import tqdm
from ..common import create_folder, cal_custom_iou_poly, box2poly, poly2box

IOU_TH = 0.5

def read_labelme_json(label_path):
    with open(label_path, 'r') as f:
        data = json.load(f)
    label = data['shapes']
    return label

def get_obj_poly(obj):
    if obj['shape_type'] == 'polygon':
        return obj['points']
    elif obj['shape_type'] == 'rectangle':
        return box2poly(obj['points'][0][0], obj['points'][0][1], obj['points'][1][0], obj['points'][1][1])
    else:
        raise ValueError('shape_type not supported')

def get_obj_box(obj):
    if obj['shape_type'] == 'polygon':
        poly = obj['points']
        return poly2box(poly)
    elif obj['shape_type'] == 'rectangle':
        x1, y1 = obj['points'][0]
        x2, y2 = obj['points'][1]
        
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        return [x_min, y_min, x_max, y_max]
    else:
        raise ValueError('shape_type not supported')

def create_label_detect(label_path, label, force_c=None):
    f = open(label_path, 'w')
    lines = []
    for obj in label:
        x1, y1, x2, y2 = obj[1], obj[2], obj[3], obj[4]    
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        if x < 0 or y < 0 or w < 0 or h < 0:
            print(label)
            print(x, y, w, h)
            print(label_path)
            raise Exception('Invalid label')
        if force_c is not None:
            line = [0, x, y, w, h]
        else:
            line = [obj[0], x, y, w, h]
        line = [str(x) for x in line]
        line = ' '.join(line)
        lines.append(line)
    lines = '\n'.join(lines)
    f.write(lines)
    f.close()

def create_label_segment(label_path, label):
    f = open(label_path, 'w')
    lines = []
    for obj in label:
        line = str(obj[0])
        for p in obj[1:]:
            line += ' ' + str(p[0]) + ' ' + str(p[1])
        lines.append(line)
    lines = '\n'.join(lines)
    f.write(lines)
    f.close()

def transform_data_two_stage_segment(input_path, output_stage1_path, output_stage2_path, stage1_c, stage2_c_list_map):
    output_imgs_stage1_path = os.path.join(output_stage1_path, 'images')
    output_labels_stage1_path = os.path.join(output_stage1_path, 'labels')
    output_imgs_stage2_path = os.path.join(output_stage2_path, 'images')
    output_labels_stage2_path = os.path.join(output_stage2_path, 'labels')
    create_folder(output_stage1_path)
    create_folder(output_imgs_stage1_path)
    create_folder(output_labels_stage1_path)
    create_folder(output_stage2_path)
    create_folder(output_imgs_stage2_path)
    create_folder(output_labels_stage2_path)

    input_imgs_path = os.path.join(input_path, 'images')
    input_labels_path = os.path.join(input_path, 'labels')

    img_names = os.listdir(input_imgs_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(input_imgs_path, img_name)
        img = cv2.imread(img_path)

        label_path = os.path.join(input_labels_path, img_name.replace('.jpg', '.json').replace('.PNG', '.json').replace('.JPG', '.json').replace('.png', '.json'))
        label = read_labelme_json(label_path)

        stage1_objs = [obj for obj in label if obj['label'] == stage1_c]
        stage2_objs = [obj for obj in label if obj['label'] in stage2_c_list_map.keys()]

        # ----------------------------------------------
        # save stage1
        output_img_stage1_path = os.path.join(output_imgs_stage1_path, img_name)
        output_label_stage1_path = os.path.join(output_labels_stage1_path, img_name.replace('.jpg', '.txt').replace('.PNG', '.txt').replace('.JPG', '.txt').replace('.png', '.txt'))
        shutil.copy(img_path, output_img_stage1_path)

        yolo_label = []
        for stage1_obj in stage1_objs:
            x1, y1, x2, y2 = get_obj_box(stage1_obj)
            x1, y1, x2, y2 = x1 / img.shape[1], y1 / img.shape[0], x2 / img.shape[1], y2 / img.shape[0]
            obj = [0, x1, y1, x2, y2]
            yolo_label.append(obj)
        create_label_detect(output_label_stage1_path, yolo_label, force_c=True)
        
        # ----------------------------------------------
        # save stage2
        for i, stage1_obj in enumerate(stage1_objs):
            inside_stage2_objs = []
            stage1_obj_poly = get_obj_poly(stage1_obj)

            for stage2_obj in stage2_objs:
                stage2_obj_poly = get_obj_poly(stage2_obj)
                if cal_custom_iou_poly(stage2_obj_poly, stage1_obj_poly) > IOU_TH:
                    inside_stage2_objs.append(stage2_obj)
            
            # save image
            stage1_obj_box = poly2box(stage1_obj_poly)
            crop_img = img[int(stage1_obj_box[1]):int(stage1_obj_box[3]), int(stage1_obj_box[0]):int(stage1_obj_box[2])]
            crop_img_name = img_name.replace('.jpg', '.txt').replace('.PNG', '.txt').replace('.JPG', '.txt').replace('.png', '.txt') + f'_{round(stage1_obj_box[0], 2)}_{round(stage1_obj_box[1], 2)}_{round(stage1_obj_box[2], 2)}_{round(stage1_obj_box[3], 2)}.jpg'
            cv2.imwrite(os.path.join(output_imgs_stage2_path, crop_img_name), crop_img)

            # scale label
            yolo_label = []
            for inside_stage2_obj in inside_stage2_objs:
                stage2_obj_poly = get_obj_poly(inside_stage2_obj)
                new_stage2_obj_poly = []
                for p in stage2_obj_poly:
                    p = [p[0] - stage1_obj_box[0], p[1] - stage1_obj_box[1]]
                    p = [p[0] / crop_img.shape[1], p[1] / crop_img.shape[0]]
                    if p[0] > 1:
                        p[0] = 1
                    if p[1] > 1:
                        p[1] = 1
                    if p[0] < 0:
                        p[0] = 0
                    if p[1] < 0:
                        p[1] = 0
                    new_stage2_obj_poly.append(p)
                stage2_obj_poly = new_stage2_obj_poly
                obj = [stage2_c_list_map[inside_stage2_obj['label']]] + stage2_obj_poly
                yolo_label.append(obj)
            crop_label_name = crop_img_name[:-4] + '.txt'
            create_label_segment(os.path.join(output_labels_stage2_path, crop_label_name), yolo_label)
            
if __name__ == '__main__':
    input_path = '/media/xuanloc/sandisk500G/deep_learning/denso_project/training_pipeline/data/anh_clock'
    output_stage1_path = '/media/xuanloc/sandisk500G/deep_learning/denso_project/training_pipeline/data/anh_clock_stage1'
    output_stage2_path = '/media/xuanloc/sandisk500G/deep_learning/denso_project/training_pipeline/data/anh_clock_stage2'

    stage1_c = 'clock'
    stage2_c_list_map = {'kim': 0, 'zone': 1}

    transform_data_two_stage_segment(input_path, output_stage1_path, output_stage2_path, stage1_c, stage2_c_list_map)