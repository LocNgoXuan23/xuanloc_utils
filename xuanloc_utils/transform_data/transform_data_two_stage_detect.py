import os
import cv2
import shutil
from tqdm import tqdm
from ..common import create_folder, read_label_detect, create_label_detect, cal_custom_iou_box

IOU_TH = 0.5

def transform_data_two_stage_detect(input_path, output_stage1_path, output_stage2_path, stage1_c, stage2_c_list_map):
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
        H, W, _ = img.shape
        
        label_name = img_name.replace('.jpg', '.txt').replace('.PNG', '.txt').replace('.JPG', '.txt').replace('.png', '.txt')
        label_path = os.path.join(input_labels_path, label_name)
        label = read_label_detect(label_path)
        
        stage1_objs = [obj for obj in label if obj[0] == stage1_c]
        stage2_objs = [obj for obj in label if obj[0] in stage2_c_list_map.keys()]
        
        # ----------------------------------------------
        # save stage1
        output_img_stage1_path = os.path.join(output_imgs_stage1_path, img_name)
        output_label_stage1_path = os.path.join(output_labels_stage1_path, label_name)
        shutil.copy(img_path, output_img_stage1_path)
        
        stage1_label = []
        for stage1_obj in stage1_objs:
            stage1_label.append(stage1_obj)
        create_label_detect(output_label_stage1_path, stage1_label, force_c=True)

        # ----------------------------------------------
        # save stage2
        for i, stage1_obj in enumerate(stage1_objs):
            inside_stage2_objs = []
            stage1_obj_box = stage1_obj[1:]
            
            for stage2_obj in stage2_objs:
                stage2_obj_box = stage2_obj[1:]
                try:
                    iou = cal_custom_iou_box(stage2_obj_box, stage1_obj_box)
                except Exception as e:
                    print(e)
                    print(img_name)
                    iou = 0
                if iou > IOU_TH:
                    inside_stage2_objs.append(stage2_obj)
                    
            # save img
            crop_img = img[int(stage1_obj_box[1] * H):int(stage1_obj_box[3] * H), int(stage1_obj_box[0] * W):int(stage1_obj_box[2] * W)]
            crop_img_name = img_name.replace('.jpg', '').replace('.PNG', '').replace('.JPG', '').replace('.png', '') + f'_{round(stage1_obj_box[0], 2)}_{round(stage1_obj_box[1], 2)}_{round(stage1_obj_box[2], 2)}_{round(stage1_obj_box[3], 2)}.jpg'
            cv2.imwrite(os.path.join(output_imgs_stage2_path, crop_img_name), crop_img)
            
            # scale label
            stage2_label = []
            for inside_stage2_obj in inside_stage2_objs:
                c, x1, y1, x2, y2 = inside_stage2_obj
                x1 = (x1 * W - stage1_obj_box[0] * W) / crop_img.shape[1]
                y1 = (y1 * H - stage1_obj_box[1] * H) / crop_img.shape[0]
                x2 = (x2 * W - stage1_obj_box[0] * W) / crop_img.shape[1]
                y2 = (y2 * H - stage1_obj_box[1] * H) / crop_img.shape[0]
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > 1:
                    x2 = 1
                c = stage2_c_list_map[c]
                new_obj = [c, x1, y1, x2, y2]
                stage2_label.append(new_obj)
                
            # save label
            output_label_stage2_path = os.path.join(output_labels_stage2_path, crop_img_name[:-4] + '.txt')
            create_label_detect(output_label_stage2_path, stage2_label)
            
if __name__ == '__main__':
    input_path = 'data/2stage_segment'
    output_stage1_path = 'data/2stage_detect/stage1'
    output_stage2_path = 'data/2stage_detect/stage2'
    stage1_c = 0
    # stage2_c_list = [1, 2, 3]
    stage2_c_list_map = {
        1: 1,
        2: 2,
        3: 3
    }
    transform_data_two_stage_detect(input_path, output_stage1_path, output_stage2_path, stage1_c, stage2_c_list)