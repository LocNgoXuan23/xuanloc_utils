import os
import shutil

import cv2
import base64
from shapely.geometry import Polygon

import time
from datetime import datetime

import yaml
import json

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# //////////////////////////////////////////

def load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def read_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def write_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f)

def get_time():
    return str(time.time()).replace('.', '_')

def get_current_date():
    return datetime.now().strftime('%d-%m-%y')

def create_folder(path, force=True):
    if force:
        try:
            shutil.rmtree(path)
        except:
            pass
        os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(path)

def remove_item(path):
    try:
        shutil.rmtree(path)
    except:
        pass

def read_label_detect(label_path, c=None):
    f = open(label_path, 'r')
    lines = f.readlines()
    f.close()

    label = []
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        x, y, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        if c is not None:
            line = [c, x1, y1, x2, y2]
        else:
            line = [int(line[0]), x1, y1, x2, y2]
    
        label.append(line)
    
    return label

def read_label_segment_yolo(label_path):
    f = open(label_path, 'r')
    lines = f.readlines()
    f.close()

    label = []
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        c = int(line[0])
        poly = []
        for i in range(1, len(line), 2):
            x, y = float(line[i]), float(line[i + 1])
            poly.append([x, y])
        label.append([c, poly])
    
    return label

def read_label_segment_labelme(label_path):
    def read_labelme_json(label_path):
        with open(label_path, 'r') as f:
            data = json.load(f)
        raw_label = data['shapes']
        img_size = data['imageHeight'], data['imageWidth']
        return raw_label, img_size
    
    def get_obj_poly(obj):
        if obj['shape_type'] == 'polygon':
            return obj['points']
        elif obj['shape_type'] == 'rectangle':
            return box2poly(obj['points'][0][0], obj['points'][0][1], obj['points'][1][0], obj['points'][1][1])
        else:
            raise ValueError('shape_type not supported')

    raw_label, img_size = read_labelme_json(label_path)
    
    label = []
    for raw_obj in raw_label:
        poly = get_obj_poly(raw_obj)
        new_poly = []
        for i in range(len(poly)):
            # poly[i][0] /= img_size[1]
            # poly[i][1] /= img_size[0]
            new_poly.append([poly[i][0] / img_size[1], poly[i][1] / img_size[0]])
        c = raw_obj['label']
        shape_type = raw_obj['shape_type']
        obj = [c, new_poly, shape_type]
        label.append(obj)
    
    return label        

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

def get_items_from_folder(folder_path, exts):
    # walk through all files in the folder
    item_names, item_paths = [], []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in exts):
                item_names.append(file)
                item_paths.append(os.path.join(root, file))
    return item_names, item_paths

def crop_frames_from_video(args):
    video_path, output_folder, num_gap_frames = args
    video_name = video_path.split('/')[-1][:-4]
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(num_frames)):
        ret, frame = video.read()
        if i % num_gap_frames == 0:
            if ret:  # Check if frame reading was successful
                frame_path = os.path.join(output_folder, f'{video_name}_{i}.jpg')
                cv2.imwrite(frame_path, frame)
    video.release()

def crop_frames_from_videos(videos_path, output_folder, num_gap_frames):
    create_folder(output_folder)
    
    video_names, video_paths = get_items_from_folder(videos_path, exts=['.mp4'])
    args = [(video_path, output_folder, num_gap_frames) for video_path in video_paths]

    # Determine the number of processes to use
    num_processes = os.cpu_count()  # You can adjust this as needed

    with Pool(num_processes) as pool:
        # Wrap tqdm around pool.imap to show progress
        list(tqdm(pool.imap(crop_frames_from_video, args), total=len(args)))

def cal_num_items_in_labels(labels_path, target_c):
    cnt = 0
    label_names = os.listdir(labels_path)
    print(f'Number of labels: {len(label_names)}')
    for label_name in label_names:
        label_path = os.path.join(labels_path, label_name)
        label = read_label(label_path)
        for obj in label:
            if obj[0] == target_c:
                cnt += 1
        
    return cnt

def resize_scale_img(img, scale):
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    return img


def cal_custom_iou_poly(poly1, poly2):
    """
    Calculate the custom IoU between two boxes.
    """

    try:
        poly1 = Polygon(poly1)
        poly2 = Polygon(poly2)

        if poly1.intersects(poly2): 
            intersection_area = poly1.intersection(poly2).area
            poly1_area = poly1.area
            union_area = poly1.area + poly2.area - intersection_area

            iou = intersection_area / poly1_area
            return iou
        return 0
    except:
        # print(poly1, poly1_area)
        return 0
    
def box_to_poly(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def read_json(path):
    # with open(path, 'r') as f:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    # with open(path, 'w') as f:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def concat_videos(video_paths, grid, output_path):
    from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip
    
    # Tải tất cả video
    clips = []
    for path in video_paths:
        clip = VideoFileClip(path)
        # Lấy tên file làm tiêu đề
        title = os.path.splitext(os.path.basename(path))[0]
        # Tạo TextClip cho tiêu đề
        title_clip = TextClip(title, fontsize=30, color='white', bg_color='black', font='Arial')
        title_clip = title_clip.set_position(('center', 'top')).set_duration(clip.duration)
        # Kết hợp video và tiêu đề
        final_clip = CompositeVideoClip([clip, title_clip])
        clips.append(final_clip)

    # Đảm bảo chúng ta có đúng số lượng clip cho lưới
    expected_num_clips = grid[0] * grid[1]
    if len(clips) != expected_num_clips:
        raise ValueError(f"Số lượng đường dẫn video ({len(clips)}) không khớp với bố cục lưới {grid}.")

    # Kiểm tra FPS của video đầu tiên (giả sử tất cả có cùng FPS)
    fps = clips[0].fps

    # Chuẩn bị mảng lưới
    clips_grid = np.array(clips).reshape(grid)
    
    # Ghép các video thành lưới
    final_clip = clips_array(list(clips_grid))
    
    # Đường dẫn file đầu ra với FPS từ video đầu vào
    final_clip.write_videofile(output_path, codec="libx264", fps=fps)  # Sử dụng FPS từ video đầu vào
    
    return output_path

def concat_videos_opencv(video_paths, grid, output_path):
    """
    Merge các video theo lưới grid (grid[0]: số hàng, grid[1]: số cột)
    Các video đầu vào có cùng kích thước và số frame.
    Mỗi video sẽ được vẽ tiêu đề (tên file) ở trên cùng.
    """
    # Kiểm tra số lượng video có khớp với bố cục grid không
    expected_num_clips = grid[0] * grid[1]
    if len(video_paths) != expected_num_clips:
        raise ValueError(f"Số lượng đường dẫn video ({len(video_paths)}) không khớp với bố cục lưới {grid}.")

    # Mở tất cả các video bằng cv2.VideoCapture
    captures = [cv2.VideoCapture(path) for path in video_paths]

    # Lấy các thuộc tính video từ video đầu tiên (giả sử tất cả giống nhau)
    fps = captures[0].get(cv2.CAP_PROP_FPS)
    frame_count = int(captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Kích thước frame đầu ra (ghép theo grid)
    out_width = grid[1] * frame_width
    out_height = grid[0] * frame_height

    # Khởi tạo VideoWriter với codec và FPS như video gốc
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Có thể dùng 'XVID' hoặc codec khác nếu cần
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    # Lấy tiêu đề từ tên file của mỗi video
    titles = [os.path.splitext(os.path.basename(path))[0] for path in video_paths]

    # Thiết lập các tham số cho việc vẽ chữ
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # Màu trắng
    text_bg_color = (0, 0, 0)     # Màu nền đen cho chữ

    # Lặp qua từng frame của video (giả sử tất cả video có cùng số frame)
    for frame_idx in tqdm(range(frame_count)):
        frames = []
        # Lấy một frame từ mỗi video
        for cap in captures:
            ret, frame = cap.read()
            # Nếu không đọc được frame (trường hợp gặp lỗi) thì thay thế bằng frame đen
            if not ret:
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Vẽ tiêu đề lên mỗi frame (ở vị trí trên cùng, canh giữa)
        for i, frame in enumerate(frames):
            title = titles[i]
            # Tính toán kích thước text để căn giữa
            (text_w, text_h), baseline = cv2.getTextSize(title, font, font_scale, font_thickness)
            text_x = (frame_width - text_w) // 2
            text_y = text_h + 10  # 10 pixel lề trên
            # Vẽ hình chữ nhật nền cho chữ
            cv2.rectangle(frame, (text_x, text_y - text_h - baseline),
                          (text_x + text_w, text_y + baseline), text_bg_color, cv2.FILLED)
            # Vẽ chữ lên frame
            cv2.putText(frame, title, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # Ghép các frame của video theo lưới
        # Chia danh sách frames thành các hàng (mỗi hàng có grid[1] frame)
        rows_frames = []
        for i in range(0, expected_num_clips, grid[1]):
            row = np.hstack(frames[i:i+grid[1]])
            rows_frames.append(row)
        # Ghép các hàng theo chiều dọc
        grid_frame = np.vstack(rows_frames)
        
        # Ghi frame ghép vào video đầu ra
        out_writer.write(grid_frame)
    
    # Giải phóng các đối tượng VideoCapture và VideoWriter
    for cap in captures:
        cap.release()
    out_writer.release()
    
    return output_path

def cal_custom_iou_box(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)    

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    iou = inter_area / box1_area

    return iou

def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def calc_dis(a, b):
    # point vs point
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    # point vs line
    elif isinstance(a, (tuple, list)) and isinstance(b, (float, int)):
        return abs(a[1] - b)
    
    else:
        raise ValueError('Invalid input type')
    

def move_files(input_path, output_path, ext):
    create_folder(output_path)
    
    file_names = os.listdir(input_path)
    for file_name in file_names:
        if file_name.endswith(ext):
            shutil.move(os.path.join(input_path, file_name), os.path.join(output_path, file_name))

def draw_poly(img, poly, c):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0)]

    pts = []
    for p in poly:
        pts.append((int(p[0]), int(p[1])))
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], isClosed=True, color=colors[c], thickness=5)
    return img

def expanding_box(
    box, 
    expand_right_ratio,
    expand_left_ratio,
    expand_top_ratio,
    expand_bottom_ratio,
    xy_max=None
):
    # range of box is 0 -> 1

    box_width = box[2] - box[0]
    box_height = box[3] - box[1]

    expand_right = box_width * expand_right_ratio
    expand_left = box_width * expand_left_ratio
    expand_top = box_height * expand_top_ratio
    expand_bottom = box_height * expand_bottom_ratio

    new_box = [
        box[0] - expand_left,
        box[1] - expand_top,
        box[2] + expand_right,
        box[3] + expand_bottom
    ]

    if not xy_max:
        # check if new box is out of range
        if new_box[0] < 0:
            new_box[0] = 0
        if new_box[1] < 0:
            new_box[1] = 0
        if new_box[2] > 1:
            new_box[2] = 1
        if new_box[3] > 1:
            new_box[3] = 1
    else:
        # check if new box is out of range
        if new_box[0] < 0:
            new_box[0] = 0
        if new_box[1] < 0:
            new_box[1] = 0
        if new_box[2] > xy_max[0]:
            new_box[2] = xy_max[0]
        if new_box[3] > xy_max[1]:
            new_box[3] = xy_max[1]
    
    return new_box

def resize_square_img_and_label(img, label, size):
    H, W, _ = img.shape
    if H > W:
        pad = (H - W) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        for i in range(len(label)):
            x1, y1, x2, y2 = label[i][1], label[i][2], label[i][3], label[i][4]
            x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H
            new_x1 = (x1 + pad) / H
            new_y1 = y1 / H
            new_x2 = (x2 + pad) / H
            new_y2 = y2 / H

            # check if new box is out of range
            if new_x1 < 0:
                new_x1 = 0
            if new_y1 < 0:
                new_y1 = 0
            if new_x2 > 1:
                new_x2 = 1
            if new_y2 > 1:
                new_y2 = 1

            label[i][1], label[i][2], label[i][3], label[i][4] = new_x1, new_y1, new_x2, new_y2


    elif W > H:
        pad = (W - H) // 2
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        for i in range(len(label)):
            x1, y1, x2, y2 = label[i][1], label[i][2], label[i][3], label[i][4]
            x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H
            new_x1 = x1 / W
            new_y1 = (y1 + pad) / W
            new_x2 = x2 / W
            new_y2 = (y2 + pad) / W

            # check if new box is out of range
            if new_x1 < 0:
                new_x1 = 0
            if new_y1 < 0:
                new_y1 = 0
            if new_x2 > 1:
                new_x2 = 1
            if new_y2 > 1:
                new_y2 = 1

            label[i][1], label[i][2], label[i][3], label[i][4] = new_x1, new_y1, new_x2, new_y2

    img = cv2.resize(img, (size, size))

    return img, label

def resize_square_img(img, size):
    """
    step 1: padding black width or height to square
    step 2: resize to equal size param
    """
    H, W, _ = img.shape

    # step 1
    if H > W:
        pad = (H - W) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    elif W > H:
        pad = (W - H) // 2
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # step 2
    img = cv2.resize(img, (size, size))
    return img

def box2poly(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def poly2box(poly):
    x1 = min([point[0] for point in poly])
    x2 = max([point[0] for point in poly])
    y1 = min([point[1] for point in poly])
    y2 = max([point[1] for point in poly])
    return [x1, y1, x2, y2]

def remove_zones(img, zones):
    for zone in zones:
        pts = np.array(zone, np.int32)
        pts = pts.reshape((-1, 1, 2))
        img = cv2.fillPoly(img, [pts], (0, 0, 0))
    return img

def crop_img(img, box, is_scale=False):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box

    if is_scale:
        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)
    else:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # Ensure coordinates are within bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return img[y1:y2, x1:x2]

def format_time(timestamp):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

def cv2_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    base64_img = base64.b64encode(buffer).decode()
    return base64_img

def vote_majority(results):
    return max(set(results), key=results.count)