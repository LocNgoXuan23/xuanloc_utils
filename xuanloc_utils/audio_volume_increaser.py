import argparse
from moviepy.editor import VideoFileClip

def increase_audio_volume(input_video_path, output_video_path, volume_factor):
    """
    Tăng âm lượng của audio trong video lên volume_factor lần
    
    Args:
        input_video_path (str): Đường dẫn đến file video đầu vào
        output_video_path (str): Đường dẫn đến file video đầu ra
        volume_factor (float): Hệ số tăng âm lượng
    """
    try:
        # Mở video
        video = VideoFileClip(input_video_path)
        
        # Kiểm tra xem video có audio không
        if video.audio is None:
            print(f"Lỗi: Video không có audio.")
            return False
        
        # Tăng âm lượng của audio
        video = video.volumex(volume_factor)
        
        # Xuất video với audio đã được tăng âm lượng
        video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
        
        # Đóng video để giải phóng tài nguyên
        video.close()
        
        print(f"Đã xử lý thành công! Video với âm lượng đã tăng {volume_factor} lần được lưu tại: {output_video_path}")
        return True
    
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Tăng âm lượng của audio trong video.')
    parser.add_argument('input_video', help='Đường dẫn đến video đầu vào')
    parser.add_argument('output_video', help='Đường dẫn đến video đầu ra')
    parser.add_argument('--factor', type=float, default=2.0,
                        help='Hệ số tăng âm lượng (mặc định: 2.0)')
    
    args = parser.parse_args()
    
    increase_audio_volume(args.input_video, args.output_video, args.factor)

if __name__ == "__main__":
    main()