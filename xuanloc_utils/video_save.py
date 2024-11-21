import cv2

class VideoSave:
    def __init__(self, output_path, video_capture, size=None):
        self.output_path = output_path
        self.video_capture = video_capture

        if size is None:
            self.size = (int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        else:
            self.size = size
            
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_writer = cv2.VideoWriter(
            self.output_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            self.fps, 
            self.size
        )

    def write_frame(self, frame):
        self.video_writer.write(frame)

    def close(self):
        self.video_writer.release()