import time

class FPS:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.time()
    
    def end(self):
        self.end_time = time.time()
    
    def get_fps(self):
        fps = round(1 / (self.end_time - self.start_time), 2)
        return fps