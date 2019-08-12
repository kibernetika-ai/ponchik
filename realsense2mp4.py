import cv2
import numpy as np
import datetime
import importlib

rs = importlib.import_module('pyrealsense2')

fps = 6
name = 'real.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
w = 1280
h = 720
out = cv2.VideoWriter(name, fourcc, fps, (w, h))
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(
    rs.stream.color,
    w,
    h,
    rs.format.bgr8, 6
)
for i in range(fps * 60):
    frames = pipeline.wait_for_frames()
    new_frame = frames.get_color_frame()
    new_frame = np.asanyarray(new_frame.get_data())
    out.write(new_frame)

out.release()
