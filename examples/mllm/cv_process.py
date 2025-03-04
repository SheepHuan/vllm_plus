import cv2


video = cv2.VideoCapture("examples/mllm/data/v1.mp4")

reuslt = []
new_shape = (640, 640)

while True:
    ret, frame = video.read()
    if not ret:
        break
    h,w = frame.shape[:2]
    frame = frame[:,:w//2]
    frame = cv2.resize(frame, new_shape)
    reuslt.append(frame)
    
video.release()

# 将result写为视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义视频编码
output_video = cv2.VideoWriter('examples/mllm/data/output_video.mp4', fourcc, 30, new_shape)  # 创建VideoWriter对象

for frame in reuslt:
    output_video.write(frame)  # 写入每一帧

output_video.release()  # 释放VideoWriter对象


