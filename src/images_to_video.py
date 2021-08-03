import cv2, os
from glob import glob

frame_rate = 20
image_size = (1920,1080)
img_seq_dir = 'D:\Ford_Intern\intern-p1\data\predicts'
image_paths = glob(os.path.join(img_seq_dir, '*.png'))
image_paths.sort()

writer = cv2.VideoWriter('D:\Ford_Intern\intern-p1\src\last_results_1080p.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, image_size)

for image_path in image_paths:
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    writer.write(img)