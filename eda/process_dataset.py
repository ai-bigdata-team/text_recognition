from paddleocr import PaddleOCR, TextDetection
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import os 

STD_THRESHOLD = 20 
data_folder = "datasets/MCOCR/images"
absolute_data_folder = os.path.abspath(data_folder)
files = os.listdir(absolute_data_folder)

def improve_contrast(img, mean, scale = 3):
    H, W, C = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    all_pixels = img_gray.flatten()
    new_all_pixels = []
    for pixel in all_pixels:
        dist = pixel - mean
        new_pixel = pixel + dist*(scale-1)
        new_pixel = max(0, new_pixel)
        new_pixel = min(255, new_pixel)
        new_all_pixels.append(new_pixel)
    new_img = np.array(new_all_pixels).reshape(H, W)
    return new_img

numImg = 0
for file in files: 
    filepath = os.path.join(absolute_data_folder, file)
    name_image = file.split('/')[-1].split('.')[0]
    img = cv2.imread(filepath)
    all_pixels = img.flatten()
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)
    img = improve_contrast(img, mean)
    cv2.imwrite(f'datasets/high_contrast/{name_image}.jpg', img)
    numImg +=1 
    # if std <= STD_THRESHOLD:
    #     img = improve_contrast(img, mean)
    #     cv2.imwrite(f'datasets/high_contrast/{name_image}.jpg', img)
    #     numImg +=1 
    # else: 
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     cv2.imwrite(f'datasets/high_contrast/{name_image}.jpg', img)
    #     numImg +=1
print(f"number of images changed: {numImg}")

