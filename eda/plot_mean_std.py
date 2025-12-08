from paddleocr import PaddleOCR, TextDetection
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import os 

data_folder = "datasets/MCOCR/images"
absolute_data_folder = os.path.abspath(data_folder)
files = os.listdir(absolute_data_folder)
means = []
stds = []
for file in files: 
    filepath = os.path.join(absolute_data_folder, file)
    name_image = file.split('/')[-1].split('.')[0]
    img = cv2.imread(filepath)
    all_pixels = img.flatten()
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)
    means.append(mean)
    stds.append(std)
assert len(means) == len(stds)
print(f"Number of data: {len(means)}")
plt.figure(figsize=(20, 10))
plt.scatter(means, stds, color='blue', s=10, alpha=0.7)
plt.xlabel('Mean')
plt.ylabel('Std')
plt.title("Distribution of Mean and Std of datasets")
plt.grid(True)
plt.savefig('Mean_Std.png')
print("Saving EDA figure successfully")
