import os
import cv2
import numpy as np

root = '/Users/Downloads/Dataset_BUSI_with_GT/'
img_save_path = '/Users/Downloads/BUSI_processed/train-image/'
mask_save_path = '/Users/Downloads/BUSI_processed/train-mask/'

for file in os.listdir(root):
    if '.DS' in file or 'normal' in file:
        continue

    for item in os.listdir(root + file):
        mask_name = item.replace('.png', '_mask.png')
        img = cv2.imread(root + file + '/' + item)
        mask = cv2.imread(root + file + '/' + mask_name, cv2.IMREAD_GRAYSCALE)
        img_new = cv2.resize(img, (256, 256))
        mask_new = cv2.resize(mask, (256, 256))
        mask_new = np.where(mask_new > 10, 255, 0)
        image_path = img_save_path+item
        mask_path = mask_save_path+item
        cv2.imwrite(image_path, img_new)
        cv2.imwrite(mask_path, mask_new)
        print('saving img', mask_name)