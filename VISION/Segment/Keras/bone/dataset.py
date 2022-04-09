import cv2
from matplotlib import pyplot as plt
import shutil
import os
from glob import glob
from tqdm import tqdm
imgs = glob('imgs/*')
masks = glob('mask/*')

error_img = []
error_mask = []

print('img processing...')

for img in tqdm(imgs):
    name = os.path.basename(img).split('.')[0]
    if not os.path.exists(f'./{name}'):
        os.makedirs(f'./{name}')
    if not os.path.exists(f'./{name}/images'):
        os.makedirs(f'./{name}/images')
    try:
        os.rename(img, f'./{name}/images/{name}.png')
    except Exception as e:
        error_img.append(img)
        
print('mask processing...')
for mask in tqdm(masks):
    name = os.path.basename(mask).split('.')[0]
    if not os.path.exists(f'./{name}'):
        os.makedirs(f'./{name}')
    if not os.path.exists(f'./{name}/masks'):
        os.makedirs(f'./{name}/masks')
    try:
        os.rename(mask, f'./{name}/masks/{name}.png')
    except Exception as e:
        error_mask.append(mask)
        