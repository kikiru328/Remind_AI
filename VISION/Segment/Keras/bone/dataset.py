import cv2
from matplotlib import pyplot as plt
import shutil
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

imgs = glob('imgs/*')
masks = glob('mask/*')

X_train, X_val, y_train, y_val = train_test_split(imgs, masks, test_size = 0.1, random_state = 0)



print('train imgs processing...')
error_img = []
for img in tqdm(X_train):
    name = os.path.basename(img).split('.')[0]
    if not os.path.exists('./train'):
        os.makedirs('./train')
    if not os.path.exists(f'./train/{name}'):
        os.makedirs(f'./train/{name}')      
    if not os.path.exists(f'./train/{name}/images'):
        os.makedirs(f'./train/{name}/images')
    try:
        os.rename(img, f'./train/{name}/images/{name}.png')
    except Exception as e:
        error_img.append(img)
print('error >> ', len(error_img))

error_mask = []
print('train_mask processing...')
for mask in tqdm(y_train):
    name = os.path.basename(mask).split('.')[0]
    if not os.path.exists(f'./train'):
        os.makedirs(f'./train')
    if not os.path.exists(f'./train/{name}'):
        os.makedirs(f'./train/{name}')
    if not os.path.exists(f'./train/{name}/masks'):
        os.makedirs(f'./train/{name}/masks')
    try:
        os.rename(mask, f'./train/{name}/masks/{name}.png')
    except Exception as e:
        error_mask.append(mask)

print('error >> ', len(error_mask))

error_val_img = []
print('val imgs processing...')

for img in tqdm(X_val):
    name = os.path.basename(img).split('.')[0]
    if not os.path.exists('./val'):
        os.makedirs('./val')
    if not os.path.exists(f'./val/{name}'):
        os.makedirs(f'./val/{name}')      
    if not os.path.exists(f'./val/{name}/images'):
        os.makedirs(f'./val/{name}/images')
    try:
        os.rename(img, f'./val/{name}/images/{name}.png')
    except Exception as e:
        error_img.append(img)
print('error >> ', len(error_val_img))

error_val_mask = []
print('val mask processing...')
for mask in tqdm(y_val):
    name = os.path.basename(mask).split('.')[0]
    if not os.path.exists(f'./val'):
        os.makedirs(f'./val')
    if not os.path.exists(f'./val/{name}'):
        os.makedirs(f'./val/{name}')
    if not os.path.exists(f'./val/{name}/masks'):
        os.makedirs(f'./val/{name}/masks')
    try:
        os.rename(mask, f'./val/{name}/masks/{name}.png')
    except Exception as e:
        error_mask.append(mask)

print('error >> ', len(error_val_mask))

