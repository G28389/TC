'''
测试
'''

import numpy as np
from matplotlib import pyplot as plt
import os
from utils_img import getStyleImg

os.makedirs('bkgrounds', exist_ok=True)

def show_rand_noise():
    img1 = np.random.rand(224,224,3)

    plt.imshow(img1)
    plt.show()

def save_all_bkground(dataset_path='./all_kinds/paper_dataset/'):
    '''
    提取并保存4918张的四周背景到文件夹./bkgrounds，文件名与原图片文件名相同
    此处判断图片通道数，如果不是三通道则记录下来，删除图片和边框。
    '''
    img_names = os.listdir(dataset_path)
    cnt = 0
    to_del_names = []
    for img in img_names:
        cnt += 1
        if cnt%100==0:
            print(cnt)
        flag = getStyleImg(dataset_path+img, "./bkgrounds/"+img, aimShape=(224,224))

if __name__ == "__main__":
    save_all_bkground()