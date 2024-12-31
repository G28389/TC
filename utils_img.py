

# 给定图片，截取指定长度的边框 拼在一起显示
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
import copy
import os

def resize_img(imgPath, aimShape):
    im = Image.open(imgPath)
    im = im.resize(aimShape, Image.ANTIALIAS)
    im.save('./Images/232_grass_dog.png')

def getStyleImg(imgPath, savePath, aimShape=(224,224)):
    '''
    缩放到aimShape-->裁剪四周作为风格图片和内容图片
    '''
    im = Image.open(imgPath)
    im = im.resize(aimShape, Image.ANTIALIAS)

    im_np = np.array(im)
    width_frame = aimShape[0]-4
    bk_frame = np.zeros((32, width_frame, 3))
    bk_frame[:4, :, :] = im_np[:4, :width_frame, :]
    bk_frame[4:8, :, :] = im_np[:width_frame, width_frame:aimShape[0], :].transpose(1,0,2)
    bk_frame[8:12, :, :] = im_np[width_frame:aimShape[0], 4:aimShape[0], :]
    bk_frame[12:16, :, :] = im_np[4:aimShape[0], :4, :].transpose(1,0,2)
    bk_frame[16:, :, :] = copy.deepcopy(bk_frame[:16, :, :])


    bk_frame = Image.fromarray(bk_frame.astype(np.uint8))
    bk_frame.save(savePath)


def get_dog_label():
    imgPath = 'imgnet_set/dogs_outer/dogs'
    img_names = os.listdir(imgPath)
    with open('dog_1000_lable.txt', 'wt') as f:
        
        for i in range(1000):
            f.write(img_names[i][6:9])
            f.write('\n')


def get_advs(frame_paths = './results/nostyle_adv_v0/', img_paths = 'E:/conda_projs/pytorch-ssim-master/ori/dogs/', save_path='./nostyleadvs/'):
    '''
     将原图和训练过的边框拼接后保存
    '''
    # frame_paths = './results/nostyle_adv_v0/'
    # img_paths = 'E:/conda_projs/pytorch-ssim-master/ori/dogs/'
    
    imgnames = os.listdir(img_paths)
    framenames = os.listdir(frame_paths)
    for i in range(len(framenames)):
        frame = Image.open(frame_paths+framenames[i]).convert('RGB')
        img = Image.open(img_paths+imgnames[i]).convert('RGB').resize((224,224), Image.ANTIALIAS)

        frame = np.array(frame)
        img = np.array(img)
        width = 4
        img[:width, :-width,:] = frame[0:4,:,:]
        img[:-width, -width:,:] = frame[4:8,:,:].reshape((220,4,3))
        img[-width:, width:,:] = frame[8:12,:,:]
        img[width:, :width,:] = frame[12:16,:,:].reshape((220,4,3))

        bk_frame = Image.fromarray(img.astype(np.uint8))
        bk_frame.save(save_path+imgnames[i])

get_advs(frame_paths='./results/mysty_content/', save_path='./my_style_content/')
    