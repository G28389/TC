# pixel deflaction defense exists or not
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from imagenet import ImageNet, load_pretrained_imagenet_framing
from resnet import resnet50

from random import randint, uniform
import numpy as np
import random
import os
random.seed(33337)

import copy
import argparse

# 防御
def pixel_deflection_without_map(img, deflections, window):
    img = np.copy(img)
    C, H, W = img.shape
    while deflections > 0:
        #for consistency, when we deflect the given pixel from all the three channels.
        for c in range(C): 
            x,y = randint(0,H-1), randint(0,W-1)
            while True: #this is to ensure that PD pixel lies inside the image
                a,b = randint(-1*window,window), randint(-1*window,window)
                if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
            # calling pixel deflection as pixel swap would be a misnomer,
            # as we can see below, it is one way copy
            img[c,x,y] = img[c,x+a,y+b] 

        deflections -= 1
    return img

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='paralle')
    parser.add_argument('--start_index', '-st', default=0, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--out_dir', default='../results/tar_adv_img/')
    parser.add_argument('--defense', default=0, type=int)
    args = parser.parse_known_args()[0]
    imgPath = '../dataset/all_kinds/' # 加载原图
    pretrained_path = '../results/adv_frame_chk/'
    out_dir = args.out_dir #保存目标攻击defense or 未经防御的对抗样本
    os.makedirs(out_dir, exist_ok=True)

    # Use GPUs
    device = torch.device("cpu")

    data_loader = ImageNet.get_data_loaders(data_path=imgPath, batch_size=1, num_workers=0, normalize=False)

    
    classifier = ImageNet.get_classifier().to(device)
    classifier.eval()
    chk_name = os.listdir(pretrained_path)
    chk_name.sort() # necessary in linux

    img_names = os.listdir('../results/adv_frame_png/')
    img_names.sort()

    cnt = 0
    cnt_effect = 0
    cnt_total = 0
    cnt_temp = 0
    for i, (input, target) in enumerate(data_loader):
        if i >= len(img_names):
            break
        if i < args.batch_size*(int)(args.start_index):
            continue
        if i >= args.batch_size+args.batch_size*(int)(args.start_index):
            break

        input = input.to(device)
        framing = load_pretrained_imagenet_framing(pretrained_path+chk_name[i]).to(device)
        with torch.no_grad():
            input_att, _ = framing(input, do_normalize=False)


        normalized_input = input.clone()
        normalized_input_att = input_att.clone()

        # pixel deflection on adversarial image
        if args.defense==1:
            for n in range(1):
                normalized_input_att[n] = torch.from_numpy(pixel_deflection_without_map(normalized_input_att[n], deflections=50, window=10))

        saved = copy.deepcopy(normalized_input_att)
        for id in range(len(normalized_input)):
            normalized_input[id] = ImageNet.normalize(normalized_input[id])
            normalized_input_att[id] = ImageNet.normalize(normalized_input_att[id])
        

        with torch.no_grad():
            pred = classifier(normalized_input)
            output = torch.argmax(pred, dim=1)
            output_att = torch.argmax(classifier(normalized_input_att), dim=1)


        input_att = input_att.cpu().permute(0, 2, 3, 1).numpy()

        #保存对抗样本到results文件夹,并计算攻击成功率
        cnt_total += 1
        print(i, output[0].item(), output_att[0].item())#, output_att[i].item())#
        if args.defense==0:
            plt.imsave(out_dir+'adv_'+'0'*(5-len(str(i)))+str(i)+'.png', saved.cpu().permute(0, 2, 3, 1).numpy()[0])

