# 计算保存成文件的图片的攻击性
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from imagenet import ImageNet, load_pretrained_imagenet_framing
from random import randint, uniform
import numpy as np
import random
import os
random.seed(33337)

# 防御
def pixel_deflection_without_map(img, deflections, window):
   # img = np.copy(img)
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


def acc_cal(adv_results_root, use_defense:bool, times=100, device="cuda"):

    # adv_png_path = os.path.join(adv_results_root, "adv_frame_png/")
    # adv_img_path = os.path.join(adv_results_root,"tar_adv_img/")
    adv_png_path = adv_results_root

    adv_img_path = adv_results_root
    device = "cuda"
    if use_defense==False:
        print("计算无防御准确率")
        adv_loader = ImageNet.get_data_loaders(data_path=adv_img_path, batch_size=1, num_workers=1, normalize=True)
    else:
        print("计算有防御PD{}准确率".format(times))
        
        adv_loader = ImageNet.get_data_loaders(data_path=adv_img_path, batch_size=1, num_workers=1, normalize=False)
    
    print("数据位置:", adv_img_path, len(adv_loader))
    classifier = ImageNet.get_classifier().to(device)
    classifier.eval()

    img_names = os.listdir(adv_png_path)
    img_names.sort()

    cnt = 0
    cnt_effect = 0
    img_eff = []
    tmp_targets = []
    fixed_targets = []
    with open("img_eff.txt", "rt") as f:
        lines = f.readlines()
        for line in lines:
            img_eff.append(int(line.strip('\n')))
    with open("targets_v2.txt", "rt") as f:
        lines = f.readlines()
        for line in lines:
            tmp_targets.append(int(line.strip('\n').split(' ')[-1]))
    
    # for i in range(len(img_eff)):
    #     if img_eff[i]==1:
    #         fixed_targets.append(tmp_targets[i])

    for i in range(len(tmp_targets)):

        fixed_targets.append(tmp_targets[i])

    for i, (input, target) in enumerate(adv_loader):
        cnt_effect += 1
        input = input.to(device)
        normalized_input = input.clone()

        if use_defense==True:
            # pixel deflection on adversarial image
            normalized_input[0] = pixel_deflection_without_map(normalized_input[0], deflections=times, window=10)
            normalized_input[0] = ImageNet.normalize(normalized_input[0])

        with torch.no_grad():
            pred = classifier(normalized_input)
            output = torch.argmax(pred, dim=1)
        #print("预测为:", output[0].item())
        if output[0].item()==fixed_targets[i]:
            cnt += 1
            #print(i, "成功")
        # else:
        #     #print(i, "失败")
        if i>0 and i%500==0:
            print("判别进度:", cnt, cnt_effect)
    print("成功率:", cnt, cnt_effect, 1.0*cnt/cnt_effect)




if __name__ == '__main__':
    
    print("hello world.")
    acc_cal("../results_w1_1_newloss/", True)

    
