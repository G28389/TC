#######################
# 显示训练后的边框
#######################
import argparse

import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from torchvision import transforms

from imagenet import ImageNet, load_pretrained_imagenet_framing
from resnet import resnet50

GRID_SIZE = 5
BATCH_SIZE = GRID_SIZE * GRID_SIZE

# postpa = transforms.Compose([#transforms.Lambda(lambda x: x.mul_(1./255)),

#                            transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
#                                                 std=[1,1,1]),
#                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
#                            ])
# postpb = transforms.Compose([transforms.ToPILImage(),
#                              transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #add imagenet mean
#                                                 std=[1,1,1])])
postpa = transforms.Compose([#transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], #add imagenet mean
                                                std=[0.229, 0.224, 0.225]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(t): # to clip results in the range [0,1]
    #t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw examples of attacked ImageNet examples')
    pretrained_path = './pretrained/imagenet-20-08-31-11-00-55.chk'
    output_name = 'examples.png'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    framing = load_pretrained_imagenet_framing(pretrained_path).to(device)

    input = torch.randn((1,3,224,224))
    input = input.to(device)

    opt_img = torch.zeros((1, 3, 32, 228))
    for f in range(4):
        opt_img[0, :, 4*f:4*f+4, :] = framing.attack[..., f]
        opt_img[0, :, 16+4*f:16+4*f+4, :] = framing.attack[..., f]
    
    # with torch.no_grad():
    #     input_att, _ = framing(input, normalize=False)
        

    # input_att = input_att.cpu().permute(0, 2, 3, 1).numpy()
    # print("最终图片形状:", input_att[0].shape)
    func = torch.nn.Sigmoid()
    #out_img2 = func(opt_img.data[0].cpu().squeeze().permute(1, 2, 0))
    out_img = postp(opt_img.data[0].cpu().squeeze())
    #print(framing.parameters())
    plt.imsave("frame_0831_v7.png", np.array(out_img))
    #plt.imsave("frame4_sigmoid.png", np.array(out_img2))
    #plt.show()
    

