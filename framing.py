# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import copy
import os
from PIL import Image

# 四周背景图片位置
imgnames = os.listdir('./dataset_train/bkgrounds')
imgnames.sort()

loader = transforms.Compose([transforms.ToTensor()])  # transform it into a torch tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class Framing(nn.Module):
    def __init__(self, width, image_side_length, normalize, scale, img_id, keep_size=True):
        '''
            img_id是当前处理的图像id
        '''
        super(Framing, self).__init__()

        self.width = width
        # If keep_size is True, unattacked and attacked images will have the same size.
        self.keep_size = keep_size
        if keep_size:
            self.length = image_side_length - width
        else:
            self.length = image_side_length + width

        # 改为8 
        self.attack_shape = [3, self.width, self.length, 8]
        # 使用相应背景初始边框
        frame_init = torch.randn(self.attack_shape) #[3,4,220,8]
        bk_img_now = image_loader('./dataset_train/bkgrounds/'+imgnames[img_id]) #[1,3,32,220]
        for i in range(8):
            frame_init[:, :, :, i] = bk_img_now[0, :, i*4:4+i*4, :]

        self.attack = nn.Parameter(frame_init)
        
        self.sigmoid = nn.Sigmoid()
        self.normalize = normalize
        self.scale = 1 #传入默认为1

    def forward(self, input, do_normalize=True):
        # 尝试修改
        #attack = self.scale * self.sigmoid(self.attack)
        attack = self.attack.clamp(0, 1)

        if do_normalize:
            attack = self.normalize(attack)

        input_size = input.size()
        attacked_size = list(input_size)
        if not self.keep_size:
            attacked_size[-2] = self.length + self.width
            attacked_size[-1] = self.length + self.width

        if len(input_size) == 5:
            # swap color and time dimensions, merge batch and time dimensions
            input = input.permute(0, 2, 1, 3, 4).contiguous().view(-1, input_size[1], input_size[3], input_size[4])

        attacked_size_merged = list(input.size())
        if not self.keep_size:
            attacked_size_merged[-2] = self.length + self.width
            attacked_size_merged[-1] = self.length + self.width

        inner = input

        # framed_input = input.new_zeros(attacked_size_merged)
        # framed_input[..., self.width:-self.width, self.width:-self.width] = inner
        framed_input = copy.deepcopy(inner)
        framed_input[..., :self.width, :-self.width] = attack[..., 0]
        framed_input[..., :-self.width, -self.width:] = attack[..., 1].transpose(1, 2)
        framed_input[..., -self.width:, self.width:] = attack[..., 2]
        framed_input[..., self.width:, :self.width] = attack[..., 3].transpose(1, 2)


        if len(input_size) == 5:
            framed_input = framed_input.view(attacked_size[0], attacked_size[2], attacked_size[1], attacked_size[3],
                                             attacked_size[4])
            framed_input = framed_input.permute(0, 2, 1, 3, 4)

        return framed_input, attack
