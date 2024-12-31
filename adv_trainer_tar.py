# -*- CODING: Utf-8 -*-
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 不指定GPU
import random
import time
import logging

import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from imagenet import ImageNet, load_pretrained_imagenet_framing
from utils import accuracy, get_current_datetime, AverageMeter

from PIL import Image
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
import torchvision.models as models

import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ZERO = torch.tensor(0, device=device)
ONE = torch.tensor(100, device=device)
unloader = transforms.ToPILImage()  # reconvert into PIL image

postpa = transforms.Compose([
                            transforms.Normalize(mean=[0, 0, 0], #add imagenet std
                                                std=[1/0.229, 1/0.224, 1/0.225]),
                           transforms.Normalize(mean=[-0.485, -0.456, -0.406], #add imagenet mean
                                                std=[1, 1, 1]),
                           #transforms.Lambda(lambda x: x[torch.LongTensor([1,2,0])]), #turn to RGB 不需要，ToPILImage自动完成
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(t): # to clip results in the range [0,1]
    #t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img
def post_whole_img(t):
    t = postpa(t)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

imsize = (32,220) 
loader = transforms.Compose([transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)




class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input



cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)


    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential()

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args

        self.target = args.target
        self.dataset = ImageNet

        self.classifier = self.dataset.get_classifier().to(self.device)
        self.classifier.eval()

        self.dataset_path = args.dataset_path
        self.bkground_path = args.bkground_path
        self.frame_chk_save_path = args.frame_chk_save_path
        self.frame_png_save_path = args.frame_png_save_path
        self.adv_png_save_path = args.adv_png_save_path

        self.start_img_index = args.start_index
    
        # 分类器已训练好，不计算梯度
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.train_loader = self.dataset.get_data_loaders(data_path=self.dataset_path, batch_size=args.batch_size)
        
        # 如果是是有目标攻击且未指定目标，则随机目标
        print("攻击类型：", self.target)
        # if self.target == -1:
        #     self.target = random.randint(0, dataset.NUM_CLASSES - 1)
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def adjust_learning_rate(self, epoch):
        if epoch % self.args.lr_decay_wait == 0:
            for param_group in self.optimizer_att_lbgfs.param_groups:
                param_group['lr'] *= self.args.lr_decay_coefficient


    '''
        LBFGS 版本训练
    '''
    def process_epoch_lbfgs(self, cnn2, normalization_mean, normalization_std,
                        style_weight=1000, content_weight=1000, train=True, epoch_id=1, numstep=1000):
        
        global opt_img
        
        if train:
            data_loader = self.train_loader
        else:
            data_loader = self.val_loader


        imgnames = os.listdir(self.bkground_path)
        imgnames.sort()

        img_eff = []
        fixed_targets = []
        with open("label_eff.txt", "rt") as f:
            lines = f.readlines()
            for line in lines:
                img_eff.append(int(line.strip('\n')))

        with open("targets.txt", "rt") as f:
            lines = f.readlines()
            for line in lines:
                fixed_targets.append(int(line.strip('\n').split(' ')[-1]))

        for i, (input, target) in enumerate(data_loader):

            if img_eff[i] ==0:
                continue
            self.img_input = input
            #oriimg = copy.deepcopy(input)
            # 无目标攻击
            self.target = fixed_targets[i]

            self.img_target = target
            
            self.img_target = self.target * self.img_target.new_ones(self.img_target.size())
            self.img_input, self.img_target = self.img_input.to(self.device), self.img_target.to(self.device)

            style_img = image_loader(self.bkground_path+imgnames[i])
            content_img = image_loader(self.bkground_path+imgnames[i])
            model, style_losses, content_losses = get_style_model_and_losses(cnn2,
            normalization_mean, normalization_std, style_img, content_img)
            
            # set model in GPU
            model = model.to(self.device)
            self.framing = self.dataset.get_framing(self.args.width, img_id=i, keep_size=True).to(self.device) #随机初始化
            self.optimizer_att_lbgfs = torch.optim.LBFGS([self.framing.attack], lr=1.0, max_iter=10) 
            self.optimizer_att_lbgfs.zero_grad()

            global final_acc, final_loss_sty, final_loss_att, flag, cnt_succ, final_loss_trunck
            # print("变量flag id", id(flag))
            cnt_succ = 0 
            # opt_img = torch.zeros((1,3,32,220)).to(self.device)
            st = time.time()

            def closure():
                global final_acc, final_loss_sty, final_loss_att, opt_img, flag, cnt_succ, final_loss_trunck
                self.optimizer_att_lbgfs.zero_grad()
                
                # correct the values of updated input image
                opt_img = torch.zeros((1,3,32,220), device=self.device)
                # 计算风格损失
                for f in range(8):
                    opt_img[0, :, 4*f:4*f+4, :] = self.framing.attack[..., f]

                #opt_img.data.clamp_(0, 1)

                model(opt_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score+content_score

                # 计算攻击后的分类准确度
                with torch.set_grad_enabled(train):
                    input_att, _ = self.framing(input=self.img_input) # 添加边框
                    input_att = input_att.to(self.device)

                    output_att = self.classifier(input_att) # logits
                    loss_att = self.args.att_weight*self.criterion(output_att, self.img_target)
                   # loss_trunked = torch.abs(self.framing.attack.mul(255).byte().div(255.0)-self.framing.attack).sum() # L1范数
                    loss_trunked = torch.norm(torch.floor(self.framing.attack.mul(255)).div(255.0)-self.framing.attack,1) # L1范数

                acc = accuracy(output_att, self.img_target)
                final_acc = acc
                final_loss_att = loss_att
                final_loss_sty = loss
                final_loss_trunck = loss_trunked

                tol_loss = loss+loss_att+0.01*loss_trunked
                if acc==ONE:
                    flag = True
                    cnt_succ += 1
                    if cnt_succ>40: #flag == True and 
                        
                        tol_loss.fill_(0.0)
                        return tol_loss
                # else:
                #     cnt_succ = max(0, cnt_succ-1)
                #     flag = False
                #print("now acc,loss_sty,loss_att:", final_acc, final_loss_sty, final_loss_att)

                # for test
                # print("acc,loss_sty,loss_att:", final_acc, final_loss_sty, final_loss_att)
                
                
                tol_loss.backward()
                return tol_loss

            j = 0
            while j<1000:
                j += 1
                self.optimizer_att_lbgfs.step(closure)
                self.framing.attack.data = self.framing.attack.clamp(0,1)
                if j%10 == 0:
                    # self.lr = self.lr*0.98
                    # self.optimizer_att_lbgfs = torch.optim.LBFGS([self.framing.attack], lr=self.lr, max_iter=10) 
                    print("acc,loss_sty,loss_att, loss_trunk:", final_acc, final_loss_sty, final_loss_att, final_loss_trunck)
                    for param_group in self.optimizer_att_lbgfs.param_groups:
                        param_group['lr'] *= self.args.lr_decay_coefficient
                #print("变量flag id", id(flag))
                if cnt_succ>40:
                    print("success")
                    break
            
            print("time for this img (min):", (time.time()-st)/60)
            
            # 保存对抗样本
            input_att, _ = self.framing(input=self.img_input) 
            saved_whole_img = np.array(post_whole_img(input_att.data[0].cpu().squeeze()))
            #print(saved_whole_img.shape, np.min(saved_whole_img), np.max(saved_whole_img))
            
            plt.imsave(self.adv_png_save_path+imgnames[i].split('.')[0]+".png", saved_whole_img)
            #exit(0)
            
            # 保存边框图片
            opt_img = torch.ones((1,3,32,220))
            for f in range(8):
                opt_img[0, :, 4*f:4*f+4, :] = self.framing.attack[..., f]

            saved_img = postp(opt_img.data[0].cpu().squeeze())
            plt.imsave(self.frame_png_save_path+imgnames[i].split('.')[0]+".png", np.array(saved_img))
            print("final acc,loss_sty,loss_att, loss_trunck:", final_acc, final_loss_sty, final_loss_att, final_loss_trunck)
            torch.save({
                    'framing': self.framing.state_dict(),
            }, self.frame_chk_save_path+imgnames[i].split('.')[0]+'.chk')

            
    def train(self, epochs):

        for epoch in range(1, 2):
            print("epoch", epoch)
            self.process_epoch_lbfgs(cnn, cnn_normalization_mean, cnn_normalization_std,
                            train=True, epoch_id=epoch)
            
            # self.adjust_learning_rate(epoch)
        
