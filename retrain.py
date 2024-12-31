'''
根据DeepRobust库
'''
import random

"""
This is an implementation of pgd adversarial training.
References
----------
..[1]Mądry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017).
Towards Deep Learning Models Resistant to Adversarial Attacks. stat, 1050, 9.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F

import numpy as np
from PIL import Image
# from deeprobust.image.attack.pgd import PGD
# from deeprobust.image.defense.base_defense import BaseDefense

from imagenet import ImageNet
from utils import read_tar

# PGD攻击超参数
epsilon_set = 8/255
step_s = 1/255
c_min = 0
c_max = 1
max_num_steps = 10
batch_size = 8

labels_all = []

with open("./dataset_train/labels_eff.txt", "rt") as f:
    lines = f.readlines()
    for line in lines:
        labels_all.append( int(line.strip("\n")) )

labels_train = []
with open("./dataset_train/labels_train.txt", "rt") as f:
    lines = f.readlines()
    for line in lines:
        labels_train.append( int(line.strip("\n")) )

labels_test = []
with open("./dataset_train/labels_test.txt", "rt") as f:
    lines = f.readlines()
    for line in lines:
        labels_test.append( int(line.strip("\n")) )

def train_all_clean(model,device, train_loader, optimizer, epoch):

    model.train()

        # 由于预训练模型在大数据上训练，这里固定BN不动，否则精度大幅下降
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    model.apply(set_bn_eval)

    correct = 0
    bs = train_loader.batch_size

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        data= data.to(device)
            
        batch_label_np = np.array(labels_all[batch_idx*batch_size:batch_idx*batch_size+batch_size])
        batch_label = torch.from_numpy(batch_label_np).to("cuda")

        output = model(data)
        # loss = calculate_loss(output, batch_label.flatten())
        loss = F.cross_entropy(output,batch_label.flatten())

        loss.backward()
        optimizer.step()
        pred = output.argmax(dim = 1, keepdim = True)
        correct = pred.eq(batch_label.view_as(pred)).sum().item()
        print("训练中, 干净样本成功率:", 1.0*correct/bs)

def calculate_loss(self, output, target, redmode = 'mean'):
        """
        Calculate loss for training.
        """

        loss = F.cross_entropy(output, target, reduction = redmode)
        return loss

def test_clean(model, device, test_loader, label_path):

    model.eval()
    labels = read_tar(label_path)
        
    # 测试对干净样本识别准确率
    model.eval()
    device="cuda"
    # 计算针对防御训练后模型的攻击成功率
    i=0
    cnt = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = torch.argmax(model(data))
            
        if output.item()==int(labels[i]):
            cnt += 1
            #print(i)
        i += 1
        # if i%200==0:
        #     print("识别进度:", i, cnt)
    print("干净识别准确率:", cnt/len(test_loader.dataset))


class Finetuning(nn.Module):

    def __init__(self, model, device):
        if not torch.cuda.is_available():
            print('CUDA not availiable, using cpu...')
            self.device = 'cpu'
        else:
            self.device = device

        self.model = model
        self.epoch = 2#10
        self.save_model = True
        self.save_dir = "./defense_models_after/"
        self.save_name = "pgdtraind_"
        self.epsilon = epsilon_set
        self.num_steps = max_num_steps
        self.perturb_step_size = step_s
        self.lr = 0.00001
        self.momentum = 0.1
        self.save_per_epoch = 1 # 每轮对抗训练后都保存
        self.train_labels = read_tar("./dataset_train/labels_train.txt")
        self.test_labels = read_tar("./dataset_train/labels_test.txt")

    def generate(self, train_loader, test_loader, test_target_file_path):
        """
        Call this function to generate robust model.
        """

        torch.manual_seed(100)
        device = torch.device(self.device)

        optimizer = optim.Adam(self.model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma = 0.1)
        save_model = True
        for epoch in range(1, self.epoch + 1):
            print('Training epoch: ', epoch, flush = True)
            self.train_all_clean(self.device, train_loader, optimizer, epoch)
            #self.test(self.model, self.device, test_loader, test_target_file_path, eps=epsilon_set)
            #self.test(self.model, self.device, test_loader, test_target_file_path, eps=0.2/0.229)

            if (self.save_model and epoch % self.save_per_epoch == 0):
                if os.path.isdir(str(self.save_dir)):
                    torch.save(self.model.state_dict(), str(self.save_dir) + self.save_name + '_epoch' + str(epoch) + '.pth')
                    print("model saved in " + str(self.save_dir))
                else:
                    print("make new directory and save model in " + str(self.save_dir))
                    os.mkdir('./' + str(self.save_dir))
                    torch.save(self.model.state_dict(), str(self.save_dir) + self.save_name + '_epoch' + str(epoch) + '.pth')

            scheduler.step()

        return self.model
    '''
    def parse_params(self,
                     epoch_num = 2,
                     save_dir = "./defense_models",
                     save_name = "fgsm_0.1",
                     save_model = True,
                     epsilon = 0.1/0.229,
                     num_steps = 10,
                     perturb_step_size = 1.01/0.229,
                     lr = 0.1,
                     momentum = 0.1,
                     save_per_epoch = 5):

        self.epoch = epoch_num
        self.save_model = True
        self.save_dir = save_dir
        self.save_name = save_name
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.perturb_step_size = perturb_step_size
        self.lr = lr
        self.momentum = momentum
        self.save_per_epoch = save_per_epoch
    '''
    def train(self, device, train_loader, optimizer, epoch):

        self.model.train()

        # 由于预训练模型在大数据上训练，这里固定BN不动，否则精度大幅下降
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.model.apply(set_bn_eval)

        correct = 0
        bs = train_loader.batch_size
        #scheduler = StepLR(optimizer, step_size = 10, gamma = 0.5)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [70], gamma = 0.1)

        adversary = PGD(self.model, device="cuda", targeted=True)
        adversary.parse_params(epsilon=self.epsilon, num_steps=self.num_steps, step_size=self.perturb_step_size, print_process = False)
        import copy
        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()

            data= data.to(device)
            data_ = copy.deepcopy(data)
            #训练阶段随机设置攻击目标
            rand_aim = np.random.randint(1000, size=batch_size)
            target = torch.from_numpy(rand_aim).to("cuda")
            
            batch_label_np = np.array(labels_train[batch_idx*batch_size:batch_idx*batch_size+batch_size])
            batch_label = torch.from_numpy(batch_label_np).to("cuda")

            data_adv = adversary.generate(data, target.flatten())
            output = self.model(data_adv)

            
            #print("on clean", torch.argmax(out_clean, dim=1))
            #print("label:", labels_train[batch_idx*batch_size:batch_idx*batch_size+batch_size])
            #print("output adv:",  torch.argmax(output, dim=1))
            #print("output clean:",  torch.argmax(out_clean, dim=1))
            optimizer.zero_grad() # 防止对抗样本产生期间的梯度影响
            
            loss = self.calculate_loss(output, batch_label.flatten())
            
            # if loss.item()>2.0:
            #     #print("更新")
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # 再次使用干净样本
            optimizer.zero_grad()
            out_clean = self.model(data_)
            loss_clean = self.calculate_loss(out_clean, batch_label.flatten())
            loss_clean.backward()
            optimizer.step()
            pred_clean = out_clean.argmax(dim = 1, keepdim = True)
            #print("干净batch样本正确率:", pred_clean.eq(batch_label.view_as(pred)).sum().item())
            #print("定向攻击成功率:", pred.eq(target.view_as(pred)).sum().item())
            #print every 10
            if batch_idx % 5 == 0:
                print("干净batch样本正确率:", pred_clean.eq(batch_label.view_as(pred_clean)).sum().item())
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t对抗样本识别正确Loss(对抗训练时越小防御越好): {:.6f}\t对抗样本定向攻击成功率:{:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), 100 * correct/(bs)))
            correct = 0

            #scheduler.step()

        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()

            data= data.to(device)
            
            batch_label_np = np.array(labels_all[batch_idx*batch_size:batch_idx*batch_size+batch_size])
            batch_label = torch.from_numpy(batch_label_np).to("cuda")

            output = self.model(data)
            loss = self.calculate_loss(output, batch_label.flatten())

            loss.backward()
            optimizer.step()
            pred = output.argmax(dim = 1, keepdim = True)
            correct = pred.eq(batch_label.view_as(pred)).sum().item()
            print("训练中, 干净样本成功率:", 1.0*correct/bs)


    def test(self, model, device, test_loader, target_file_path):

        # 测试有目标攻击
        model.eval()

        att_loss = 0
        correct_adv = 0
        targets = read_tar(target_file_path) # 攻击目标
        index=0
        adversary = PGD(self.model, device="cuda", targeted=True)
        adversary.parse_params(epsilon=self.epsilon, num_steps=self.num_steps, step_size=self.perturb_step_size, print_process = False)
        device="cuda"
        # 计算针对防御训练后模型的攻击成功率
        i=0
        import matplotlib.pyplot as plt
        import torchvision.transforms as transforms
        postpb = transforms.Compose([transforms.ToPILImage()])
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #target[0] = int(targets[index])

            # 测试训练集使用随机目标  
            rand_aim = np.random.randint(1000, size=1)
            target = torch.from_numpy(rand_aim).to("cuda")
            data_adv = adversary.generate(data, target.flatten())
            output_adv = self.model(data_adv)

            att_loss += self.calculate_loss(output_adv, target, redmode = 'sum').item()  # sum up batch loss
            pred_adv = output_adv.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
            correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
            plt.imsave("att_imgs/pgd0409/imgs/"+str(i)+".png", np.array(postpb(data_adv[0].cpu().squeeze())))
            i += 1
            index += 1
            if i%200==0:
                print("识别进度:", i, correct_adv)
        att_loss /= len(test_loader.dataset)
        
        #print("测试集上,eps =", eps)
        print('\n测试集上攻击损失(越小攻击效果越好): {:.3f}, 定向攻击成功率: {}/{} ({:.0f}%)\n'.format(
            att_loss, correct_adv, len(test_loader.dataset),
            100. * correct_adv / len(test_loader.dataset)))
        

    def test_clean(self, model, device, test_loader, label_path):
        model.eval()
        labels = read_tar(label_path)
        
        # 测试对干净样本识别准确率
        model.eval()
        device="cuda"
        # 计算针对防御训练后模型的攻击成功率
        i=0
        cnt = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = torch.argmax(self.model(data))
            
            if output.item()==int(labels[i]):
                cnt += 1
                #print(i)
            i += 1
            # if i%200==0:
            #     print("识别进度:", i, cnt)
        print("干净识别准确率:", cnt/len(test_loader.dataset))


    def calculate_loss(self, output, target, redmode = 'mean'):
        """
        Calculate loss for training.
        """

        loss = F.cross_entropy(output, target, reduction = redmode)
        return loss


if __name__ == '__main__':

    torch.manual_seed(100)

    save_dir = "./defense_models_after/"
    save_name = "cleantraind_"
    epsilon = epsilon_set
    num_steps = max_num_steps
    perturb_step_size = step_s
    lr = 0.00001
    momentum = 0.1
    save_per_epoch = 1 # 每轮对抗训练后都保存
    train_labels = read_tar("./dataset_train/labels_train.txt")
    test_labels = read_tar("./dataset_train/labels_test.txt")

    model = models.resnet50(pretrained=True)
    optimizer = optim.Adam(model.parameters(),lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma = 0.1)
    save_model = True
    epochs = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = ImageNet.get_data_loaders("dataset_train/train", batch_size=batch_size, num_workers=2)
    test_loader = ImageNet.get_data_loaders("dataset_train/test", batch_size=1, num_workers=2)
    test_target_file_path = "dataset_train/targets_test.txt"
    model.to(device)

    test_clean(model=model,test_loader=test_loader,device=device,label_path=test_target_file_path)
  
    #  for epoch in range(1,epochs+1):
    #     print('Training epoch: ', epoch, flush = True)
        
    #     train_all_clean(model=model,device=device,train_loader=train_loader,optimizer=optimizer,epoch=epoch)
        
    #     if (save_model and epoch % save_per_epoch == 0):
    #             if os.path.isdir(str(save_dir)):
    #                 torch.save(model.state_dict(), str(save_dir) + save_name + '_epoch' + str(epoch) + '.pth')
    #                 print("model saved in " + str(save_dir))
    #             else:
    #                 print("make new directory and save model in " + str(save_dir))
    #                 os.mkdir('./' + str(save_dir))
    #                 torch.save(model.state_dict(), str(save_dir) + save_name + '_epoch' + str(epoch) + '.pth')

    #     scheduler.step()

   
    