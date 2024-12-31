'''
根据DeepRobust库
'''

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
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.defense.base_defense import BaseDefense

from imagenet import ImageNet
from utils import read_tar

# PGD攻击超参数
epsilon_set = 0.1/0.229
step_s = 0.01/0.229
c_min = -2.12
c_max = 2.63
max_num_steps = 40
batch_size = 16

class PGDtraining(BaseDefense):

    def __init__(self, model, device):
        if not torch.cuda.is_available():
            print('CUDA not availiable, using cpu...')
            self.device = 'cpu'
        else:
            self.device = device

        self.model = model
        self.epoch = 20
        self.save_model = True
        self.save_dir = "./defense_models"
        self.save_name = "pgdtraining_0.1"
        self.epsilon = epsilon_set
        self.num_steps = max_num_steps
        self.perturb_step_size = step_s
        self.lr = 0.1
        self.momentum = 0.1
        self.save_per_epoch = 5

    def generate(self, train_loader, test_loader, test_target_file_path):
        """
        Call this function to generate robust model.
        """
        #self.parse_params(**kwargs)

        torch.manual_seed(100)
        device = torch.device(self.device)

        optimizer = optim.Adam(self.model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma = 0.1)
        save_model = True
        for epoch in range(1, self.epoch + 1):
            print('Training epoch: ', epoch, flush = True)
            self.train(self.device, train_loader, optimizer, epoch)
            self.test(self.model, self.device, test_loader, test_target_file_path)

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

    def parse_params(self,
                     epoch_num = 20,
                     save_dir = "./defense_models",
                     save_name = "pgdtraining_0.1",
                     save_model = True,
                     epsilon = 0.1/0.229,
                     num_steps = 10,
                     perturb_step_size = 0.01/0.229,
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

    def train(self, device, train_loader, optimizer, epoch):

        self.model.train()
        correct = 0
        bs = train_loader.batch_size
        #scheduler = StepLR(optimizer, step_size = 10, gamma = 0.5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [70], gamma = 0.1)

        adversary = PGD(self.model, device="cuda")
        adversary.parse_params(epsilon=epsilon_set, num_steps=max_num_steps, step_size=0.05, clip_min=-2.12, clip_max=2.63, print_process = False)

        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()

            data, target = data.to(device), target.to(device)

            data_adv = adversary.generate(data, target.flatten())
            output = self.model(data_adv)

            optimizer.zero_grad() # 防止对抗样本产生期间的梯度影响
            loss = self.calculate_loss(output, target)

            loss.backward()
            optimizer.step()

            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            #print every 10
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), 100 * correct/(bs)))
            correct = 0

            scheduler.step()


    def test(self, model, device, test_loader, target_file_path):

        # 测试有目标攻击
        model.eval()

        test_loss = 0
        correct = 0
        test_loss_adv = 0
        correct_adv = 0
        targets = read_tar(target_file_path) # 攻击目标
        index=0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target[0] = targets[index]

            # print clean accuracy
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # print adversarial accuracy
            data_adv, output_adv = self.adv_data(data, target, ep = self.epsilon, num_steps = self.num_steps)

            test_loss_adv += self.calculate_loss(output_adv, target, redmode = 'sum').item()  # sum up batch loss
            pred_adv = output_adv.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
            correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_loss_adv /= len(test_loader.dataset)

        print('\nTest set: Clean loss: {:.3f}, Clean Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        print('\nTest set: Adv loss: {:.3f}, Adv Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss_adv, correct_adv, len(test_loader.dataset),
            100. * correct_adv / len(test_loader.dataset)))
        index += 1

    def adv_data(self, data, output, ep = 0.3, num_steps = 10, perturb_step_size = 0.01):
        """
        Generate input(adversarial) data for training.
        """

        adversary = PGD(self.model, device="cuda")
        data_adv = adversary.generate(data, output.flatten(), epsilon = ep, num_steps = num_steps, step_size = perturb_step_size)
        output = self.model(data_adv)

        return data_adv, output

    def calculate_loss(self, output, target, redmode = 'mean'):
        """
        Calculate loss for training.
        """

        loss = F.cross_entropy(output, target, reduction = redmode)
        return loss


if __name__ == '__main__':
    resnet50 = models.resnet50(pretrained=True)

    pgd_adv_trainer = PGDtraining(resnet50, "cuda")
    train_loader = ImageNet.get_data_loaders("0613/resnet/adv_tar_img", batch_size=16)
    test_loader = ImageNet.get_data_loaders("dataset_train/test", batch_size=1)
    test_target_file_path = "dataset_train/targets_test.txt"
    pgd_adv_trainer.generate(train_loader, test_loader, test_target_file_path)
