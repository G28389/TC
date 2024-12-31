# 有目标攻击, epsilon=0.2
import numpy as np
import torch
from torchvision import transforms,datasets, models
from imagenet import ImageNet
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.config import attack_params
from matplotlib import pyplot as plt
from utils import read_label, read_tar
import argparse

postpa = transforms.Compose([#transforms.Lambda(lambda x: x.mul_(1./255)),
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[1., 1., 1.]),  # turn to RGB
                            ])

postpb = transforms.Compose([transforms.ToPILImage()])
def postp(t): # to clip results in the range [0,1]
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='核实对抗训练效果')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--targeted', type=int) # 0表示无目标 1有目标
    parser.add_argument('--worker_num', type=int, default=0)
    args = parser.parse_known_args()[0]
    
    print("测试有目标攻击?", args.targeted)
    model = models.resnet50(pretrained=False)
    params = torch.load(args.model_dir)

    model.load_state_dict(params)
    model.eval()

    data_loader = ImageNet.get_data_loaders("dataset/", batch_size=1, normalize=True)

    '''
    pgd 攻击
    '''
    epsilon_set = 0.2/0.229
    step_s = 0.01/0.229
    if args.targeted == 1:
        adversary = PGD(model, device="cuda", targeted=True)
    else:
        adversary = PGD(model, device="cuda", targeted=False)
    adversary.parse_params(epsilon=epsilon_set, num_steps=40, step_size=0.05, clip_min=-2.12, clip_max=2.63,
                             print_process=False)   
    print("epsilon:", adversary.epsilon)
    labels, names = read_label("labels.txt")
    targets = read_tar("targets_v2.txt")


    i=0
    for i, (x, y) in enumerate(data_loader):
        x = x.to("cuda")
        y = None
        if args.targeted == 1:
            y=torch.tensor([int(targets[i])])
        else:
            y=torch.tensor([int(labels[i])])
        y = y.to("cuda")
        Adv_img = adversary.generate(x, y)
        out = model(Adv_img)
        print(torch.argmax(out).cpu().item())

    #saved_img = postp(postpa(Adv_img[0]).data.cpu().squeeze())
    #plt.imsave(save_root+(4-len(str(i)))*'0'+str(i)+".jpg", np.array(saved_img))

