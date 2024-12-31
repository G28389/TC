# 测试对抗训练后模型在干净图片上表现
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
    parser = argparse.ArgumentParser(description='计算对抗训练后正常图片准确率')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--worker_num', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default="dataset_train/test/")
    args = parser.parse_known_args()[0]
    
    model = models.resnet50(pretrained=False)
    params = torch.load(args.model_dir)

    model.load_state_dict(params)
    model = model.cuda()
    model.eval()

    data_loader = ImageNet.get_data_loaders(args.data_dir, batch_size=1, normalize=True)


    i=0
    for i, (x, y) in enumerate(data_loader):
        x = x.to("cuda")
        out = model(x)
        print(torch.argmax(out).cpu().item())

    #saved_img = postp(postpa(Adv_img[0]).data.cpu().squeeze())
    #plt.imsave(save_root+(4-len(str(i)))*'0'+str(i)+".jpg", np.array(saved_img))


