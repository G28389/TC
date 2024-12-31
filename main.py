# 20220404 模块化重构
import argparse
import os
import time
from adv_trainer_tar import Trainer
# from adv_trainer_untar import
from resnet50_no_defence import acc_cal
import torch
import numpy as np
import random
def init_random():
    seed = 65537
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
#-------------------------


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Style Adversarial Framing')
    parser.add_argument('--width', '-w', default=4, type=int, help='Width of the framing')
    parser.add_argument('--keep-size', action='store_true',
                        help='If set, image will be rescaled before applying framing, so that unattacked and attacked '
                             'images have the same size.')
    parser.add_argument('--target', type=int, default=1, # 
                        help='Target class. If unspecified, untargeted attack will be performed. If set to -1, '
                             'target will be chosen randomly. Note that in targeted attack we aim for higher accuracy '
                             'while in untargeted attack we aim for lower accuracy.')
    parser.add_argument('--epochs', default=1,  type=int)
    parser.add_argument('--batch-size', default=1,  type=int) # batch-size设置为1，加载单张图片
    parser.add_argument('--lr', default=1, type=float, help='Initial learning rate for the framing')
    parser.add_argument('--lr-decay-wait', default=1, type=int, help='How often (in epochs) is lr being decayed')
    parser.add_argument('--lr-decay-coefficient', default=0.95, type=float,
                        help='When learning rate is being decayed, it is multiplied by this number')

    parser.add_argument('--dataset_path', default='./dataset_train/all_kinds')
    parser.add_argument('--bkground_path', default='./dataset_train/bkgrounds/')
    parser.add_argument('--adv_save_root', default="0614/")
    parser.add_argument('--start_index', default=0) #从哪张图片开始处理
    parser.add_argument('--att_weight', type=int, default=2000)
    args = parser.parse_known_args()[0]

    parser.add_argument('--frame_chk_save_path', default=os.path.join(args.adv_save_root,'adv_frame_chk/'))
    parser.add_argument('--frame_png_save_path', default=os.path.join(args.adv_save_root,'adv_frame_png/'))

    # parser.add_argument('--adv_png_save_path', default=os.path.join(args.adv_save_root,'tar_adv_img/data/'))
    parser.add_argument('--adv_png_save_path', default="0614/")
    args = parser.parse_known_args()[0]

    os.makedirs(args.frame_chk_save_path, exist_ok=True)
    os.makedirs(args.frame_png_save_path, exist_ok=True)
    os.makedirs(args.adv_png_save_path, exist_ok=True)
    print("本次运行参数:")
    print('攻击权重:', args.att_weight)
    print("save path:", args.frame_chk_save_path, args.frame_png_save_path, args.adv_png_save_path)
    print()
    
    if not os.listdir(args.adv_png_save_path):
     # 固定各种随机种子
     init_random()
     start_time = time.time()
     print('开始攻击')
     Trainer(args).train(args.epochs)
     end_time = time.time()
     print("Total generating time(minute):", (end_time-start_time)/60)

    # 计算无防御攻击成功率
    print("计算无防御攻击成功率...")
    acc_cal(args.adv_save_root, use_defense=False)

    # 计算PD防御攻击成功率
    print("计算PD防御攻击成功率...")
    # 固定各种随机种子
    init_random()
    acc_cal(args.adv_save_root, use_defense=True, times=50)
    acc_cal(args.adv_save_root, use_defense=True, times=100)


