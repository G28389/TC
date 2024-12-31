import json
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from framing import Framing
from resnet import resnet50

with open(os.path.join(os.path.dirname(__file__), 'assets/imagenet_class_index.json')) as f:
    IMAGE_NET_CLASSES_JSON = json.load(f)


class ImageNet:
    MEAN = (0.485, 0.456, 0.406)
    STD_DEV = (0.229, 0.224, 0.225)
    NUM_CLASSES = 1000
    SIDE_LENGTH = 224
    ID_TO_CLASS = [IMAGE_NET_CLASSES_JSON[str(k)][1] for k in range(len(IMAGE_NET_CLASSES_JSON))]
    normalize = transforms.Normalize(MEAN, STD_DEV)

    @staticmethod
    def get_classifier():
        # cnn = models.densenet121(pretrained=True)
        cnn = models.resnet50(pretrained=True)
        # cnn = models.vgg19(pretrained=True)
        
        # pre = torch.load('./resnet_model/resnet50-19c8e357.pth')
        # cnn.load_state_dict(pre)

        # cnn = models.resnet50(pretrained=False)
        # param = torch.load('finetuning_resnet50.pth')
        # cnn.load_state_dict(param)
        return cnn

    @staticmethod
    def get_framing(width, img_id, keep_size=True):
        return Framing(width=width, image_side_length=ImageNet.SIDE_LENGTH, normalize=ImageNet.normalize_not_in_place,img_id=img_id,
                       scale=1., keep_size=keep_size)

    @staticmethod
    def get_data_loaders(data_path, batch_size, num_workers=2, normalize=True, shuffle_val=False):

        val_dir = data_path

        val_transforms= [transforms.Resize((224,224)),transforms.ToTensor()]#  
        if normalize:
            val_transforms.append(ImageNet.normalize)
        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(val_dir, transforms.Compose(val_transforms)),
            batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers, pin_memory=False)
        return val_loader 

    @staticmethod
    def normalize_not_in_place(input):
        params_shape = list(input.size())
        for i in range(1, len(params_shape)):
            params_shape[i] = 1
        mean = input.new_tensor(ImageNet.MEAN).view(params_shape)
        std = input.new_tensor(ImageNet.STD_DEV).view(params_shape)
        return (input - mean) / std


def load_pretrained_imagenet_framing(checkpoint_path, imgid = 1):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    args = checkpoint['args']
    framing = ImageNet.get_framing(args.width, imgid, keep_size=True)#getattr(args, 'keep_size', False))
    framing.load_state_dict(checkpoint['framing'])
    framing.eval()
    return framing
