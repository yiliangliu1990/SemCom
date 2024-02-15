# coding:utf-8
import os
import argparse
import time
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


from util.util import visualize
from model.MFNet_Ray_pre_awgn import MFNet_Ray
# from train_Ray import model_dir

torch.random.manual_seed(123456)

# device = torch.device("cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

image_dir = '/home/nuosen/SemComPLS/SemCom_PLS/image/testimage/'
save_dir = '/home/nuosen/SemComPLS/SemCom_PLS/image/Eve_std0.1/'
model_dir = '/home/nuosen/SemComPLS/SemCom_PLS/weights/MFNet_Ray/'
n_class = 9

# Eve联合信源信道解码器参数加噪声的标准差
std_dev = 0.1

def main():

    model = eval(args.model_name)(n_class=n_class)
    if args.gpu >= 0: model.to(device)
    if os.path.exists(final_model_file):
        model.load_state_dict(torch.load(final_model_file))
    elif os.path.exists(checkpoint_model_file):
        model.load_state_dict(torch.load(checkpoint_model_file))
    else:
        raise Exception('| model file do not exists in %s' % model_dir)
    print('| model loaded!')

    # print(model.decode0.conv.weight.shape)

    # decode7 ~ decode0 shape
    # [16, 2, 3, 3]
    # [48, 16, 3, 3]
    # [96, 48, 3, 3]
    # [96, 96, 3, 3]
    # [48, 96, 3, 3]
    # [48, 48, 3, 3]
    # [16, 48, 3, 3]
    # [9, 16, 3, 3]

    model.decode7.conv.weight = Parameter(model.decode7.conv.weight.to(device) + torch.normal(mean=0, std=std_dev, size=[16, 2, 3, 3]).to(device))
    model.decode6.conv.weight = Parameter(model.decode6.conv.weight.to(device) + torch.normal(mean=0, std=std_dev, size=[48, 16, 3, 3]).to(device))
    model.decode5.conv.weight = Parameter(model.decode5.conv.weight.to(device) + torch.normal(mean=0, std=std_dev, size=[96, 48, 3, 3]).to(device))
    model.decode4.conv.weight = Parameter(model.decode4.conv.weight.to(device) + torch.normal(mean=0, std=std_dev, size=[96, 96, 3, 3]).to(device))
    model.decode3.conv.weight = Parameter(model.decode3.conv.weight.to(device) + torch.normal(mean=0, std=std_dev, size=[48, 96, 3, 3]).to(device))
    model.decode2.conv.weight = Parameter(model.decode2.conv.weight.to(device) + torch.normal(mean=0, std=std_dev, size=[48, 48, 3, 3]).to(device))
    model.decode1.conv.weight = Parameter(model.decode1.conv.weight.to(device) + torch.normal(mean=0, std=std_dev, size=[16, 48, 3, 3]).to(device))
    model.decode0.conv.weight = Parameter(model.decode0.conv.weight.to(device) + torch.normal(mean=0, std=std_dev, size=[9, 16, 3, 3]).to(device))
    # print(model.decode1.conv.weight)

    files = os.listdir(image_dir)
    fpath  = []
    for _, file in enumerate(files):
        fpath1  = []
        if file[-3:] != 'png': continue
        images = (np.asarray(Image.open(image_dir + file))/255)

        images = torch.unsqueeze(torch.tensor(images, dtype=torch.float32), dim=0).permute((0,3,1,2))
        images = Variable(torch.tensor(images))
        if args.gpu >= 0: images = images.to(device)        

        if file[-3:] != 'png': continue
        fpath1.append(save_dir + file)

        model.eval()
        with torch.no_grad():
            logits = model(images)
            predictions = logits.argmax(1)
            visualize(fpath1, predictions)

    print('| prediction files have been saved')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run MFNet_Ray demo with pytorch')
    parser.add_argument('--model_name', '-M',  type=str, default='MFNet_Ray')
    parser.add_argument('--gpu',        '-G',  type=int, default=2)
    args = parser.parse_args()

    # model_dir = os.path.join(model_dir, 'MFNet_Ray')

    checkpoint_model_file = os.path.join(model_dir, 'tmp.pth')
    final_model_file      = os.path.join(model_dir, 'final.pth')

    print('| running %s demo on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()
