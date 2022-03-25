import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc
import time

from model.ACCoNet_VGG_models import ACCoNet_VGG
from model.ACCoNet_Res_models import ACCoNet_Res
from data import test_dataset

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')
opt = parser.parse_args()

dataset_path = './dataset/test_dataset/'

if opt.is_ResNet:
    model = ACCoNet_Res()
    model.load_state_dict(torch.load('./models/ACCoNet_ResNet/ACCoNet_Res.pth.39'))
else:
    model = ACCoNet_VGG()
    model.load_state_dict(torch.load('./models/ACCoNet_VGG/ACCoNet_VGG.pth.54'))

model.cuda()
model.eval()

# test_datasets = ['EORSSD']
test_datasets = ['ORSSD']

for dataset in test_datasets:
    if opt.is_ResNet:
        save_path = './results/ResNet50/' + dataset + '/'
    else:
        save_path = './results/VGG/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/image/'
    print(dataset)
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, s2, s3, s4, s5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))