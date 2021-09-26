import os
import argparse
import numpy as np
import torch
import pylab as plt
import torch.nn as nn
import time
import math
from models.channel_select import channel_selection
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import torch.optim as optim
import tqdm
import torch.nn.functional as F
import random
from PIL import Image
from latency import test_latency
from math import sin, cos

__all__ = ['resnet']
activations = []

def func(x):
    return 0.3004 + 0.215*cos(x*4.307) + 0.18*sin(x*4.307) + 0.06662*cos(2*x*4.307) - 0.03691*sin(2*x*4.307) + 0.03065*cos(3*x*4.307) - 0.04711*sin(3*x*4.307)
#100 3004


dataSet = 'CIFAR10'

if dataSet == 'CIFAR10':
    data_path = './data/'
    mean = [0.4940607, 0.4850613, 0.45037037]
    std = [0.20085774, 0.19870903, 0.20153421]

elif dataSet == 'CIFAR100':
    data_path = './data/'
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

"""
preactivation resnet with bottleneck design.
"""

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class resnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar10', cfg=None):
        super(resnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
            # cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [[64, 64, 64], [256, 64, 64]*(n-1), [256, 128, 128], [512, 128, 128]*(n-1), [512, 256, 256], [1024, 256, 256]*(n-1), [1024]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 64 #16
                                    #16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 64, n, cfg = cfg[0:3*n]) #16
        self.layer2 = self._make_layer(block, 128, n, cfg = cfg[3*n:6*n], stride=2) #32
        self.layer3 = self._make_layer(block, 256, n, cfg = cfg[6*n:9*n], stride=2) #64
        self.bn = nn.BatchNorm2d(256 * block.expansion) #64
        self.select = channel_selection(256 * block.expansion)#64
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def activation_hook(self, model, input, output):
        BLOCKS = ['layer1', 'layer1.0', 'layer2', 'layer2.0', 'layer3', 'layer3.0',
                  'layer1.0.downsample', 'layer2.0.downsample', 'layer3.0.downsample',
                  '']

        RESIDUALS = ['layer1.0.downsample.0', 'layer2.0.downsample.0', 'layer3.0.downsample.0']

        IF_RESIDUAL = ['layer1.0', 'layer2.0', 'layer3.0']

        x = input[0]
        cov_residual_flag = False
        for name, module in model.named_modules():
            if name in IF_RESIDUAL:
                if module.downsample is not None:
                    cov_residual_flag = True
                residual = x

            if cov_residual_flag is True and name in RESIDUALS:
                residual = module.forward(residual)
                x += residual
                cov_residual_flag = False
                continue

            if name in BLOCKS:
                continue

            if isinstance(module, nn.Linear):
                x = x.view(x.size(0), -1)

            x = module.forward(x)
            # print(name, x.size())
            if isinstance(module, nn.BatchNorm2d):
                activations.append(x.sum(dim=0))

        return

def train_model_all(model, trainSet_path, epoches, batchSize, model_save_path):

    model.cuda()

    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)

    model.train()

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(trainSet_path, transform=transform, download=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(trainSet_path, transform=transform, download=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=True, num_workers=32, drop_last=False)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    optimizer.zero_grad()

    best_acc = 0

    for epoch in range(epoches):
        epoch_loss = 0

        for idx, (data, label) in tqdm.tqdm(enumerate(data_loder)):
            data = data.cuda()
            label = label.cuda()

            output = model(data)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            if idx % 100 == 0:
                print('epoch:', epoch, '\tloss:', loss.item())

        epoch_res, epoch_acc = test(model, data_path, batchSize=64, class_num=100)
        model.train()

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model, model_save_path)
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     torch.save(model, model_save_path)

        scheduler.step()


def train_model(model, trainSet_path, epoches, batchSize, model_save_path, forzen = False):

#    model.cuda()
    model.to('1')

#    if not isinstance(model, torch.nn.DataParallel):
#        model = nn.DataParallel(model)

    model.train()

    if forzen == True:
        for param in model.parameters():
            param.requires_grad = False

        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad = True
                module.bias.requires_grad = True


    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(trainSet_path, transform=transform, download=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(trainSet_path, transform=transform, download=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=8, drop_last=False)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

    optimizer.zero_grad()

    best_loss = 999999999
    best_acc = 0
    inputs = []
    targets = []
    target_class = []
    counts = []

    for idx in range(0, class_num):
        target_class.append(idx)

    for idx in range(0, model.module.fc.out_features):
        counts.append(0)

    for data, label in tqdm.tqdm(data_loder):
        for idx in range(len(label)):
            if label[idx] in target_class and counts[label[idx]] < image_num:
                inputs.append(data[idx].unsqueeze(0).cuda())
                targets.append(label[idx].unsqueeze(0).cuda())
                counts[label[idx]] += 1
   

    file = open('./result10/log'+str(class_num)+'.txt',mode='w')

    for epoch in range(epoches):
        epoch_loss = 0

        for idx in range(0, len(inputs), batchSize):

            if idx + batchSize > len(inputs):
                # break
                input = inputs[idx:len(inputs)-1]
                target = targets[idx:len(inputs)-1]
                data = torch.empty(len(inputs)-idx-1, 3, 32, 32).cuda()
                label = torch.empty(len(inputs)-idx-1).long().cuda()

            else:
                input = inputs[idx:idx+batchSize]
                target = targets[idx:idx+batchSize]
                data = torch.empty(batchSize, 3, 32, 32).cuda()
                label = torch.empty(batchSize).long().cuda()

            for ii, (elem1, elem2) in enumerate(zip(input, target)):
                data[ii] = elem1
                label[ii] = elem2

            output = model(data)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            if idx % 100 == 0:
                print('epoch:', epoch, '\tloss:', loss.item())

        epoch_res, epoch_acc = test(model, data_path, batchSize=512, class_num=class_num)
        model.train()
        file.write('epoch: '+ str(epoch) + '\tacc:' + str(epoch_acc)+ '\n')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model, model_save_path)

        scheduler.step()
    file.close()



def test(model, dataSet_path, batchSize, class_num):

    model.cuda()
#    model.to('1')
#
#    if not isinstance(model, torch.nn.DataParallel):
#        model = nn.DataParallel(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(dataSet_path, transform=transform, download=True, train=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(dataSet_path, transform=transform, download=True, train=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=32, drop_last=False)

    model.eval()
    correct = 0

    classes = []
    for idx in range(model.module.fc.out_features):
        classes.append(0)

    times = 0
    with torch.no_grad():
        for idx, (data, label) in tqdm.tqdm(enumerate(data_loder)):
            input = data.cuda()
            target = label.cuda()

            start = time.time()
            output = model(input)
            pred = torch.argmax(output, 1)
            end = time.time()
            times += end - start

            for idx in range(len(target)):
                if pred[idx] == target[idx]:
                    classes[target[idx]] += 1
            correct += (pred == target).sum()

#    total_acc = float(correct / 10000)

    class_acc = []
    for elem in classes:
        class_acc.append(elem / 1000)


    total_acc = sum(class_acc[0:class_num])/class_num

    # temp = np.array(class_acc)
    # np.save('./result/ResNet50_CIFAR100_all.npy', temp)

    print('\n', 'each class corrects:', classes, '\n',
          'each class accuracy:', class_acc, '\n',
          'time:', times, '\n',
          'total accuracy:', total_acc)

    return classes, round(total_acc, 2)


def compute_L2_activation(activations, CT = None):
    L2_activations = []
    for activation in activations:
        activation = nn.LeakyReLU()(activation)
        L2_activations.append(activation.cpu().norm(dim=(1, 2), p=2).cuda())
    return L2_activations

def read_Img2(dataSet_path, batchSize):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(data_path, transform=transform, download=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(data_path, transform=transform, download=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=8, drop_last=False)

    counts = []
    target_class = []
    inputs = []
    for idx in range(0, model.fc.out_features):
        counts.append(0)

    for idx in range(0, class_num):
        target_class.append(idx)

    #print(len(counts))
    # print(target_class,len(target_class))
    for data, label in tqdm.tqdm(data_loder):
        for idx in range(len(label)):
           # print(idx, label[idx], counts[label[idx]])
            if label[idx] in target_class and counts[label[idx]] < image_num:
                inputs.append(data[idx].cuda())
                counts[label[idx]] += 1

    imgs = torch.empty(len(inputs), 3, 32, 32).cuda()

    for idx, img in enumerate(inputs):
        imgs[idx] = img

    return imgs


# def get_percent(class_num):

def sort_L2_activations(L2_activations, percent, thresholds):
    for tensor in L2_activations:
        y, i = torch.sort(tensor)
        total = len(y)
        thre_index = int(total * percent)
        thre = y[thre_index]
        thre = y.mean()
        thresholds.append(thre)
    return thresholds

def read_Img(ImgFloder_path, transformer, size_H=32, size_W=32):

    path, dirs, files = next(os.walk(ImgFloder_path))
    imgs = torch.empty(len(files), 3, size_H, size_W).cuda()


    for idx, ImgName in enumerate(files):
        img_dir = os.path.join(ImgFloder_path, ImgName)

        img = Image.open(img_dir)
        img = transformer(img)
        img = img.cuda()
        imgs[idx] = img

    return imgs

def get_mask(imgs):

    imgs_masks = []
    thresholds = []
    one_img_masks = []
    CT = 0
    for img in tqdm.tqdm(imgs):

        thresholds.clear()
        activations.clear()
        one_img_masks.clear()

        img = torch.unsqueeze(img, 0)
        handle = model.register_forward_hook(model.activation_hook)

        output = model(img)
        handle.remove()

        L2_activations = compute_L2_activation(activations)
        thresholds = sort_L2_activations(L2_activations, percent=p, thresholds=thresholds)
        CT += 1


        for idx in range(len(thresholds)):
            layer_mask = L2_activations[idx].gt(thresholds[idx]).cuda()
            one_img_masks.append(layer_mask)

        imgs_masks.append(one_img_masks)

    layer_num = len(one_img_masks)
    img_num = len(imgs_masks)
    mask = imgs_masks[0]
    for idx1 in range(layer_num):
        for idx2 in range(img_num):
            mask[idx1] |= imgs_masks[idx2][idx1]

    return mask


def get_img_socre(imgs, sampling=1000):
    img_num = len(imgs)
    score = 0
    for sam_idx in range(sampling):
        img_idx1 = random.randint(0, img_num - 1)
        img_idx2 = random.randint(0, img_num - 1)
        img1 = imgs[img_idx1]
        img2 = imgs[img_idx2]
        KL_div = F.kl_div(img1.softmax(dim=-1).log(), img2.softmax(dim=-1), reduction='mean')
        score += KL_div
    return score / img_num


# def get_compression_rate(img_score, epoches, class_num):
#     return

'''
该剪枝是通过BN层求mask向量来实现的,剪枝的基准为BN层的参数
目前我看的一些demo和paper的方法都是根据BN层去剪枝其它层和BN层本身
还没有根据其它层来进行剪枝的方法
需要使用特殊定义的CS(Channel Selection)层对非BN层之后的卷积层进行剪枝
'''

# Prune settings




print(os.getcwd())

parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=2048, metavar='N',
                    help='input batch size for testing (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=56,
                    help='depth of the resnet, default = 164')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='./logs', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./new_models', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

#model = resnet(depth=args.depth, dataset=args.dataset).cuda()
#print(model)
#exit(0)
# model._initialize_weights()
#
#model = torch.load('./original_models/ResNet50_cifar100.pkl')
model = torch.load('./original_models/ResNet56_cifar10.pkl')
test(model, data_path, batchSize=256, class_num=10)
model = torch.load('./original_models/ResNet56_cifar10_N.pkl')
test(model, data_path, batchSize=256, class_num=10)
exit(0)
#print(model)
#
#
# train_model_all(model=model, epoches=300, batchSize=128,
#            model_save_path='./original_models/Res0Net50_cifar100_2.pkl',
#            trainSet_path=data_path)

# test(model, data_path, batchSize=512, class_num=100)

image_num = 400


for class_num in range(2, 11, 1):
#    # test(model, './data.cifar100', batchSize=512, class_num=class_num)
    model = torch.load('./original_models/ResNet50_cifar10.pkl')
    
    res = './result10/ResNet50_' + str(class_num) + 'C_AutoP.pdf'

    # model = torch.load('./original_models/ResNet56_cifar10_N.pkl')

    res1, acc1 = test(model, data_path, batchSize=512, class_num=class_num)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # test_latency('resnet101',model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    imgs = read_Img2(data_path, 8192)

    img_score = 1.23 * float(get_img_socre(imgs, sampling=class_num*10))
    p = func(class_num / 100) #+ img_score
    print('pruning rate is', p)
    # imgs = read_Img('./mix/', transformer=transform)
    masks = get_mask(imgs)


    # 统计总的通道数
    total = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    # 提取bn层的权重信息存入变量bn中
    # bn = torch.zeros(total)
    # index = 0
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         size = m.weight.data.shape[0]
    #         bn[index:(index+size)] = m.weight.data.abs().clone()
    #         index += size

    # 计算用于bn层预剪枝的阈值,计算方式为按bn层权重大小进行百分比淘汰,淘汰百分之几为超参数
    # y, i = torch.sort(bn)
    # thre_index = int(total * args.percent)
    # thre = y[thre_index]

    '''
    cfg存储每个bn层剩余通道的数目
    pruned存储被取代的通道总数
    cfg_mask代表存储每个bn层对应的mask向量
    '''
    pruned = 0
    cfg = []
    count = 0
    cfg_mask = []
    # 根据上述阈值对bn层进行预剪枝,这里的剪枝并没有真正的裁剪通道,而是简单的设置权重为0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = masks[count]
            if count <= 1:  #100 5
                mask = mask | True
            count += 1
            # mask = weight_copy.gt(thre).float().cuda()
            # mask.shape[0]是当前层通道总数,torch.sum(mask)是修剪的数目
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            # 数乘,与mask向量相乘
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            # 当前层剩余通道数
            cfg.append(int(torch.sum(mask)))
            # 当前层对应的mask向量
            cfg_mask.append(mask.clone())
            # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            #     format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned/total

    print('Pre-processing Successful!')

    # simple test model after Pre-processing prune (simple set BN scales to zeros)


    # 预剪枝之后的模型
    newmodel = resnet(depth=args.depth, dataset=args.dataset, cfg=cfg)
    # if args.cuda:
    #     newmodel.cuda()

    # 计算新模型参数数量
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])

    # 预剪枝的相关日志信息
    '''
    日志内容包括：
    Configuration---按照模型顺序,每个bn层剩余的通道数
    Number of parameters---剩余参数数量
    Test accuracy---测试集精确度
    '''
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        # fp.write("Test accuracy: \n"+str(acc))


    # 正式剪枝
    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())


    '''
    layer_id代表当前的一层
    layer_id_in_cfg代表当前一层对应的mask向量在cfg_mask中的索引
    start_mask用于存储上一层的mask向量
    end_mask表示当前层的mask向量
    conv_count计数被剪枝的卷基层的数目
    '''
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
#    newmodel.classifier = nn.Linear(in_features=cfg[-1], out_features=class_num)

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            # idx1存储当前bn层剩余通道的索引
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(idx1)
            # 只有1个神经元则需要特殊处理
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                # 设置indexes,选择保留的通道
                m2.indexes.data[idx1.tolist()] = 1.0

                # 指向下一层的mask向量
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]

            # 下一层不是cs层
            else:
                # 通过idx1中保存的剩余通道的索引来剪枝
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()

                # 指向下一层的mask向量
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.Conv2d):
            # 对于输入的第一个卷基层不做操作
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue

            # 根据bn层对卷积层进行剪枝
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This converts the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.

                # idx0存储当前层剩余通道的索引
                # idx1存储上一层剩余通道的索引
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))

                # 对于只有1个通道的层进行特殊处理
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                # 通过idx0中保存的剩余通道的索引来剪枝
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                # 对于不是模块的最后一层卷积层,才可以改变输出的通道数
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions.
            # For these convolutions, we just copy the weights.
            # 对于downsampling的卷积层,不作处理
            m1.weight.data = m0.weight.data.clone()

        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            # 同理处理fc层
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
    #
    # modules = []
    # for module in newmodel.modules():
    #     if isinstance(module, Bottleneck):
    #         if module.downsample is not None:
    #             modules.append(module)
    classifier = nn.Linear(in_features=cfg[-1], out_features=class_num)
    idx0 = np.arange(0,class_num)
    classifier.weight.data = newmodel.fc.weight.data[idx0,:].clone()
    classifier.bias.data = newmodel.fc.bias.data[idx0].clone()
    newmodel.fc = classifier
    print(newmodel)
    torch.save(newmodel, './new_models/resnet50_CIFAR10_pruned_trained_'+str(class_num)+'C.pkl')
#    model = torch.load('./new_models/resnet50_CIFAR10_pruned_trained_'+str(class_num)+'C.pkl')
#    # test(model, './data.cifar100', batchSize=512, class_num=class_num)
#
#    
#    train_model(model=model, epoches=100, batchSize=512, forzen=False,
#                trainSet_path=data_path,
#                model_save_path='./new_trained_models/ResNet56_cifar100_pruned_trained_'+str(class_num)+'C.pkl')
#
#    torch.cuda.empty_cache()
#
#    model = torch.load('./new_trained_models/ResNet56_cifar100_pruned_trained_'+str(class_num)+'C.pkl')
#    res2, acc2 = test(model, data_path, batchSize=512, class_num=class_num)
#
#    if isinstance(model, torch.nn.DataParallel):
#        model = model.module
#
#    # test_latency('resnet101',model)
#
#    x_axis_data = [i for i in range(100)]
#
#    plt.bar(x_axis_data, res1, label='original')
#    plt.bar(x_axis_data, res2, label='pruned')
#
#    plt.xlabel('Label Index')
#    plt.ylabel('Number of Corrects')
#    plt.title('Class:' + str(class_num) + ' original acc:' + str(acc1) + ' pruned acc:' + str(acc2))
#    plt.legend()
#
#    plt.savefig(res)
#    plt.show()
#    plt.close()
#
#    temp = np.array(res2)
#    np.save('./result10/ResNet50_'+str(class_num)+'C.npy', temp)
#
#    res1.clear()
#    res2.clear()
    #

