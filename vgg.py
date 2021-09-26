import math
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import torch.optim as optim
import tqdm
import argparse
import torch.nn.functional as F
import random
import os
import time
import pylab as plt
import numpy as np
# from latency import test_latency
import torch.nn.utils.prune
import torch.autograd.function
from math import sin, cos
import torchvision
from torch.utils import data as Data

torch.set_printoptions(sci_mode=False)
def func(x):
    return 0.6034 + 0.215 * cos(x * 4.307) + 0.18 * sin(x * 4.307) + 0.06662 * cos(2 * x * 4.307) - 0.03691 * sin(
        2 * x * 4.307) + 0.03065 * cos(3 * x * 4.307) - 0.04711 * sin(3 * x * 4.307)


dataSet = 'CIFAR10'

if dataSet == 'CIFAR10':
    data_path = './data/'
    mean = [0.4940607, 0.4850613, 0.45037037]
    std = [0.20085774, 0.19870903, 0.20153421]

elif dataSet == 'CIFAR100':
    data_path = './data/'
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

__all__ = ['vgg']
activations = []

# vgg结构
defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

train_data = Data.DataLoader(
    torchvision.datasets.CIFAR10("./data", train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                 ]))
                                ,batch_size=4, shuffle=False)


class vgg(nn.Module):
    def __init__(self, dataset='cifar100', depth=11, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.feature = self.make_layers(cfg, True)
        self.activations = []

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            num_classes = 1000

        in_feature = cfg[-1]
        self.classifier = nn.Linear(in_features=in_feature, out_features=num_classes, bias=True)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        # x = nn.AvgPool2d(kernel_size=1, stride=1)(x)
        x = nn.AvgPool2d(2)(x)
        x = torch.flatten(x, start_dim=1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # print(m.kernel_size, m.out_channels)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def activation_hook(self, model, input, output):
        x = input[0]
        for module in model.feature:
            x = module.forward(x)
            if isinstance(module, nn.BatchNorm2d):
                activations.append(x.sum(dim=0))
        return


# 计算CRV 输出维度为 [n, c]
def compute_L2_activation(activations, CT=None):
    L2_activations = []
    for activation in activations:
        activation = nn.LeakyReLU()(activation)
        L2_activations.append(activation.cpu().norm(dim=(1, 2), p=2).cuda())
    if CT is not None:
        temp = np.array(L2_activations)
        np.save('./result9/Activations' + str(CT) + '.npy', temp)
        # print(activation.cpu().norm(dim=(1, 2)).cuda().size())
    return L2_activations


# 计算每层输出CRV的阈值 每层一个阈值 维度为[n]
def sort_L2_activations(L2_activations, percent, thresholds):
    for tensor in L2_activations:
        y, i = torch.sort(tensor)
        total = len(y)
        thre_index = int(total * percent)
        thre = y[thre_index]    # 求阈值
        # thre = y.mean()       # 两种办法
        thresholds.append(thre)
    return thresholds


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# 读取储存文件
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


def get_mask(imgs, pruned_percent):
    # pruned_percent 剪枝率
    imgs_masks = []
    thresholds = []
    one_img_masks = []
    CT = 0
    for img in tqdm.tqdm(imgs, desc="calculate masks:"):

        thresholds.clear()
        activations.clear()
        one_img_masks.clear()

        img = torch.unsqueeze(img, 0)
        handle = model.register_forward_hook(model.activation_hook)
        output = model(img)
        handle.remove()

        # 输出维度为[n, c] n为卷积输出个数
        L2_activations = compute_L2_activation(activations)
        # 输出阈值队列 长度为 n
        thresholds = sort_L2_activations(L2_activations, percent=pruned_percent, thresholds=thresholds)
        CT += 1

        # 计算每一层的mask掩码
        for idx in range(len(thresholds)):
            # 前者大 则将这mask置为 True layer_mask维度为[n, c]
            layer_mask = L2_activations[idx].gt(thresholds[idx]).cuda()
            one_img_masks.append(layer_mask)

        # 总图片的mask掩码 维度为 [b, n, c]
        imgs_masks.append(one_img_masks)

    layer_num = len(one_img_masks)   # n
    img_num = len(imgs_masks)        # b
    # print(layer_num, img_num)
    # for i in imgs_masks:
    #     for j in i:
            # print(j.shape)
    mask = imgs_masks[0]
    for idx1 in range(layer_num):
        for idx2 in range(img_num):
            mask[idx1] |= imgs_masks[idx2][idx1]
    # print("````````````````````````````")

    # for k in mask:
    #     print(k.shape)

    return mask

# p = 0.5
# model = vgg()
# imgs1 = next(iter(train_data))[0]
# a = get_mask(imgs1)
# exit(0)

# 根据保留的类标签 读取微调时候需要的数据 可以加入KL-divergence
def read_Img2(dataSet_path, batchSize, target_class, pics_num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(dataSet_path, transform=transform, download=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(dataSet_path, transform=transform, download=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=1, drop_last=True)

    counts = []
    inputs = []

    # out_features = 总类数量
    for idx in range(0, model.classifier.out_features):
        counts.append(0)

    # # 目标类标签 class_num保留类数量
    # for idx in range(0, class_num):
    #     target_class.append(idx)

    # image_num表示微调时选区的图片数量
    for data, label in tqdm.tqdm(data_loder, desc="loading_imgs: "):
        for idx in range(len(label)):
            if label[idx] in target_class and counts[label[idx]] < pics_num:
                inputs.append(data[idx].cuda())
                counts[label[idx]] += 1

    imgs = torch.empty(len(inputs), 3, 32, 32).cuda()

    for idx, img in enumerate(inputs):
        imgs[idx] = img

    return imgs

# 选择最优数据进行剪枝和测试
def choose_best_data(model, reserved_classes, num_images, dataSet_path,
                     correct_rate, use_KL=False, divide_radio = 1):

    # Correct_rate 选择正确率大于该值的图片
    # use_KL 图片存满后是否使用KL-divergence来选择更好图片
    # divide_radio 当保存图片数量超过 num_images/divide_radio 时
    # 使用随机选择 num_images/divide_radio 张图片，而不是使用全部图片来
    # 计算 KL-divergence 以减少计算量

    # 默认第一张图片KL-divergence为1 （很大的值）

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 使用训练数据
    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(dataSet_path, transform=transform, download=False, train=True)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(dataSet_path, transform=transform, download=False, train=True)

    data_loder = dataloader.DataLoader(data_set, batch_size=1024, shuffle=True)

    image_data_list = []   # 保存图片数据
    image_lable_list = []  # 保存图片标签
    image_counts = []      # 保存每个已存图片数量

    if use_KL:             # 保存每个图片Kc 即 KL-divergence score # 值
        image_Kc_list = np.zeros([len(reserved_classes), num_images])

    for _ in range(len(reserved_classes)):
        image_counts.append(0)
        image_data_list.append([])
        image_lable_list.append([])



    with torch.no_grad():

        for x, lable in tqdm.tqdm(data_loder, desc="loading best data:"):

            x, lable = x.cuda(), lable.cuda()

            out = F.softmax(model(x), dim=1)
            # 计算置信度
            out_a, out_l = torch.max(out, dim=1)

            # print(lable.eq(out_l).sum().item())
            # kl_divergence = F.kl_div(x[0].softmax(dim=-1).log(), x[1].softmax(dim=-1), reduction='mean')


            # 图片加入条件 1.在要保留类的列表中 2.模型预测此类正确
            #            3.该类保存图片未满  4.模型给出的图片准确率大于阈值correct_rate
            #            5.如果加入KL-divergence 则还需计算KL-divergence
            for idx_ in range(len(lable)):
                class_idx = lable[idx_] #  class_idx : 当前循环下图片的类id
                if class_idx in reserved_classes and out_l[idx_] == lable[idx_] \
                    and image_counts[class_idx] < num_images and out_a[idx_] > correct_rate:

                    # 如果使用KL-divergence 则计算每张图片的KL值
                    if use_KL:
                        # 如果是第一张图片 则将其Kc值置为1
                        if image_counts[class_idx] == 0:
                            image_Kc_list[class_idx][0] = 1
                        # 不是第一张图
                        else:
                            KL_all = 0
                            # 小于划分阈值 则全部计算
                            if image_counts[class_idx] < num_images/divide_radio:
                                for image in image_data_list[class_idx]:
                                    KL_all += F.kl_div(x[idx_].softmax(dim=-1).log(), image.softmax(dim=-1), reduction='mean')
                                Kc = KL_all/image_counts[class_idx]

                            # 大于划分阈值 则随机选择计算
                            else:
                                for _ in range(int(num_images/divide_radio)):
                                    random_i = random.randint(0, image_counts[class_idx])
                                    # x[idx_]当前图片 image_data_list[class_idx][random_i]已存图片随机选择一张
                                    KL_all += F.kl_div(x[idx_].softmax(dim=-1).log(),
                                                       image_data_list[class_idx][random_i].softmax(dim=-1),
                                                       reduction='mean')
                                Kc = KL_all/(num_images/divide_radio)
                            # 储存当前图片的Kc值
                            image_Kc_list[class_idx][image_counts[class_idx]] = Kc

                    image_data_list[class_idx].append(x[idx_])
                    image_lable_list[class_idx].append(lable[idx_])
                    image_counts[class_idx] += 1


def train_model_all(model, trainSet_path, epoches, batchSize, model_save_path):

    model.cuda()

    # if not isinstance(model, torch.nn.DataParallel):
    #     model = nn.DataParallel(model)

    model.train()

    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),      # 随机裁剪
        transforms.RandomHorizontalFlip(),           # 随机左右翻转
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(trainSet_path, transform=transform, download=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(trainSet_path, transform=transform, download=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=True, num_workers=1, drop_last=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    optimizer.zero_grad()

    best_acc = 0
    for epoch in range(epoches):
        epoch_loss = 0
        idx = 0
        for data, label in tqdm.tqdm(data_loder, desc="training:", leave=True, ):
            data = data.cuda()
            label = label.cuda()

            output = model(data)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            if idx != 0 and idx % 100 == 0:
                print('epoch:', epoch, '\tloss:', loss.item())

            idx += 1

        epoch_res, epoch_acc = test(model, data_path, batchSize=128, class_num=10)

        model.train()

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model, model_save_path)

        # 学习率衰减
        scheduler.step()

# 大概是微调训练
def train_model(model, trainSet_path, epoches, batchSize, model_save_path, forzen=False):
    model.cuda()

    # if not isinstance(model, torch.nn.DataParallel):
    #     model = nn.DataParallel(model)

    model.train()

    if forzen == True:
        for param in model.parameters():
            param.requires_grad = False

        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad = True
                module.bias.requires_grad = True

    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(trainSet_path, transform=transform, download=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(trainSet_path, transform=transform, download=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=1, drop_last=False)

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

    for idx in range(0, model.module.classifier.out_features):
        counts.append(0)

    # 选取微调图片 数据集和标签
    for data, label in tqdm.tqdm(data_loder):
        for idx in range(len(label)):
            if label[idx] in target_class and counts[label[idx]] < image_num:
                inputs.append(data[idx].unsqueeze(0).cuda())
                targets.append(label[idx].unsqueeze(0).cuda())
                counts[label[idx]] += 1

    file = open('./result9/log' + str(class_num) + '.txt', mode='w')

    for epoch in range(epoches):
        epoch_loss = 0

        for idx in range(0, len(inputs), batchSize):

            if idx + batchSize > len(inputs):
                # break
                input = inputs[idx:len(inputs) - 1]
                target = targets[idx:len(inputs) - 1]
                data = torch.empty(len(inputs) - idx - 1, 3, 32, 32).cuda()
                label = torch.empty(len(inputs) - idx - 1).long().cuda()

            else:
                input = inputs[idx:idx + batchSize]
                target = targets[idx:idx + batchSize]
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
        file.write('epoch: ' + str(epoch) + '\tacc:' + str(epoch_acc) + '\n')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model, model_save_path)

        scheduler.step()
    file.close()


def test(model, dataSet_path, batchSize, class_num):
    model.cuda()

    # if not isinstance(model, torch.nn.DataParallel):
    #     model = nn.DataParallel(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(dataSet_path, transform=transform, download=False, train=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(dataSet_path, transform=transform, download=False, train=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=1, drop_last=False)

    model.eval()
    correct = 0

    classes = []
    # out_features为输出类别数
    for idx in range(model.classifier.out_features):
        classes.append(0)

    times = 0
    with torch.no_grad():
        idx = 0
        for data, label in tqdm.tqdm(data_loder, desc="testing:"):
            input = data.cuda()
            target = label.cuda()

            # start = time.time()
            output = model(input)
            pred = torch.argmax(output, 1)

            # end = time.time()
            # times += end - start

            for idx in range(len(target)):
                if pred[idx] == target[idx]:
                    classes[target[idx]] += 1
            correct += (pred == target).sum()

            idx += 1

    class_acc = []

    for elem in classes:
        class_acc.append(elem / 1000)

    total_acc = sum(class_acc[0:class_num]) / class_num
    # total_acc = float(correct / 10000)
    # temp = np.array(class_acc)
    # np.save('./result/VGG13_CIFAR100_all.npy', temp)

    print('\n', 'each class corrects:', classes, '\n',
          'each class accuracy:', class_acc, '\n',
          'total accuracy:', total_acc)

    # classes 每个类的正确率    total_acc 保留两位小数
    return classes, round(total_acc, 2)


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


if __name__ == '__main__':

    # Prune settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=16,
                        help='depth of the vgg')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='none', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--save', default='none', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # model = vgg(dataset=args.dataset, depth=args.depth)

    model = torch.load('./models/VGG16_cifar10_Transfer_5.pkl').cuda()
    # print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # test(model, data_path, batchSize=1024, class_num=10)

    choose_best_data(model=model, dataSet_path="./data", reserved_classes=[], num_images=10, correct_rate=0.8)
    exit(0)

    # train_model_all(model=model, epoches=50, batchSize=1024,
    #                model_save_path='./models/VGG16_cifar10_Transfer_5.pkl',
    #                trainSet_path=data_path)

    # model = torch.load('./original_models/VGG16_cifar10.pkl').cuda()
    # test(model, data_path, batchSize=1024, class_num=10)
    # exit(0)
    image_num = 10
    class_num = 3
    classes = []

    for _ in range(5):
        cc = []
        for _ in range(class_num):
            k = random.randint(0, 10)
            cc.append(k)
        classes.append(cc)


    for reserved_classes in classes:
#
        class_num = len(reserved_classes)
#        res = './result9/VGG16_' + str(class_num) + 'C_AutoP.pdf'
#
#        model = torch.load('./original_models/VGG16_cifar10.pkl').cuda()
#        res1, acc1 = test(model, data_path, batchSize=512, class_num=class_num)
#
#
#        if isinstance(model, torch.nn.DataParallel):
#            model = model.module
#
#        # test_latency('vgg19',model)
#
        # 读取剪枝用到的数据
        imgs = read_Img2(data_path, batchSize=512, target_class=reserved_classes, pics_num=image_num)
#
        print(imgs.size())

        img_score = 1.23 * float(get_img_socre(imgs, sampling=class_num * 10))

        # 剪枝率
        pruned_percent = func(class_num / 10) + img_score
        print('pruning rate is', pruned_percent)
#
        # imgs = read_Img('./mix/', transformer=transform)
        masks = get_mask(imgs, pruned_percent)
#
        for i in masks:
            print(i.size())

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#
#
#
#        # if args.cuda:

#        #     model.cuda()
#
#
#
#        total = 0
#        for m in model.modules():
#            if isinstance(m, nn.BatchNorm2d):
#                total += m.weight.data.shape[0]
#
#        # bn = torch.zeros(total)
#        # index = 0
#        # for m in model.modules():
#        #     if isinstance(m, nn.BatchNorm2d):
#        #         size = m.weight.data.shape[0]
#        #         bn[index:(index + size)] = m.weight.data.abs().clone()
#        #         index += size
#        #
#        # y, i = torch.sort(bn)
#        # thre_index = int(total * args.percent)
#        # thre = y[thre_index]
#
#
#
        pruned = 0     # 这变量好像没用 ？？？？？？？？？
        count = 0      # 层数标记
        cfg = []       # 新模型的层参数 如 [64,64,64,'M',128,128.....]
        cfg_mask = []  # 记录型模型的mask 主要是前两层mask与masks有改变
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):

                # weight_copy = m.weight.data.abs().clone()
                # 第 count 层剪枝的mask
                mask = masks[count]
                # 前两层不剪枝
                if count <= 1:
                    mask = mask | True

                count += 1
                # 这变量好像没用 ？？？？？？？？？
                pruned = pruned + mask.shape[0] - torch.sum(mask)

                # m.weight.data[mask] = 0
                # 权重参数乘以mask True保留原值，False位置置0
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)

                # 统计mask该层保留的通道数量
                cfg.append(int(torch.sum(mask)))
                # 记录新模型的mask
                cfg_mask.append(mask.clone())
                print(mask.shape, mask.shape[0])
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(k, mask.shape[0], int(torch.sum(mask))))

            elif isinstance(m, nn.MaxPool2d):
                cfg.append('M')

        # ？？？？？？？？？
        pruned_ratio = pruned / total

        print()
        print('~~~~~~~~~~预剪枝完成，生成新模型的cfg~~~~~~~~~~')

        # 剪枝后，创建新模型，并进行新模型参数复制
        newmodel = vgg(dataset='cifar10', cfg=cfg)
        if args.cuda:
            newmodel.cuda()

        model.eval()
        newmodel.eval()
        num_parameters = sum([param.nelement() for param in newmodel.parameters()])

        savepath = os.path.join(args.save, "prune.txt")
        with open(savepath, "w") as fp:
            fp.write("Configuration: \n" + str(cfg) + "\n")
            fp.write("Number of parameters: \n" + str(num_parameters / 1e6) + "M \n")
            # fp.write("Test accuracy: \n"+str(acc))

        # for i in masks:
        #     print(i.size())
        #
        # print()
        # for i in cfg_mask:
        #     print(i.size())

        for i in range(2):
            print()

        print('~~~~~~~~~~~~~开始复制参数~~~~~~~~~~~~~~~~~~~~')

        layer_id_in_cfg = 0                     #
        start_mask = torch.ones(3)              #
        end_mask = cfg_mask[layer_id_in_cfg]    #

        layer_idx = 0

        # 复制参数
        for idx, [m0, m1] in enumerate(zip(model.modules(), newmodel.modules())):
            # print(idx, "model:   ", m0)
            # BatchNorm2d
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()

                # 实现batch Normal数据复制
                m1_weight = m0.weight.data.clone()[cfg_mask[layer_idx]]
                m1_bias = m0.bias.data.clone()[cfg_mask[layer_idx]]
                m1_running_mean = m0.running_mean.clone()[cfg_mask[layer_idx]]
                m1_running_var = m0.running_var.clone()[cfg_mask[layer_idx]]
                layer_idx += 1
                if not (m1.weight.data.equal(m1_weight)
                        and m1.bias.data.equal(m1_bias)
                        and m1.running_mean.equal(m1_running_mean)
                        and m1.running_var.equal(m1_running_var)):
                    print("BatchNormal copy not equall")
                    exit(0)

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]

            # 卷积
            elif isinstance(m0, nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()

                # 实现Conv2 数据复制
                out_mask = cfg_mask[layer_idx]
                if layer_idx > 0:
                    in_mask = cfg_mask[layer_idx-1]
                    w1_weight = m0.weight.data.clone()[:, in_mask, :, :]
                    w1_weight = w1_weight.clone()[out_mask, :, :, :]
                else:
                    w1_weight = m0.weight.data.clone()[out_mask, :, :, :]
                if not m1.weight.data.equal(w1_weight):
                    print("Conv2 copy not equall")
                    print(m1.weight.data.size())
                    print(w1_weight.size())
                    exit(0)

            # 全连接
            # elif isinstance(m0, nn.Linear):
            #     idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            #     if idx0.size == 1:
            #         idx0 = np.resize(idx0, (1,))
            #     m1.weight.data = m0.weight.data[:, idx0.tolist()].clone()
            #     m1.bias.data = m0.bias.data.clone()

        newmodel.classifier = nn.Linear(in_features=cfg[-1], out_features=class_num)
        classifier = nn.Linear(in_features=cfg[-1], out_features=class_num)
        idx0 = np.arange(0, class_num)
        classifier.weight.data = newmodel.classifier.weight.data[idx0,:].clone()
        classifier.bias.data = newmodel.classifier.bias.data[idx0].clone()
        newmodel.classifier = classifier
        # print(newmodel)
#
#        torch.save(newmodel, './new_models/vgg16_CIFAR10_pruned'+str(class_num)+'C.pkl')
#         model = torch.load('./new_models/vgg16_CIFAR10_pruned'+str(class_num)+'C.pkl').cuda()
        test(model, './data/', batchSize=1024, class_num=10)

     #   train_model(model=model, epoches=100, batchSize=512, forzen=False,
      #              trainSet_path=data_path,model_save_path='./new_trained_models/vgg16_CIFAR10_pruned_trained_'+str(class_num)+'C.pkl')

       # torch.cuda.empty_cache()
       #
       # model = torch.load('./new_trained_models/vgg16_CIFAR100_pruned_trained_'+str(class_num)+'C.pkl').cuda()
       # res2, acc2 = test(model, data_path, batchSize=2048, class_num=class_num)
       #
       #
       # if isinstance(model, torch.nn.DataParallel):
       #     model = model.module
       #
       # # test_latency('vgg19', model)
       #
       # x_axis_data = [i for i in range(class_num)]
       #
       # plt.bar(x_axis_data, res1[0:class_num], label='oringin')
       # plt.bar(x_axis_data, res2[0:class_num], label='pruned')
       #
       # plt.xlabel('Label Index')
       # plt.ylabel('Number of Corrects')
       # plt.title('Class:' + str(class_num) + ' original acc:' + str(acc1) + ' pruned acc:' + str(acc2))
       # plt.legend()
       #
       # plt.savefig(res)
       # plt.show()
       # plt.close()
       #
       # temp = np.array(res2)
       # np.save('./result9/VGG16_' + str(class_num) + 'C.npy', temp)
       #
       # res1.clear()
       # res2.clear()
