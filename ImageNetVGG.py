from torchvision.models import vgg16_bn
import torch
import os
import numpy as np
import torch
import torch.nn as nn
import time
import pylab as plt
from math import sin, cos
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import torch.optim as optim
import tqdm
import torch.nn.functional as F
import random
from PIL import Image
import torch
from latency import test_latency
import torch.nn as nn
from torch.hub import load_state_dict_from_url

image_num = 200
target_class = [15, 77, 176, 462, 296]

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def activation_hook(self, model, input, output):
        x = input[0]
        for module in model.features:
            x = module.forward(x)
            if isinstance(module, nn.BatchNorm2d):
                activations.append(x.sum(dim=0))
        return


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


activations = []




def func(x):
    return 0.5234 + 0.215*cos(x*4.307) + 0.18*sin(x*4.307) + 0.06662*cos(2*x*4.307) - 0.03691*sin(2*x*4.307) + 0.03065*cos(3*x*4.307) - 0.04711*sin(3*x*4.307)

dataSet = 'ImageNet'

if dataSet == 'CIFAR10':
    data_path = './data.cifar10'
elif dataSet == 'CIFAR100':
    data_path = './data.cifar100'
else:
    data_path = './Imagenet2012/'


def compute_L2_activation(activations):
    L2_activations = []
    for activation in activations:
        L2_activations.append(activation.cpu().norm(dim=(1, 2)).cuda())
    return L2_activations

def sort_L2_activations(L2_activations, percent, thresholds):
    for tensor in L2_activations:
        y, i = torch.sort(tensor)
        total = len(y)
        thre_index = int(total * percent)
        thre = y[thre_index]
        thresholds.append(thre)
    return thresholds

def read_Img2(dataSet_path, batchSize):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(dataSet_path, transform=transform, download=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(dataSet_path, transform=transform, download=False)

    else:
        data_set = datasets.ImageNet(dataSet_path, transform=transform)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=32, drop_last=True)

    counts = []
    target_class = []
    inputs = []
    for idx in range(0, class_num):
        counts.append(0)

    for idx in range(0, class_num):
        target_class.append(idx)


    for data, label in tqdm.tqdm(data_loder):
        for idx in range(len(label)):
            if label[idx] in target_class and counts[label[idx]] < image_num:
                inputs.append(data[idx].unsqueeze(0).cuda())
                counts[label[idx]] += 1
        if sum(counts) == class_num * image_num:
            break

    imgs = torch.empty(len(inputs), 3, 224, 224).cuda()

    for idx, img in enumerate(inputs):
        imgs[idx] = img

    return imgs

def get_mask(imgs):

    model.eval()
    imgs_masks = []
    thresholds = []

    one_img_masks = []
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

def train_model(model, trainSet_path, epoches, batchSize, model_save_path, forzen = False):

    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)

    model.cuda()
    model.train()

    if forzen == True:
        for param in model.parameters():
            param.requires_grad = False

        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad = True
                module.bias.requires_grad = True


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    # if dataSet == 'CIFAR100':
    #     data_set = datasets.CIFAR100(trainSet_path, transform=transform, download=False)
    #
    # elif dataSet == 'CIFAR10':
    #     data_set = datasets.CIFAR10(trainSet_path, transform=transform, download=False)
    #
    # else:
    #     data_set = datasets.ImageNet(trainSet_path, transform=transform)
    data_set = datasets.ImageFolder('./mixes/mix'+str(class_num)+'C', transform=transform)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=True, num_workers=32, drop_last=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.45)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.65)

    optimizer.zero_grad()

    best_loss = 999999999
    inputs = []
    targets = []
    target_class = []
    counts = []

    # for idx in range(0, class_num):
    #     target_class.append(idx)

    for idx in range(0, class_num):
        counts.append(0)


    # for data, label in tqdm.tqdm(data_loder):
    #     for idx in range(len(label)):
    #         if label[idx] in target_class and counts[label[idx]] < image_num:
    #             inputs.append(data[idx].unsqueeze(0).cuda())
    #             targets.append(label[idx].unsqueeze(0).cuda())
    #             counts[label[idx]] += 1
    #     if sum(counts) == class_num * image_num:
    #         break

    best_acc = 0
    for epoch in range(epoches):
        epoch_loss = 0
        for ii, (data, label) in tqdm.tqdm(enumerate(data_loder)):

            output = model(data.cuda())
            loss = loss_func(output, label.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            if ii % 25 == 0:
                print('epoch:', epoch, '\tloss:', loss.item(), '\nres:', torch.argmax(output, 1))

        epoch_res, epoch_acc = test(model, data_path, batchSize=64, class_num=class_num)
        model.train()

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model, model_save_path)

        scheduler.step()


    # for epoch in range(epoches):
    #     epoch_loss = 0
    #
    #     for idx in range(0, len(inputs), batchSize):
    #
    #         if idx + batchSize > len(inputs):
    #             # break
    #             input = inputs[idx:len(inputs)-1]
    #             target = targets[idx:len(inputs)-1]
    #             data = torch.empty(len(inputs)-idx-1, 3, 224, 224).cuda()
    #             label = torch.empty(len(inputs)-idx-1).long().cuda()
    #
    #         else:
    #             input = inputs[idx:idx+batchSize]
    #             target = targets[idx:idx+batchSize]
    #             data = torch.empty(batchSize, 3, 224, 224).cuda()
    #             label = torch.empty(batchSize).long().cuda()
    #
    #         for ii, (elem1, elem2) in enumerate(zip(input, target)):
    #             data[ii] = elem1
    #             label[ii] = elem2
    #
    #         output = model(data)
    #         loss = loss_func(output, label)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         epoch_loss += loss
    #
    #         if idx % 100 == 0:
    #             print('epoch:', epoch, '\tloss:', loss.item())
    #
    #     if epoch_loss < best_loss:
    #         best_loss = epoch_loss
    #         torch.save(model, model_save_path)
    #
    #     scheduler.step()




def test(model, dataSet_path, batchSize, class_num):


    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)

    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # if dataSet == 'CIFAR100':
    #     data_set = datasets.CIFAR100(dataSet_path, transform=transform, download=False, train=False)
    #
    # elif dataSet == 'CIFAR10':
    #     data_set = datasets.CIFAR10(dataSet_path, transform=transform, download=False, train=False)
    #
    # else:
    #     data_set = datasets.ImageNet(dataSet_path, transform=transform, split='val')
    data_set = datasets.ImageFolder('./mixes/mix'+str(class_num)+'CT/', transform=transform)
    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=32, drop_last=True)

    correct = 0

    classes = []
    for idx in range(class_num):
        classes.append(0)

    times = 0
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

    class_acc = []
    for elem in classes:
        class_acc.append(elem / 50)


    total_acc = sum(class_acc[0:class_num])/class_num

    # temp = np.array(class_acc)
    # np.save('./result6/all.npy', temp)

    print('\n', 'each class corrects:', classes, '\n',
          'each class accuracy:', class_acc, '\n',
          'time:', times, '\n',
          'total accuracy:', total_acc)

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

# model = vgg16_bn(pretrained=False)
#
# model.load_state_dict(torch.load('./original_models/vgg16_bn-6c64b313.pth'))


# model = vgg16_bn(pretrained=False).cuda()
# model.load_state_dict(torch.load('./original_models/vgg16_bn-6c64b313.pth'))
# test_latency('VGG13', model)
# model = nn.DataParallel(model)
# for idx in range(0, 100):
#     inpt = torch.randn(64, 3, 224, 224).cuda()
#     oupt = model(inpt)
#     print(idx)
plt.style.use('seaborn-paper')
target_class = [15, 77, 176, 296, 462]
# res1, acc_all = test(model, data_path, batchSize=64, class_num=1000)
res_all = np.load('./result6/all.npy')
# # print(len(res1))
att = np.load('./result5/Time.npy')
acc = []
for e in target_class:
    acc.append(res_all[e])

acc_all = []
for idx in range(1, len(acc)+1):
    acc_all.append(round(sum(acc[0:idx])/idx, 2))

x = np.arange(5)
x_axis = ['1', '2', '3', '4', '5']
acc_pruned = [1, 0.82, 0.8, 0.78, 0.72]
acc_other_x = [1, 2, 3, 4]
acc_other_y = [0.69, 0.69, 0.68, 0.68]
r_other = [0.76, 0.74, 0.72, 0.7]

plt.plot(x_axis, acc_all, label='Original Model')
plt.plot(x_axis, acc_pruned, label='OCAP')
plt.plot(acc_other_x, acc_other_y, label='CAP\'NN')
plt.plot(x_axis, acc_pruned, 'o', label='Compression Ratio', color='red')
plt.plot(acc_other_x, acc_other_y, 'o', color='red')

for x, y, r in zip(acc_other_x, acc_other_y, r_other):
    plt.text(x, y, r, color='black', fontsize=10)


for idx in range(len(x_axis)):
    if idx == 4:
        plt.text(x_axis[idx], acc_pruned[idx], 0.99, color='black', fontsize=10, horizontalalignment='right',verticalalignment='top' )
    else:
        plt.text(x_axis[idx], acc_pruned[idx], 0.99, color='black', fontsize=10, horizontalalignment='left', verticalalignment='bottom')

plt.tick_params(labelsize=10.5)
plt.xlabel('Number of Retained Classes', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.legend(prop=dict(size='10'))

plt.savefig('./result6/res.pdf')
plt.show()
plt.close()

exit(0)
# model = torch.load('./new_trained_models/vgg16_ImageNet_pruned_trained_5C.pkl').cuda()
# test_latency('VGG13', model)
# exit(0)
T = []
for class_num in range(2, 6, 1):

    res1 = []
    for idx in range(class_num):
        res1.append(res_all[target_class[idx]])

    acc1 = sum(res1)/class_num

    model = vgg16_bn(pretrained=False).cuda()
    model.load_state_dict(torch.load('./original_models/vgg16_bn-6c64b313.pth'))

    # res1, acc1 = test(model, data_path, batchSize=512, class_num=class_num)
    res = './result5/VGG16_' + str(class_num) + 'C_AutoP.pdf'

    # acc1 = round(sum(res1[0:class_num])/(class_num*50), 2)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    start = time.time()
    # imgs = read_Img2(data_path, 8192)
    imgs = read_Img('./mixes/mix'+str(class_num), transformer=transform, size_H=224, size_W=224)
    img_score = 1.23 * float(get_img_socre(imgs, sampling=class_num*10))
    p = func(class_num / 1000)
    print('pruning rate is', p)
    # imgs = read_Img('./mix/', transformer=transform)
    masks = get_mask(imgs)

    # masks = get_mask(imgs)
    # print(get_img_socre(imgs, sampling=1000))


    # 统计总的通道数
    total = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    #
    # bn = torch.zeros(total)
    # index = 0
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         size = m.weight.data.shape[0]
    #         bn[index:(index+size)] = m.weight.data.abs().clone()
    #         index += size
    #
    # # 计算用于bn层预剪枝的阈值,计算方式为按bn层权重大小进行百分比淘汰,淘汰百分之几为超参数
    # y, i = torch.sort(bn)
    # thre_index = int(total * 0.1)
    # thre = y[thre_index]



    pruned = 0
    cfg = []
    count = 0
    cfg_mask = []
    # 根据上述阈值对bn层进行预剪枝,这里的剪枝并没有真正的裁剪通道,而是简单的设置权重为0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = masks[count]
            if count <= 5:
                mask = mask | True
            count += 1
            # mask = weight_copy.gt(thre).float().cuda()
            # mask.shape[0] #是当前层通道总数,torch.sum(mask)是修剪的数目
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

    cfg.insert(0, 3)

    new_model = model


    for c in cfg:
        if c == 'M':
            cfg.remove(c)

    cfg_idx = 0
    for module in new_model.modules():

        if isinstance(module, nn.Conv2d):
            module.in_channels = cfg[cfg_idx]
            module.out_channels = cfg[cfg_idx + 1]

        elif isinstance(module, nn.BatchNorm2d):
            module.num_features = cfg[cfg_idx + 1]
            cfg_idx += 1

        elif isinstance(module, nn.Linear):
            module.in_features = cfg[cfg_idx]*49
            break

    # new_model.classifier =
    # print(new_model)

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]

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
            m1.bias.data = m0.bias.data[idx1.tolist()]

        # elif isinstance(m0, nn.Linear):
        #     idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        #     if idx0.size == 1:
        #         idx0 = np.resize(idx0, (1,))
        #     idxn = []
        #     for n in idx0:
        #         for idx in range((n-1)*49, n*49):
        #             idxn.append(idx)
        #
        #     m1.weight.data = m0.weight.data[:, idxn].clone()
        #     m1.bias.data = m0.bias.data.clone()
        #     break

    new_model.avgpool = nn.AvgPool2d(7)
    new_model.classifier = nn.Linear(in_features=cfg[-1], out_features=class_num)
    # model = nn.ModuleList([])
    # for module in new_model.modules():
    #     if isinstance(module, nn.Linear) and LinearCount < 1:
    #         model.append(module)
    # modules = []
    # for module in new_model.named_modules():
    #     modules.append(module)
    #
    # new_model = nn.DataParallel(new_model)
    # for idx in range(0, 100):
    #     inpt = torch.randn(64, 3, 224, 224).cuda()
    #     oupt = model(inpt)
    #     print(idx)


    torch.save(new_model, './new_models/vgg16_ImageNet_pruned.pkl')
    model = torch.load('./new_models/vgg16_ImageNet_pruned.pkl').cuda()
    # modules = []
    # for module in model.children():
    #     print(module)
    #     modules.append(module)


    # test(model, './data.cifar100/', batchSize=2048)
    train_model(model=model, epoches=50, batchSize=64, forzen=False,
                trainSet_path=data_path,
                model_save_path='./new_trained_models/vgg16_ImageNet_pruned_trained_'+str(class_num)+'C.pkl')
    end = time.time() - start
    T.append(round(end, 2))
    torch.cuda.empty_cache()

    model = torch.load('./new_trained_models/vgg16_ImageNet_pruned_trained_'+str(class_num)+'C.pkl').cuda()
    res2, acc2 = test(model, data_path, batchSize=64, class_num=class_num)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # test_latency('vgg19', model)

    x_axis_data = [i for i in range(class_num)]

    plt.bar(x_axis_data, res1, label='oringin')
    plt.bar(x_axis_data, res2, label='pruned')

    plt.xlabel('Label Index')
    plt.ylabel('Number of Corrects')
    plt.title('Class:' + str(class_num) + ' original acc:' + str(acc1) + ' pruned acc:' + str(acc2))
    plt.legend()

    plt.savefig(res)
    plt.show()
    plt.close()

    temp = np.array(res2)
    np.save('./result5/VGG16_' + str(class_num) + 'C.npy', temp)

    res1.clear()
    res2.clear()


temp = np.array(T)
np.save('./result5/Time.npy', temp)