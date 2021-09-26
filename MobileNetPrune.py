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
from torch.hub import load_state_dict_from_url
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ['CUDA_VISIBLE_DEVICE']='1'

activations = []

dataSet = 'CIFAR10'

if dataSet == 'CIFAR10':
    data_path = './data/'
    mean = [0.4940607, 0.4850613, 0.45037037]
    std = [0.20085774, 0.19870903, 0.20153421]

elif dataSet == 'CIFAR100':
    data_path = './data/'
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

def func(x):
    return 0.4834 + 0.215*cos(x*4.307) + 0.18*sin(x*4.307) + 0.06662*cos(2*x*4.307) - 0.03691*sin(2*x*4.307) + 0.03065*cos(3*x*4.307) - 0.04711*sin(3*x*4.307)

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
#            print(x.size())
            return x + self.conv(x)
        else:
#            print(x.size())
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def activation_hook(self, model, input, output):

        x = input[0]
        IF_RESIDUAL = False

        for name, module in model.named_modules():
            RESIDUAL_FLAG = name.split('.')

            if isinstance(module, InvertedResidual):
                if module.use_res_connect is True:
                    residual = x
                    IF_RESIDUAL = True
                    continue

            if isinstance(module, nn.Conv2d):
                x = module.forward(x)

            elif isinstance(module, nn.BatchNorm2d):
                x = module.forward(x)
                activations.append(x.sum(dim=0))

            if len(RESIDUAL_FLAG) == 2 and IF_RESIDUAL is True:
                x += residual
                IF_RESIDUAL = False
                continue

        return


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def compute_L2_activation(activations):
    L2_activations = []
    for activation in activations:
        activation = nn.LeakyReLU()(activation)
        L2_activations.append(activation.cpu().norm(dim=(1, 2), p=2).cuda())
    return L2_activations

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



def read_Img2(dataSet_path, batchSize):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(dataSet_path, transform=transform, download=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(dataSet_path, transform=transform, download=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=8, drop_last=True)

    counts = []
    target_class = []
    for idx in range(0, 100):
        counts.append(0)
    for idx in range(0, 30):
        target_class.append(idx)
    inputs = []

    for data, label in tqdm.tqdm(data_loder):
        for idx in range(len(label)):
            if label[idx] in target_class and counts[label[idx]] < image_num:
                inputs.append(data[idx].cuda())
                counts[label[idx]] += 1

    imgs = torch.empty(len(inputs), 3, 32, 32).cuda()

    for idx, img in enumerate(inputs):
        imgs[idx] = img

    return imgs




def train_model_all(model, trainSet_path, epoches, batchSize, model_save_path):

    model.cuda()

    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)

    model.train()

    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(trainSet_path, transform=transform, download=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(trainSet_path, transform=transform, download=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=True, num_workers=32, drop_last=False)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    optimizer.zero_grad()

    best_acc = 0
    best_loss = 9999999999

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

        epoch_res, epoch_acc = test(model, data_path, batchSize=512, class_num=10)
        model.train()

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model, model_save_path)
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     torch.save(model, model_save_path)

        scheduler.step()


def train_model(model, trainSet_path, epoches, batchSize, model_save_path, forzen = False):

    model.cuda()

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
#        transforms.RandomCrop(32, padding=4),
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
    optimizer = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.5)
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
        counts.append(0)

#    for idx in range(0, class_num):
        

    for data, label in tqdm.tqdm(data_loder):
        for idx in range(len(label)):
            if label[idx] in target_class and counts[label[idx]] < image_num:
                inputs.append(data[idx].unsqueeze(0).cuda())
                targets.append(label[idx].unsqueeze(0).cuda())
                counts[label[idx]] += 1
   

    file = open('./result11/log'+str(class_num)+'.txt',mode='w')

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

        epoch_res, epoch_acc = test(model, data_path, batchSize=1024, class_num=class_num)
        model.train()
        file.write('epoch: '+ str(epoch) + '\tacc:' + str(epoch_acc)+ '\n')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model, model_save_path)

        scheduler.step()
    file.close()

def test(model, dataSet_path, batchSize, class_num):

    model.cuda()
#    model.to("cuda:1")
#
#    if not isinstance(model, torch.nn.DataParallel):
#        model = nn.DataParallel(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    if dataSet == 'CIFAR100':
        data_set = datasets.CIFAR100(dataSet_path, transform=transform, download=False, train=False)

    elif dataSet == 'CIFAR10':
        data_set = datasets.CIFAR10(dataSet_path, transform=transform, download=False, train=False)

    data_loder = dataloader.DataLoader(data_set, batch_size=batchSize, shuffle=False, num_workers=8, drop_last=False)

    model.eval()
    correct = 0

    classes = []
    for idx in range(model.module.classifier[1].out_features):
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
        class_acc.append(elem / 100)


    total_acc = sum(class_acc[0:class_num])/class_num

    # temp = np.array(class_acc)
    # np.save('./result/MobileNetV2_CIFAR100_all.npy', temp)

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
# ��ȡbn���Ȩ����Ϣ�������bn��



# plt.plot(x, MobileNet_acc_pruned, label='Pruned Model')
# plt.plot(x_axis, MobileNet_acc_pruned, 'o', label='Compression Ratio', color='red')



#model = mobilenet_v2(pretrained=False, num_classes=1000).cuda()
# #
#model.classifier[1] = nn.Linear(in_features=1280, out_features=100)

#model = torch.load('./original_models/MobileNetV2_cifar100.pkl')
#test(model, data_path, batchSize=2048, class_num=100)
#train_model_all(model=model, epoches=500, batchSize=512,
#            model_save_path='./original_models/MobileNetV2_cifar100_Test.pkl',
#            trainSet_path=data_path)
model = torch.load('./original_models/MobileNet_cifar10.pkl', map_location='cpu')
# res1, acc1 = test(model, data_path, batchSize=2048, class_num=10)
test(model, data_path, batchSize=64, class_num=10)
exit(0)

image_num = 1200

for class_num in range(5, 105, 5):

    res = './result11/MobileNetV2_' + str(class_num) + 'C_AutoP.pdf'

    model = torch.load('./original_models/MobileNetV2_cifar100.pkl')

#    print(model)
#    res1, acc1 = test(model, data_path, batchSize=1024, class_num=class_num)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])


    # imgs = read_Img('./mix/', transformer=transform)
    imgs = read_Img2(data_path, 8192)

    img_score = 0.98 * float(get_img_socre(imgs, sampling=class_num*10))
    p = func(class_num / 100) + img_score
    print('pruning rate is', p)
    # imgs = read_Img('./mix/', transformer=transform)
    masks = get_mask(imgs)

    # masks = get_mask(imgs)
    # print(get_img_socre(imgs, sampling=1000))



    # ͳ���ܵ�ͨ����
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
    # # ��������bn��Ԥ��֦����ֵ,���㷽ʽΪ��bn��Ȩ�ش�С���аٷֱ���̭,��̭�ٷ�֮��Ϊ������
    # y, i = torch.sort(bn)
    # thre_index = int(total * 0.1)
    # thre = y[thre_index]



    pruned = 0
    cfg = []
    count = 0
    cfg_mask = []
    # ����������ֵ��bn�����Ԥ��֦,����ļ�֦��û�������Ĳü�ͨ��,���Ǽ򵥵�����Ȩ��Ϊ0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = masks[count]
            if count <= 1:
                mask = mask | True
            count += 1
            # mask = weight_copy.gt(thre).float().cuda()
            # mask.shape[0] #�ǵ�ǰ��ͨ������,torch.sum(mask)���޼�����Ŀ
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            # ����,��mask�������
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            # ��ǰ��ʣ��ͨ����
            cfg.append(int(torch.sum(mask)))
            # ��ǰ���Ӧ��mask����
            cfg_mask.append(mask.clone())
            # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            #     format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned/total

    print('Pre-processing Successful!')
#    for idx in range(len(cfg_mask)):
#        print(int(sum(cfg_mask[idx])), int(cfg[idx]))   

#    for idx in range(len(cfg_mask)):
#        print(len(cfg_mask[idx]), cfg[idx])    

    cfg.insert(0, 3)
    new_model = model

    cfg_idx = 0
    GROUP = False
    for module in new_model.modules():
        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                module.out_channels = cfg[cfg_idx]
                module.in_channels = cfg[cfg_idx]
                module.groups = module.in_channels
                cfg_mask[cfg_idx] = torch.ones(cfg[cfg_idx])	
                cfg[cfg_idx + 1] = cfg[cfg_idx]
#                cfg_mask[cfg_idx + 1] = cfg_mask[cfg_idx]
#                print(module)
                
            else:
                module.groups == 1     
                module.in_channels = cfg[cfg_idx]
                module.out_channels = cfg[cfg_idx + 1]
#                print(module)
                
        
        elif isinstance(module, nn.BatchNorm2d):
            module.num_features = cfg[cfg_idx + 1]
            cfg_idx += 1

            
    print(cfg)
#    for idx in range(len(cfg_mask)):
#        print(int(sum(cfg_mask[idx])), int(cfg[idx + 1]))


        # if isinstance(module, nn.Conv2d):
        #     if module.groups != 1:
        #         module.out_channels = cfg[cfg_idx]
        #         module.in_channels = cfg[cfg_idx]
        #         module.groups = module.in_channels
        #         GROUP = True
        #         print(module)
        #
        #     elif module.groups == 1 and GROUP is False:
        #         module.in_channels = cfg[cfg_idx]
        #         module.out_channels = cfg[cfg_idx + 1]
        #         print(module)
        #
        #     else:
        #         module.in_channels = cfg[cfg_idx - 1]
        #         module.out_channels = cfg[cfg_idx]
        #         GROUP = False
        #         print(module)
        #
        #
        # elif isinstance(module, nn.BatchNorm2d):
        #     if GROUP is False:
        #         module.num_features = cfg[cfg_idx + 1]
        #         cfg_idx += 1
        #
        #     else:
        #         module.num_features = cfg[cfg_idx]
        #         cfg_idx += 1
##        print(cfg_idx, cfg[cfg_idx])
#
#
#
    print(new_model)
##    print
#
#
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    GROUP = False
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            print(layer_id_in_cfg ,end_mask, '\n',start_mask)
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            if GROUP:
                m1.weight.data = data0
                m1.bias.data = data0
                m1.running_mean = data1
                m1.running_var = data1
                nn.init.ones_(m1.weight)
                nn.init.zeros_(m1.bias)

                GROUP = False
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                	end_mask = cfg_mask[layer_id_in_cfg]
                continue

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
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))


            if m0.groups != 1:
                m1.weight.data = torch.ones(m1.in_channels, 1, m1.kernel_size[0], m1.kernel_size[1], requires_grad=True).cuda()
                nn.init.kaiming_normal_(m1.weight, mode='fan_out')
                GROUP = True
                channel_flag = m1.in_channels
                data0 = torch.ones(channel_flag, requires_grad=True).cuda()
                data1 = torch.ones(channel_flag).cuda()
                #m1.groups = m1.in_channels
                print('Groups:In shape: {:d}, Out shape {:d}, Group:{:d}'.format(idx0.size, idx0.size, m1.out_channels))
                continue

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()

        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0.tolist()].clone()

    RESIDUAL = False
    for module in new_model.modules():
        if isinstance(module, InvertedResidual):
            if RESIDUAL is True:
                Conv = module.conv[0][0]

                Conv.in_channels = res_features
                Conv.weight.data = torch.ones(Conv.out_channels, res_features, Conv.kernel_size[0], Conv.kernel_size[1], requires_grad=True).cuda()
                nn.init.kaiming_normal_(Conv.weight, mode='fan_out')
                RESIDUAL = False
                
            if module.use_res_connect is True:
                res_features = module.conv[0][0].in_channels
                Conv = module.conv[-2]
                Conv.out_channels = res_features
                Conv.weight.data = torch.ones(res_features, Conv.in_channels, Conv.kernel_size[0], Conv.kernel_size[1], requires_grad=True).cuda()
                nn.init.kaiming_normal_(Conv.weight, mode='fan_out')

                BN = module.conv[-1]
                data0 = torch.ones(res_features, requires_grad=True).cuda()
                data1 = torch.ones(res_features).cuda()
                BN.num_features = res_features
                BN.weight.data = data0
                BN.bias.data = data0
                BN.running_mean = data1
                BN.running_var = data1
                nn.init.ones_(BN.weight)
                nn.init.zeros_(BN.bias)

                RESIDUAL = True
                
    classifier = nn.Linear(in_features=cfg[-1], out_features=class_num)
    idx0 = np.arange(0,class_num)
    classifier.weight.data = new_model.classifier[1].weight.data[idx0,:].clone()
    classifier.bias.data = new_model.classifier[1].bias.data[idx0].clone()
    new_model.classifier[1] = classifier
    print(new_model)

#
#


#    torch.save(new_model, './new_trained_models/mobilenetv2_CIFAR100_pruned'+str(class_num)+'C.pkl')
    model = torch.load('./new_trained_models/mobilenetv2_CIFAR100_pruned'+str(class_num)+'C.pkl')
    # test(model, './data.cifar100', batchSize=1024)
#    model.cuda()

            

    model = mobilenet_v2(pretrained=True, num_classes=1000).cuda()
# #
    model.classifier[1] = nn.Linear(in_features=1280, out_features=class_num)


    train_model(model=model, epoches=200, batchSize=1024, forzen=False,
                trainSet_path=data_path,
                model_save_path='./new_trained_models/MobileNet_cifar100_pruned_trained'+str(class_num)+'C.pkl')
    model = torch.load('./new_trained_models/MobileNet_cifar100_pruned_trained'+str(class_num)+'C.pkl')
    res2, acc2 = test(model, data_path, batchSize=2048, class_num=class_num)
    #
    #x_axis_data = [i for i in range(100)]

    #plt.bar(x_axis_data, res1, label='oringin')
    #plt.bar(x_axis_data, res2, label='pruned')

    #plt.xlabel('Label Index')
    #plt.ylabel('Number of Corrects')
    #plt.title('Class:' + str(class_num) + ' original acc:' + str(acc1) + ' pruned acc:' + str(acc2))
    #plt.legend()

    #plt.savefig(res)
    #plt.show()
    #plt.close()
    #
    #temp = np.array(res2)
    #np.save('./result11/MobileNetV2_' + str(class_num) + 'C.npy', temp)
    #
    #res1.clear()
    #res2.clear()

