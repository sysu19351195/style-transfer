from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import PIL
import glob
import random
import time
import copy
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from style_transfer_model_way_1 import VGG16, Reset_Generator

test_path = '../examples/test_content'#测试图片路径

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)#全部初始化为均值为0，方差为0.02的正态分布
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.02)#全部初始化为均值为0，方差为0.02的正态分布
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1.0, 0.02)#全部初始化为均值为1，方差为0的正态分布
        m.bias.data.fill_(0)#全部初始化为0

#生成并保存生成效果图
def predicted_imsave(G_net_path):
    namestr = G_net_path.split('.pth')[0].split('net_')[-1]
    path = './predicted_' + namestr
    if not os.path.exists(path):
        os.makedirs(path)
    G_net = Reset_Generator(num_blocks=6)
    checkpoint = torch.load(G_net_path)
    G_net.load_state_dict(checkpoint['model'])
    global test_path
    for index, img_path in enumerate(glob.glob(test_path + '/*.jpg')):
        img_path = img_path.replace('\\', '/')
        content_num = img_path.split('.jpg')[0].split('/')[-1]
        img = Image.open(img_path)
        content_img = transform(img)
        content_img = content_img.unsqueeze(0)
        transfer_img = G_net(content_img)
        transfer_img = transfer_img.squeeze()
        transfer_img = transfer_img / 2 + 0.5
        npimg = transfer_img.detach().numpy()
        x = np.transpose(npimg, (1, 2, 0))
        plt.imsave(path + '/content_' + content_num + '.jpg', x)

def GramMatrix(input):
    b, c, h, w = input.size()
    F = input.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h * w)
    return G

def content_loss(mse_loss, true_content, predicted_content):
    return mse_loss(true_content, predicted_content) * 0.5

def style_loss(mse_loss, true_style, predicted_style):
    Gram_true = GramMatrix(true_style)
    Gram_predicted = GramMatrix(predicted_style)
    Gram_true = Gram_true.squeeze()
    loss = 0.0
    for index in range(Gram_predicted.shape[0]):
        loss += mse_loss(Gram_true, Gram_predicted[index])
    return loss / Gram_predicted.shape[0]

#计算测试集损失
def testing(G_net, style_img, style_mse_loss, content_mse_loss):
    global test_path
    test_glob = glob.glob(test_path + '/*.jpg')
    test_imgs = torch.zeros(len(test_glob), 3, 256, 256)
    for index, img_path in enumerate(test_glob):
        img = Image.open(img_path)
        test_imgs[index] = transform(img)
    _, true_style = my_vgg16(style_img)
    true_content, _ = my_vgg16(test_imgs)
    output_imgs = G_net(test_imgs)
    false_content, false_style = my_vgg16(output_imgs)
    loss_content = 0.0
    for _, key in enumerate(false_content):
        loss_content += content_loss(content_mse_loss, true_content[key], false_content[key])
    loss_style = 0.0
    for _, key in enumerate(false_style):
        loss_style += style_loss(style_mse_loss, true_style[key], false_style[key])
    print('test: style_loss:%0.3f content_loss:%0.3f' % (
       loss_style.item(), loss_content.item()))


def training(my_vgg16, G_net, style_mse_loss, content_mse_loss, G_scheduler, G_optimizer, style_img, train_loader, transform, G_net_path):
    my_vgg16.eval()
    for param in my_vgg16.parameters():
        param.requires_grad = False
    s = time.time()
    _, true_style = my_vgg16(style_img)
    for epoch in range(0,2):
        for index, data in enumerate(train_loader,0):
            content_img, _ = data
            G_optimizer.zero_grad()
            output_img = G_net(content_img)
            false_content, false_style = my_vgg16(output_img)
            true_content, _ = my_vgg16(content_img)
            loss_content = 0.0
            for _, key in enumerate(false_content):
                loss_content += content_loss(content_mse_loss, true_content[key], false_content[key])
            loss_style = 0.0
            for _, key in enumerate(false_style):
                loss_style += style_loss(style_mse_loss, true_style[key], false_style[key])
            G_loss = 4 * loss_style + 5 * loss_content
            G_loss.backward()
            G_optimizer.step()
            print('epoch:%d batches:%d style_loss:%0.3f content_loss:%0.3f' % (
                epoch + 1, index + 1, loss_style.item(), loss_content.item()))
            if (index + 1) % 10 == 0:
                testing(G_net, style_img, style_mse_loss, content_mse_loss)
                e = time.time()
                print('epoch:%d time:%0.2f min' % (epoch + 1, (e - s) / 60))
            if (index + 1) * 12 % 420 == 0:
                epo = G_net_path.split('epoch ')[-1].split('_imgs')[0]
                iters = G_net_path.split('imgs num ')[-1].split('_Glr')[0]
                new_path = G_net_path.replace(iters, str((index + 1) * 12)).replace(epo, str(epoch + 1))
                state = {'model': copy.deepcopy(G_net.state_dict()),
                         'optimizer': copy.deepcopy(G_optimizer.state_dict()),
                         'scheduler': copy.deepcopy(G_scheduler.state_dict())}
                torch.save(state, new_path)
                predicted_imsave(new_path)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 导入预训练好的VGG16
    vgg16 = models.vgg16(pretrained=True)
    my_vgg16 = VGG16()

    model_dict = my_vgg16.state_dict()
    pretrained_dict = vgg16.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    my_vgg16.load_state_dict(model_dict)

    style_path = '../examples/style/0.jpg'#可以改变序号来训练其他风格
    style_img = Image.open(style_path)
    style_img = transform(style_img)
    style_img = style_img.unsqueeze(0)

    content_path = '../examples/train_content'
    train_dataset = datasets.ImageFolder(root='../examples/train_content', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=False, num_workers=1)

    G_net_path = 'G_net_epoch 1_imgs num 4620_Glr 5e-5_style ' + str(style_path.split('.jpg')[0].split('/')[-1]) + '.pth'
    G_net = Reset_Generator(num_blocks=6)
    G_optimizer = optim.Adam([{'params': G_net.parameters(), 'initial_lr': 5e-5}], lr=5e-5, betas=(0.5, 0.999))
    G_scheduler = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[340], gamma=0.5, last_epoch=314)
    G_net.apply(weight_init)

    style_mse_loss = nn.MSELoss(reduction='mean')
    content_mse_loss = nn.MSELoss(reduction='mean')

    training(my_vgg16, G_net, style_mse_loss, content_mse_loss, G_scheduler, G_optimizer, style_img, train_loader, transform, G_net_path)






