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
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from style_transfer_model_baseline import VGG16

#展示更新图像数据后的图片
def my_imshow(img_tensor):
    img = img_tensor.clone()
    img = img.squeeze()
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    x = np.transpose(npimg, (1, 2, 0))
    plt.imshow(x)
    plt.show()

def GramMatrix(input):
    b, c, h, w = input.size()
    F = input.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h * w)
    return G

def content_loss(mse_loss, true_content, predicted_content):
    return mse_loss(true_content, predicted_content) / 2

def style_loss(mse_loss, true_style, predicted_style):
    Gram_true = GramMatrix(true_style)
    Gram_predicted = GramMatrix(predicted_style)
    return mse_loss(Gram_true, Gram_predicted)

def training(my_vgg16, mse_loss, optimizer, style_img, content_img, parameter):
    for param in my_vgg16.parameters():
        param.requires_grad = False
    epoch = [0]
    while epoch[0] < 1000:
        def closure():
            optimizer.zero_grad()
            out_content, out_style = my_vgg16(parameter)
            true_content, _ = my_vgg16(content_img)
            _, true_style = my_vgg16(style_img)
            loss_content = content_loss(mse_loss, true_content, out_content)
            loss_style = 0.0
            for _, key in enumerate(out_style):
                loss_style += style_loss(mse_loss, true_style[key], out_style[key])
            total_loss = loss_style + loss_content
            total_loss.backward()
            epoch[0] += 1
            print('epoch:%d content loss:%0.3f style loss:%0.3f' % (epoch[0], loss_content, loss_style))
            return total_loss
        optimizer.step(closure)
        parameter.data.clamp_(-1, 1)
    torch.save(parameter.data, 'style_0.pt')

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #导入训练好的VGG16模型
    vgg16 = models.vgg16(pretrained=True)
    my_vgg16 = VGG16()

    model_dict = my_vgg16.state_dict()
    pretrained_dict = vgg16.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    my_vgg16.load_state_dict(model_dict)

    content_img = Image.open('../examples/test_content/test_0.jpg')#可以改变图像的编号选择要训练的内容图
    content_img = transform(content_img)
    content_img = content_img.unsqueeze(0)

    style_img = Image.open('../examples/style/0.jpg')#可以改变图像的编号选择要训练的风格
    style_img = transform(style_img)
    style_img = style_img.unsqueeze(0)

    input_img = content_img.clone()
    parameter = torch.nn.Parameter(input_img.data)

    mse_loss = nn.MSELoss(reduction='mean')
    optimizer = optim.LBFGS([parameter])

    training(my_vgg16, mse_loss, optimizer, style_img, content_img, parameter)

    y = torch.load('style_0.pt')
    my_imshow(y)





