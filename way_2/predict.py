import torch
import torchvision
import torchvision.transforms as transforms
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from style_transfer_model_elu import VGG16, Reset_Generator

test_path = '../examples/test_content'

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def predicted_imsave(G_net_path):
    path = './predicted'
    if not os.path.exists(path):
        os.makedirs(path)
    G_net = Reset_Generator(num_blocks=5)
    # checkpoint = torch.load(G_net_path)
    # G_net.load_state_dict(checkpoint['model'])
    G_net.load_state_dict(torch.load(G_net_path))
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

predicted_imsave('model_state_style_0.pth')#可以更改编号为6，来生成6号风格的效果图