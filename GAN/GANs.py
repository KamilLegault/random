# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 22:48:56 2018

@author: Moneim
"""

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt

from torchvision.utils import make_grid


import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
import argparse
import gzip
import zipfile

# root path depends on your computer
os.chdir("/Users/Moneim/Desktop/Umontreal/ML Master/IFT6135/Assignment4/data")
print(os.getcwd())


#%%
'''
parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default='datasets', 
                        help='directory to save the dataset')
args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
'''
#%%
def data_proc(resize_size=64):
    
    filename  = 'img_align_celeba.zip'
    filepath = os.path.join( filename)
    zip_ref = zipfile.ZipFile(filepath, 'r')
    zip_ref.extractall()
    zip_ref.close()

    root = 'img_align_celeba/'
    save_root = 'resized_celebA/'
    resize_size = 64

    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if not os.path.isdir(save_root + 'celebA'):
        os.mkdir(save_root + 'celebA')
    img_list = os.listdir(root)

    # ten_percent = len(img_list) // 10

    for i in range(len(img_list)):
        img = plt.imread(root + img_list[i])
        img = imresize(img, (resize_size, resize_size))
        plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

        if (i % 1000) == 0:
            print('%d images complete' % i)




#%%
latent_dim = 100
n_gen = 32
n_disc = 32
n_epochs = 25
lr = 0.0002
beta = 0.5
use_cuda = True
num_gpu = 1
path_gen = ""
path_disc = ""
path_output = "output"
nc = 3
try:
    os.makedirs(path_output)
except OSError:
    pass

#%%
def dataloader(data_path="resized_celebA/",img_size=64,batch_size=64,num_workers=2):
    

    dataset = dset.ImageFolder(root=data_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(img_size),
                                       transforms.CenterCrop(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=int(num_workers))
    return dataloader ,dataset






#%%
'''
Visual inspection of the difference between the different schemes to increase feature map size:
'''
def compare_ups(img_no=2):
    
    _,dataset=dataloader()
    
    Img = dataset[img_no][0]

    # Original resized image
    plt.subplot(2,2,1)
    print ("image size : ", Img.size())
    plt.imshow((Img.numpy().transpose(1, 2, 0)*0.5 +0.5))
    img = Variable(Img.view(1,3,64,64))
    conv = nn.Conv2d(3,10,4,2,1)
    conv_img = conv(img) 

    # Deconvolution (transposed convolution) with paddings and strides.
    plt.subplot(2,2,2)
    deconv = nn.ConvTranspose2d(10,3,4,2,1)
    deconv.weight.data.normal_(mean=0.0, std=0.05)
    deconv.bias.data.zero_()
    deconv_img = deconv(conv_img).squeeze(0)
    print ("deconv_img size : ", deconv_img.size())
    plt.imshow((deconv_img.data.numpy().transpose(1, 2, 0)*0.5 +0.5))

    # Nearest-Neighbor Upsampling followed by regular convolution.
    plt.subplot(2,2,3)
    conv2 = nn.Conv2d(10,3,4,2,1)
    ups_nearest = nn.UpsamplingNearest2d(scale_factor=2)
    ups_nearest_img = conv2(ups_nearest(conv_img)).squeeze(0)
    print ("nearest_img size : ", ups_nearest_img.size())
    plt.imshow((ups_nearest_img.data.numpy().transpose(1, 2, 0)*0.5 +0.5))

    # Bilinear Upsampling followed by regular convolution
    plt.subplot(2,2,4)
    ups_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
    ups_bilinear_img = conv2(ups_bilinear(conv_img)).squeeze(0)
    print ("bilinear_img size : ", ups_bilinear_img.size())
    plt.imshow((ups_bilinear_img.data.numpy().transpose(1, 2, 0)*0.5 +0.5))
    plt.savefig('Q3.pdf')


compare_ups(10)
#%%
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





nn.ConvTranspose2d()

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, n_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_gen * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(n_gen * 8, n_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_gen * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(n_gen * 4, n_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_gen * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(n_gen * 2,     n_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_gen),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    n_gen,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
            output = self.main(input)
            return output
        
        
        
netG = _netG()
netG.apply(weights_init)


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, n_disc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(n_disc, n_disc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_disc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(n_disc * 2, n_disc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_disc * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(n_disc * 4, n_disc * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_disc * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(n_disc * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
    


netD = _netD()
netD.apply(weights_init)
print(netD)




criterion = nn.BCELoss()

input = torch.FloatTensor(batch_size, 3, img_size, img_size)
noise = torch.FloatTensor(batch_size, latent_dim, 1, 1)
fixed_noise = torch.FloatTensor(batch_size, latent_dim, 1, 1).normal_(0, 1)
label = torch.FloatTensor(batch_size)
real_label = 1
fake_label = 0

if use_cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
    


for epoch in range(1000):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if use_cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, latent_dim, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()


        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 1000, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

            #plt.imshow(image)
            #vutils.save_image(real_cpu,
             #       '%s/real_samples.png' % path_output,
              #      normalize=True)
            fake = netG(fixed_noise)

            #print(np.swapaxes(im,0,2).shape)
            #show(make_grid(image, padding=100, normalize=True))
            #vutils.save_image(fake.data,
            #        '%s/fake_samples_epoch_%03d.png' % ("output", epoch),
            #        normalize=True)

    #show result
    fake = netG(fixed_noise)
    vutils.save_image(fake.data,
            '%s/fake_samples_epoch_%03d.png' % ("output", epoch),
            normalize=True)
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (path_output, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (path_output, epoch))


