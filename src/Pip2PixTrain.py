import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os

import argparse

from MyDataset.ShipDataset import ShipDataset
from MyNetwork.Pix2Pix import Generator, Discriminator

# Root directory for dataset
dataroot = "../data/seg_ship_data/4-yyc_pic_output_train"

# Root directory for model
modelroot = "../model"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 16

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nic = 3

# Number of channels in the label images. For color images this is 1
noc = 1

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_model(netG, netD, modelroot, epoch=num_epochs):
    path_g = os.path.join(modelroot, 'Generator_epoch_{0}.pkl'.format(epoch))
    path_d = os.path.join(modelroot, 'Discriminator_epoch_{0}.pkl'.format(epoch))
    torch.save(netG.state_dict(), path_g)
    torch.save(netD.state_dict(), path_d)

def train(netG, netD, dataloader, num_epochs, modelroot):

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    # img_list = []
    G_losses = []
    D_losses = []
    iters = 1

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(1, num_epochs+1):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            batch_imgs = data[0].to(device)
            batch_labels = data[1].to(device)
            b_size = batch_imgs.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(batch_imgs, batch_labels).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate fake image batch with G
            fake = netG(batch_imgs)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(batch_imgs, fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(batch_imgs, fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 2 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        save_model(netG, netD, modelroot, epoch)

    return G_losses, D_losses

def test():
    pass

def show_losses(G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--num_epochs', '-n', type=int, default=num_epochs,  help='训练的轮数')
    parse.add_argument('--dataroot', '-d', default=dataroot, help='数据集的目录')
    parse.add_argument('--modelroot', '-m', default=modelroot, help='保存网络模型的目录')

    args = parse.parse_args()
    num_epochs = args.num_epochs
    dataroot = args.dataroot
    modelroot = args.modelroot

    # print(num_epochs)
    # print(dataroot)
    # print(modelroot)

    dataset = ShipDataset(dataroot, transform_img=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                           ]), transform_label=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,)),
                            ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # print(len(dataloader))

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

    # Create the generator
    netG = Generator(in_channels=nic, out_channels=noc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(in_channels1=nic, in_channels2=noc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    G_losses, D_losses = train(netG, netD, dataloader, num_epochs, modelroot)

    show_losses(G_losses, D_losses)

    plt.show()

