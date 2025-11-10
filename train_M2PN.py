import os
import argparse
import cv2


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
# from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import *
from util import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from openpyxl import Workbook
from torchstat import stat
# from networks import *
# from net import PReNet_LSTM
# from net1 import PReNet_LSTM
# from SimAM_net2 import PReNet_LSTM
# from SENet_net import PReNet_LSTM
# from SENet_net2 import PReNet_LSTM
# from Dense_networks import PReNet_LSTM
from m2pn import M2PN
# from fcanet_biLSTM import PReNet_LSTM
# from fcanet_pixelshuffle import PReNet_LSTM
# from fcanetLSTM import PReNet_LSTM

parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/log/fcaDCT16/rainHwithtestdatarate1e-3", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/RainTrainH",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    model = M2PN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    print_network(model)


    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('MODEL SAVE AT {}'.format(opt.save_path))
    print('Total number of parameters: %d' % num_params)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates milestone=[30,50,80]

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)), False)

    # start training
    # workbook = Workbook()
    # worksheet = workbook.active

    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        SSIM_list = []
        psnr_list = []
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()
            out_train, _ = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric
            loss.backward()
            optimizer.step()
            # training curve
            model.eval()
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            data_row = [step, epoch, loss.item(), pixel_metric.item(), psnr_train]
            SSIM_list.append(pixel_metric.item())
            psnr_list.append(psnr_train)
            # worksheet.append(data_row)
            print("[epoch %d][%d/%d] loss: %.4f, SSIM: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))


            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('SSIM on training data', pixel_metric.item(), step)

            step += 1
        ## epoch training end
        #workbook.save('outputTrainHbig.xlsx')
        # log the images
        model.eval()
        out_train, _ = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)

        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)

        # im_target = im_target.cpu().numpy()
        # im_input = im_input.cpu().numpy()
        # im_derain = im_derain.cpu().numpy()
        # dct_target = cv2.dct(np.float32(im_target))
        # dct_input = cv2.dct(np.float32(im_input))
        # dct_derain = cv2.dct(np.float32(im_derain))
        # writer.add_image('dctclean image', dct_target, epoch + 1)
        # writer.add_image('dctrainy image', dct_input, epoch + 1)
        # writer.add_image('dctderaining image', dct_derain, epoch + 1)尝试保留特征图暂时失败

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))

        SSIM_average = sum(SSIM_list) / len(SSIM_list)
        psnr_average = sum(psnr_list) / len(psnr_list)
        print("[epoch %d], pixel_metric_average: %.4f, PSNR_average: %.4f" %
              (epoch + 1, SSIM_average, psnr_average))
        SSIM_list.clear()
        psnr_list.clear()

if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            # print(opt.data_path.find('RainTrainH'))
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=98, stride=120)
        elif opt.data_path.find('RainTrain200H') != -1:
            prepare_data_RainTrain200H(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('RainTrain200L') != -1:
            prepare_data_RainTrain200L(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain800') != -1:
            prepare_data_Rain800(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain1200H') != -1:
            prepare_data_Rain1200H(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain1200M') != -1:
            prepare_data_Rain1200M(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain1200L') != -1:
            prepare_data_Rain1200L(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('LSUI') != -1:
            prepare_data_LSUI(data_path=opt.data_path, patch_size=98, stride=80)
        else:
            print('unknown datasets: please define prepare data function in DerainDataset.py')


    main()
