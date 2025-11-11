import os
import argparse
import cv2


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from m2pn import M2PN
from LM2PN import LM2PN
from distill import distillation_loss


parser = argparse.ArgumentParser(description="Distillation_train_M2PN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=24, help="Training batch size")
parser.add_argument("--epochs", type=int, default=131, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--temperature", type=float, default=20.0, help="Temperature parameter for knowledge distillation")
parser.add_argument("--alpha", type=float, default=0.8, help="SSIM损失权重")
parser.add_argument("--beta", type=float, default=0.1, help="L1损失权重")
parser.add_argument("--gamma", type=float, default=0.1, help="特征SSIM损失权重")
parser.add_argument("--lambda_distill", type=float, default=0.6, help="蒸馏损失总权重")
parser.add_argument("--save_path", type=str, default="logs/log/fcaDCT16/youth2025_100LbaseontestL", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/RainTrainL",help='path to training data')
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
    # build tea&stu model
    teacher_model = M2PN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    student_model = LM2PN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    
    # 加载预训练的教师模型
    teacher_checkpoint = torch.load('logs/log/fcaDCT16/testL/net_latest.pth')  # 使用你训练好的M2PN模型路径
    if isinstance(teacher_checkpoint, dict) and 'state_dict' in teacher_checkpoint:
        teacher_model.load_state_dict(teacher_checkpoint['state_dict'])
    else:
        teacher_model.load_state_dict(teacher_checkpoint)
    teacher_model.eval()  # 设置为评估模式
    
    print_network(teacher_model)
    print_network(student_model)

    # loss function
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        checkpoint = torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch))
        student_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('Successfully loaded checkpoint from epoch %d' % initial_epoch)

    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        # 用于收集每个epoch的指标
        epoch_psnr_list = []
        epoch_ssim_list = []
        epoch_teacher_psnr_list = []
        epoch_teacher_ssim_list = []
        
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            student_model.train()
            student_model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            # 教师模型的输出（确保在eval模式下）
            with torch.no_grad():
                teacher_out, teacher_features = teacher_model(input_train)
                # 验证教师模型的输出质量
                teacher_ssim = criterion(teacher_out, target_train)
                teacher_psnr = batch_PSNR(teacher_out, target_train, 1.)
                epoch_teacher_psnr_list.append(teacher_psnr)
                epoch_teacher_ssim_list.append(teacher_ssim)
            
            # 学生模型的输出
            student_out, student_features = student_model(input_train)
            
            # 1. 直接监督损失（与ground truth比较）
            ssim_value = criterion(student_out, target_train)
            direct_loss = -ssim_value  # 最大化SSIM
            
            # 2. 蒸馏损失（只有当教师模型输出质量好时才使用）
            if teacher_ssim > 0.87:  # 设置一个阈值
                distill_loss, loss_details = distillation_loss(
                    student_out, teacher_out,
                    student_features, teacher_features,
                    input_train,
                    alpha=0.3,
                    beta=0.2,
                    gamma=0.1
                )
                # 总损失：主要是直接监督，辅以蒸馏
                total_loss = 0.7 * direct_loss + 0.3 * distill_loss
            else:
                # 如果教师模型输出质量不好，只使用直接监督
                total_loss = direct_loss
                distill_loss = torch.tensor(0.0).to(student_out.device)
            
            total_loss.backward()
            optimizer.step()
            
            # 评估阶段
            student_model.eval()
            with torch.no_grad():
                student_out, _ = student_model(input_train)
                student_out = torch.clamp(student_out, 0., 1.)
                psnr_train = batch_PSNR(student_out, target_train, 1.)
                ssim_train = criterion(student_out, target_train)
                epoch_psnr_list.append(psnr_train)
                epoch_ssim_list.append(ssim_train)

            print("[epoch %d][%d/%d] direct_loss: %.4f, distill_loss: %.4f, PSNR: %.4f, SSIM: %.4f, teacher_SSIM: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), 
                   direct_loss.item(),
                   distill_loss.item(),
                   psnr_train,
                   ssim_train,
                   teacher_ssim))

            if step % 10 == 0:
                writer.add_scalar('direct_loss', direct_loss.item(), step)
                writer.add_scalar('distill_loss', distill_loss.item(), step)
                writer.add_scalar('total_loss', total_loss.item(), step)
                writer.add_scalar('PSNR', psnr_train, step)
                writer.add_scalar('SSIM', ssim_train, step)
                writer.add_scalar('teacher_SSIM', teacher_ssim, step)
                writer.add_scalar('teacher_PSNR', teacher_psnr, step)

            step += 1
            
        # 计算epoch平均值
        avg_psnr = sum(epoch_psnr_list) / len(epoch_psnr_list)
        avg_ssim = sum(epoch_ssim_list) / len(epoch_ssim_list)
        avg_teacher_psnr = sum(epoch_teacher_psnr_list) / len(epoch_teacher_psnr_list)
        avg_teacher_ssim = sum(epoch_teacher_ssim_list) / len(epoch_teacher_ssim_list)
        
        print("\n[Epoch %d] Average Metrics:" % (epoch + 1))
        print("Student - PSNR: %.4f, SSIM: %.4f" % (avg_psnr, avg_ssim))
        print("Teacher - PSNR: %.4f, SSIM: %.4f\n" % (avg_teacher_psnr, avg_teacher_ssim))
        
        # log the images

        # model.eval()
        student_model.eval()
        with torch.no_grad():
            student_out, student_features = student_model(input_train)
            teacher_out, teacher_features = teacher_model(input_train)
        
        # 确保输出在合理范围内
        student_out = torch.clamp(student_out, 0., 1.)
        teacher_out = torch.clamp(teacher_out, 0., 1.)
        
        # 基本图像的可视化
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_deraint = utils.make_grid(teacher_out.data, nrow=8, normalize=True, scale_each=True)
        im_derains = utils.make_grid(student_out.data, nrow=8, normalize=True, scale_each=True)

        # 可视化中间特征图
        def visualize_features(features, prefix):
            # 选择最后一次迭代的特征图
            last_feature = features[-1]  # [B, C, H, W]
            # 计算特征图的平均值，得到注意力图
            attention_map = torch.mean(torch.abs(last_feature), dim=1, keepdim=True)  # [B, 1, H, W]
            # 归一化到[0,1]范围
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
            # 转换为热力图形式
            attention_grid = utils.make_grid(attention_map, nrow=8, normalize=True, scale_each=True)
            return attention_grid

        # 生成教师和学生模型的特征可视化
        teacher_features_vis = visualize_features(teacher_features, 'teacher')
        student_features_vis = visualize_features(student_features, 'student')

        # 记录到tensorboard
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('teacher derain', im_deraint, epoch+1)
        writer.add_image('student derain', im_derains, epoch+1)
        writer.add_image('teacher features', teacher_features_vis, epoch+1)
        writer.add_image('student features', student_features_vis, epoch+1)

        # 保存当前epoch的图像到文件
        if epoch % opt.save_freq == 0:
            save_dir = os.path.join(opt.save_path, 'images', f'epoch_{epoch+1}')
            os.makedirs(save_dir, exist_ok=True)
            
            def save_tensor_image(tensor, path):
                image = tensor.cpu().numpy().transpose(1, 2, 0)
                image = (image * 255).astype(np.uint8)
                cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            save_tensor_image(im_target.cpu(), os.path.join(save_dir, 'target.png'))
            save_tensor_image(im_input.cpu(), os.path.join(save_dir, 'input.png'))
            save_tensor_image(im_deraint.cpu(), os.path.join(save_dir, 'teacher_derain.png'))
            save_tensor_image(im_derains.cpu(), os.path.join(save_dir, 'student_derain.png'))
            save_tensor_image(teacher_features_vis.cpu(), os.path.join(save_dir, 'teacher_features.png'))
            save_tensor_image(student_features_vis.cpu(), os.path.join(save_dir, 'student_features.png'))

        # save model
        torch.save({
            'epoch': epoch,
            'state_dict': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step
        }, os.path.join(opt.save_path, 'net_latest.pth'))
        
        if epoch % opt.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step
            }, os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch + 1)))

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
        else:
            print('unknown datasets: please define prepare data function in DerainDataset.py')


    main()
