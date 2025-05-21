import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from util import *
from SSIM import SSIM

import time
# from m2pn import M2PN
from LM2PN import LM2PN

parser = argparse.ArgumentParser(description="Test_M2PN")
parser.add_argument("--logdir", type=str, default="/root/autodl-tmp/pycharm_project_618/logs/log/fcaDCT16/youth2025_100LbaseontestL/net_epoch131.pth", help='path to model and log files')
parser.add_argument("--data_path", type=str, default=r"datasets/test/Rain200L", help='path to test data')
parser.add_argument("--gt_path", type=str, default=r"datasets/test/2", help='path to ground truth data')
parser.add_argument("--save_path", type=str, default=r"/root/autodl-fs/release/fcaDCT16/youth2025/youth2025_100LbaseontestL_epoch131_200L", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = LM2PN(opt.recurrent_iter, opt.use_GPU)
    # print_network(model)

    criterion = SSIM()
    if opt.use_GPU:
        model = model.cuda()
        criterion.cuda()
    
    # 加载模型检查点
    checkpoint = torch.load(os.path.join(opt.logdir))
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    time_test = 0
    count = 0
    psnr_list = []
    ssim_list = []
    
    print("\nProcessing images:")
    print("Image Name\t\tPSNR\t\tSSIM\t\tTime")
    print("-" * 50)
    
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)
            # 修改文件名匹配方式：将'rain-'替换为'norain-'
            gt_name = img_name.replace('rain-', 'norain-')
            target_path = os.path.join(opt.gt_path, gt_name)

            # 检查输入图像和目标图像是否存在
            if not os.path.exists(img_path):
                print(f"Warning: Input image {img_path} does not exist, skipping...")
                continue
            if not os.path.exists(target_path):
                print(f"Warning: Target image {target_path} does not exist, skipping...")
                continue

            # input image
            y = cv2.imread(img_path)
            if y is None:
                print(f"Warning: Cannot read input image {img_path}, skipping...")
                continue
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])

            # target image
            target = cv2.imread(target_path)
            if target is None:
                print(f"Warning: Cannot read target image {target_path}, skipping...")
                continue
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            
            # 预处理
            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            target = normalize(np.float32(target))
            target = np.expand_dims(target.transpose(2, 0, 1), 0)
            target = Variable(torch.Tensor(target))

            if opt.use_GPU:
                y = y.cuda()
                target = target.cuda()

            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()

                out, _ = model(y)
                out = torch.clamp(out, 0., 1.)

                if opt.use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                # 计算PSNR和SSIM
                psnr_value = batch_PSNR(out, target, 1.0)
                ssim_value = criterion(out, target).item()
                
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)

                print(f"{img_name:<20} {psnr_value:>8.2f} dB\t{ssim_value:>8.4f}\t{dur_time:>8.4f}s")

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)

            count += 1

    # 打印平均指标
    print("-" * 50)
    print(f"Average Results:")
    print(f"PSNR: {sum(psnr_list)/len(psnr_list):.2f} dB")
    print(f"SSIM: {sum(ssim_list)/len(ssim_list):.4f}")
    print(f"Time: {time_test/count:.4f}s")


if __name__ == "__main__":
    main()

