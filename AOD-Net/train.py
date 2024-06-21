import torchvision
import torch.optim
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchinfo import torchinfo

import dataloader
import net
import torch
from model.models_info import model_info
from ms_ssim_l1_loss import MS_SSIM_L1_LOSS
import ms_ssim_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

    AODnet = net.AODnet_ours().cuda()
    AODnet.apply(weights_init)

    train_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                               config.hazy_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                             config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
                                             num_workers=config.num_workers, pin_memory=True)

    # criterion = torch.nn.MSELoss().cuda()
    criterion = MS_SSIM_L1_LOSS().cuda()
    # criterion = torch.nn.L1Loss().cuda()
    # criterion = ms_ssim_loss().cuda()



    optimizer = torch.optim.Adam(AODnet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    logger = SummaryWriter(log_dir='logs')

    AODnet.train()
    model_info(AODnet)  ## 打印网络模型参数
    print(torchinfo.summary(AODnet, (3, 640, 480), batch_dim=0,
                            col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose=0))
    for epoch in range(config.num_epochs):
        running_loss = 0
        intermediate_outputs_list = []
        for iteration, (img_orig, img_haze) in enumerate(train_loader):

            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image,intermediate_outputs = AODnet(img_haze)

            loss = criterion(clean_image, img_orig)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(AODnet.parameters(), config.grad_clip_norm)
            optimizer.step()
            running_loss += loss.item()
            # 将中间特征图添加到列表中
            intermediate_outputs_list.append(intermediate_outputs)
            if ((iteration + 1) % config.display_iter) == 0:
                logger.add_scalar('iteration_loss', loss.item(), iteration)
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(AODnet.state_dict(), config.weights_folder + "Epoch" + str(epoch) + '.pth')
        # torch.save(intermediate_outputs_list, config.intermediate_outputs_path)
        logger.add_scalar('loss', running_loss / len(train_loader), epoch + 1)
        # Validation Stage
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = AODnet(img_haze)

            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         config.sample_output_folder + str(iter_val + 1) + ".jpg")

        torch.save(AODnet.state_dict(), config.weights_folder + "dehazer.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="data/mydata/clean/")
    parser.add_argument('--hazy_images_path', type=str, default="data/mydata/haze/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=50)
    parser.add_argument('--weights_folder', type=str, default="weights/newweights/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")
    # parser.add_argument('--intermediate_outputs_path', type=str, default="feature_map/")
    config = parser.parse_args()

    if not os.path.exists(config.weights_folder):
        os.mkdir(config.weights_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)
