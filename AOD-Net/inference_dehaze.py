import torch
import torchvision
import argparse
import os

import net
import dataloader

def inference(config):

    AODnet = net.AODnet_ours().cuda()
    AODnet.load_state_dict(torch.load(config.model_weights_path))
    AODnet.eval()

    test_dataset = dataloader.dehazing_loader(config.test_orig_images_path,
                                               config.test_hazy_images_path, mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False,
                                              num_workers=config.num_workers, pin_memory=True)

    with torch.no_grad():
        for iter_test, (img_orig, img_haze) in enumerate(test_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            # 前向传播
            clean_image, intermediate_outputs = AODnet(img_haze)

            # 保存中间特征图
            torch.save(intermediate_outputs, os.path.join(config.output_folder, f"intermediate_output_{iter_test}.pth"))

            # 保存结果图像
            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         os.path.join(config.output_folder, f"result_{iter_test}.jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 输入参数
    parser.add_argument('--model_weights_path', type=str, default="weights/newweights/dehazer.pth")
    parser.add_argument('--test_orig_images_path', type=str, default="data/test/clean/")
    parser.add_argument('--test_hazy_images_path', type=str, default="data/test/haze/")
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--output_folder', type=str, default="results/new_results/")

    config = parser.parse_args()

    if not os.path.exists(config.output_folder):
        os.mkdir(config.output_folder)

    inference(config)
