import torch
import torchvision
import torch.optim
import net
import numpy as np
from PIL import Image
import glob


def dehaze_image(image_path):
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)

    AODnet = net.AODnet_ours1().cuda()
    # AODnet = net.AODnet_ours().cuda()
    AODnet.load_state_dict(torch.jit.load('weights/AOD-Net_ours/l1_ssim/dehazer.pth'))

    clean_image = AODnet(data_hazy)
    # torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results/" + image_path.split("\\")[-1])
    # torchvision.utils.save_image(data_hazy, "results/" + "hazy/" + image_path.split("\\")[-1])
    torchvision.utils.save_image(clean_image, "results/new_res/" + image_path.split("\\")[-1])


if __name__ == '__main__':

    # test_list = glob.glob("images/*")
    test_list = glob.glob("E:/Dehazing/Defog/AOD-Net/mydata/*")
    num = 0
    for image in test_list:
        num = num + 1
        dehaze_image(image)
        print(num, "done!")
