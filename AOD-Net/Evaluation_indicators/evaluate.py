import os
import numpy as np
from glob import glob
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import matplotlib.pyplot as plt
import csv


def read_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# def mse(tf_img1, tf_img2):
#     return mean_squared_error(tf_img1, tf_img2)


def psnr(tf_img1, tf_img2):
    return peak_signal_noise_ratio(tf_img1, tf_img2)
    # return tf.image.psnr(tf_img1, tf_img2, max_val=255)

def ssim(tf_img1, tf_img2):
    return structural_similarity(tf_img1, tf_img2)


def main():
    ## DCP
    # WSI_MASK_PATH1 = 'F:/PycharmProjects/Defog/Dark Channel Prior/results/'  ## 去雾后的图像
    ## AOD-Net
    # WSI_MASK_PATH1 = 'F:/PycharmProjects/Defog/AOD-Net/results/clean/'  ## 去雾后的图像
    # Light-DehazeNet
    # WSI_MASK_PATH1 = 'F:/PycharmProjects/Defog/Light-DehazeNet/results/'  ## 去雾后的图像
    # GCA-Net
    # WSI_MASK_PATH1 = 'F:/PycharmProjects/Defog/GCANet/output/'  ## 去雾后的图像
    # FFA-Net
    WSI_MASK_PATH1 = 'F:/PycharmProjects/Defog/FFA-Net/pred_FFA_ots/'  ## 去雾后的图像
    WSI_MASK_PATH2 = 'F:/PycharmProjects/Defog/DataSetGeneration/SeaShips/SeaShips2007-DET-test/images_1985_clean/'  ## 真实场景下的图像

    path_real = glob(os.path.join(WSI_MASK_PATH1, '*.jpg'))
    path_fake = glob(os.path.join(WSI_MASK_PATH2, '*.jpg'))
    print(path_real)
    print(path_fake)
    list_psnr = []
    list_ssim = []
    # list_mse = []

    for i in range(len(path_real)):
        t1 = read_img(path_real[i])
        t2 = read_img(path_fake[i])
        result1 = np.zeros(t1.shape, dtype=np.float32)
        result2 = np.zeros(t2.shape, dtype=np.float32)
        cv2.normalize(t1, result1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.normalize(t2, result2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # mse_num = mse(result1, result2)
        psnr_num = psnr(result1, result2)
        ssim_num = ssim(result1, result2)
        list_psnr.append(psnr_num)
        list_ssim.append(ssim_num)
        # list_mse.append(mse_num)

        # 输出每张图像的指标：
        print("{}/".format(i + 1) + "{}:".format(len(path_real)))
        str = "\\"
        print("image:" + path_real[i][(path_real[i].index(str) + 1):])
        print("PSNR:", psnr_num)
        print("SSIM:", ssim_num)
        # print("MSE:", mse_num)
    # 将PSNR和SSIM值转换为二维列表
    data = []
    for i in range(len(list_psnr)):
        data.append([list_psnr[i], list_ssim[i]])
    # 将数据写入CSV文件中
    # with open('AOD-Net_ours/psnr_ssim.csv', 'w', newline='') as file:
    with open('FFA-Net/psnr_ssim.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PSNR', 'SSIM'])  # 写入表头
        writer.writerows(data)  # 写入数据

    ## 计算平均值
    print("平均PSNR:", np.mean(list_psnr))
    print("平均SSIM:", np.mean(list_ssim))

    ## 可视化
    x = np.arange(1, len(path_real) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, list_psnr, label='PSNR')
    ax.plot(x, list_ssim, label='SSIM')
    ax.axhline(y=np.mean(list_psnr), linestyle='--', color='red', linewidth=2, label='Average PSNR')
    ax.axhline(y=np.mean(list_ssim), linestyle='--', color='green', label='Average SSIM')
    ax.set_xlabel('Image', fontsize=16)
    ax.set_ylabel('Value', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title('PSNR and SSIM of Images', fontsize=18)
    ## 生成一个y轴范围在0到35之间，并且刻度之间的间距为5
    ax.set_ylim([0, 45])
    ax.set_yticks(np.arange(0, 46, 5))
    ax.legend()
    plt.show()
    fig.savefig('psnr_ssim.png', dpi=200)


if __name__ == '__main__':
    main()
