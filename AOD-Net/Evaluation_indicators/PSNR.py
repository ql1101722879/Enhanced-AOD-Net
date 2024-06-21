from PIL import Image
import numpy
import math


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


img1 = Image.open('../results/AOD-Net-PM-Results/clean_canyon.png')
img2 = Image.open('../results/AOD-Net-PM-Results/hazy_canyon.png')

i1_array = numpy.array(img1)
i2_array = numpy.array(img2)

r12 = psnr(i1_array, i2_array)
print(r12)