from PIL import Image
import os


def resize_images_in_folder(folder_path, output_size):
    num = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 打开图像文件
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            # 调整分辨率
            image = image.resize(output_size)

            # 保存调整后的图像
            output_path = os.path.join(folder_path, filename)
            image.save(output_path)
            num += 1
            print(num)


# 调用函数
folder_path = 'F:/PycharmProjects/Defog/DataSetGeneration/YOLOv5/FogVal/'
output_size = (1920, 1080)  # (宽, 高) 或者 (长, 宽)
resize_images_in_folder(folder_path, output_size)
