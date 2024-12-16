import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from tqdm import tqdm
from colorama import Fore

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


# 获取一个文件夹下的文件名
def get_filePath(path):
    pathDir = os.listdir(path)

    filePaths = []

    for file in pathDir:
        filePaths.append(file)
    return filePaths


def save_all_bone(file_name, fold_name, order_list=['clavicel', 'postrib', 'prerib']):
    pre_mask_root = fold_name
    pre_mask_clavicel_path = os.path.join(pre_mask_root, order_list[0], file_name)
    pre_mask_postrib_path = os.path.join(pre_mask_root, order_list[1], file_name)
    pre_mask_prerib_path = os.path.join(pre_mask_root, order_list[2], file_name)


    h = 512
    w = 512

    bone = getImg(pre_mask_clavicel_path, pre_mask_postrib_path, pre_mask_prerib_path)

    plt.figure(figsize=(w, h), dpi=1)
    plt.imshow(bone)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    # 图片的路径
    save_img_path = os.path.join(pre_mask_root, 'all_bone')
    mkdir(save_img_path)

    # 保存图像
    plt.savefig(os.path.join(save_img_path, file_name), format='png', transparent=True, dpi=1, pad_inches=0)




if __name__ == '__main__':
    root = r'../../dataset/bone'


    # set = ['train', 'test', 'val']  #
    set = ['mask']
    # img_lable = ['clavicle', 'heart', 'image', 'lung'] # jsrt
    # img_lable = ['image', 'lung']  # mg
    img_lable = ['bone', 'clavicle', 'post_rib', 'pre_rib']

    for sub_set in set:
        print(sub_set)
        for imgorlable in img_lable:
            print(imgorlable)
            temp_path = os.path.join(root, sub_set, imgorlable)
            # 读入图像
            file_list = get_filePath(temp_path)
            print(file_list)
            for file_name in file_list:
                img = Image.open(os.path.join(temp_path, file_name)).convert('RGB')
                h, w = 512, 512

                # resize
                img_temp = img.resize((h, w))

                # 保存图像
                plt.figure(figsize=(w, h), dpi=1)
                plt.imshow(img_temp)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                # plt.show()

                # 保存图像
                plt.savefig(os.path.join(temp_path, file_name), format='png', transparent=True, dpi=1, pad_inches=0)
