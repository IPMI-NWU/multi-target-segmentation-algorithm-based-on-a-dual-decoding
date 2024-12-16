import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from tqdm import tqdm
from colorama import Fore

'''
    注释见color_solid_area
'''
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


# 获取一个文件夹下的文件名
def get_filePath(path):
    pathDir = os.listdir(path)

    filePaths = []

    for file in pathDir:
        # filePath = os.path.join(path, file)
        # filePaths.append(filePath)
        filePaths.append(file)
    return filePaths


def save_solid_pre_gt(file_name, fold_name, order_list=['all_bone', 'bone_contour']):
    pre_mask_root = fold_name
    pre_mask_path = os.path.join(pre_mask_root, order_list[0], file_name)

    gt_root = r'../../dataset/bone/mask'
    gt_path = os.path.join(gt_root, order_list[1], file_name)

    h = 512
    w = 512

    bone = getImg(pre_mask_path, gt_path)

    plt.figure(figsize=(w, h), dpi=1)
    plt.imshow(bone)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    # 图片的路径
    save_img_path = pre_mask_root + str('_color_solid')
    mkdir(save_img_path)
    save_img_path = os.path.join(save_img_path, order_list[0])
    mkdir(save_img_path)

    # 保存图像
    plt.savefig(os.path.join(save_img_path, file_name), format='png', transparent=True, dpi=1, pad_inches=0)


def getImg(path_pre, path_gt):
    img_pre = Image.open(path_pre).convert('L')
    img_gt = Image.open(path_gt).convert('L')

    # resize
    h = 512
    w = 512

    img_pre = img_pre.resize((h, w))
    img_gt = img_gt.resize((h, w))

    # array
    img_pre_array = np.array(img_pre)
    img_gt_array = np.array(img_gt)

    # index
    img_pre_index = np.where(img_pre_array > 0)
    img_gt_index = np.where(img_gt_array > 0)

    color_pre = [255, 0, 0]  # red
    color_gt = [0, 0, 255]  # blue
    color_pre_gt = [255, 255, 0]  # yellow

    pre_gt = np.zeros((h, w, 3))

    pre_gt[img_pre_index[0], img_pre_index[1], :] = color_pre

    for i in range(len(img_gt_index[0])):
        if np.all(pre_gt[img_gt_index[0][i], img_gt_index[1][i], :] == 0):
            pre_gt[img_gt_index[0][i], img_gt_index[1][i], :] = color_gt
        else:
            pre_gt[img_gt_index[0][i], img_gt_index[1][i], :] = color_pre_gt

    final_img = Image.fromarray(np.uint8(pre_gt))
    return final_img


if __name__ == '__main__':
    pre_order_list = ['all_bone', 'clavicel', 'post_rib', 'pre_rib']  # 预测图像
    gt_order_list = ['bone', 'clavicle', 'post_rib', 'pre_rib']  #

    fold_name = ['bone_fold_0', 'bone_fold_1', 'bone_fold_2', 'bone_fold_3']  #

    result_fold = ['result_bone5_JSRT3_bone_r_j_v_m/bone5_others3_split_r_j_v_m']

    for fold_dir in result_fold:
        for fold in fold_name:
            path = os.path.join('../../', fold_dir,
                                fold, 'bone', pre_order_list[0])

            file_list = get_filePath(path)
            print(file_list)

            for i in range(len(pre_order_list)):
                pbar = tqdm(total=len(file_list), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE))

                temp_order = [pre_order_list[i], gt_order_list[i]]
                print(temp_order)

                for file_name in file_list:
                    # file_name = '00007.png'
                    temp_fold_dir = os.path.join(
                        '../../', fold_dir, fold, 'bone')
                    save_solid_pre_gt(file_name, temp_fold_dir, order_list=temp_order)
                    pbar.update(1)
                pbar.close()
