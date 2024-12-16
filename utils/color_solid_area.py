import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from tqdm import tqdm
from colorama import Fore


'''
    作用：可视化图像，预测的为红色，金标准为蓝色，两者重合区域为黄色，见save_solid_pre_gt函数
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
    '''
        file_name: 文件名
        fold_name: 预测图像父级路径
        order_list = ["预测图像文件夹名称“， ”金标准图像文件夹名称“]
    '''
    pre_mask_root = fold_name
    pre_mask_path = os.path.join(pre_mask_root, order_list[0], file_name)

    # 可以将gt_root提到参数中，为金标准父级路径，需要根据需要进行更改
    gt_root = r'/home/zdd2020/zdd_experiment/3_15_experiment_set/image_preprocess/images'
    gt_path = os.path.join(gt_root, order_list[1], file_name)

    h = 512
    w = 512

    bone = getImg(pre_mask_path, gt_path)

    # 绘制图像，取消其坐标轴
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


# 判断像素类别（预测，金标准，重合），并为其赋合适的值
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
    '''
        需要根据需要进行更改，我自身的路径配置为：
        预测的：/home/zdd2020/ZDD_paper/ZDD_data_aug/result/Bone/no_aug(可变)/fold_0(可变)/all_bone(可变)/具体文件名(.png)
        金标准：/home/zdd2020/zdd_experiment/3_15_experiment_set/image_preprocess/images/all_bone(可变)/具体文件名(.png)
    '''
    pre_order_list = ['all_bone', 'clavicel', 'postrib', 'prerib']  # 预测图像
    gt_order_list = ['bone', 'clavicle', 'post_rib', 'pre_rib']  #

    fold_name = ['fold_0']  #

    result_fold = ['no_aug', '12_10_trad_aug10']

    for fold_dir in result_fold:
        for fold in fold_name:
            path = os.path.join('/home/zdd2020/ZDD_paper/ZDD_data_aug/result/Bone', fold_dir,
                                fold, 'all_bone')

            file_list = get_filePath(path)
            print(file_list)

            for i in range(len(pre_order_list)):
                pbar = tqdm(total=len(file_list), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE))

                temp_order = [pre_order_list[i], gt_order_list[i]]
                print(temp_order)

                for file_name in file_list:
                    # file_name = '00007.png'
                    temp_fold_dir = os.path.join(
                        '/home/zdd2020/ZDD_paper/ZDD_data_aug/result/Bone', fold_dir,
                        fold)
                    save_solid_pre_gt(file_name, temp_fold_dir, order_list=temp_order)
                    pbar.update(1)
                pbar.close()
