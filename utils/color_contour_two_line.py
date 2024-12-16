import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
import cv2
from tqdm import tqdm
from colorama import Fore
import torchvision.transforms.functional as tf


# 创建文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)


# 获取一个文件夹下的文件名
def get_filePath(path):
    pathDir = os.listdir(path)

    filePaths = []

    for file in pathDir:
        filePaths.append(file)
    return filePaths


def sort_edge(image):
    gray = (image * 255).astype(np.uint8)
    ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours[0] != []:
        return contours[0]
    else:
        return 0


def draw_contour_new(img, mask, color):
    # h, w = mask.size
    n = 5
    # mask 向下平移 n 个像素
    # mask = tf.crop(mask, top=0, left=0, height=n, width=w)
    # mask = tf.pad(mask, padding=[0, n, 0, 0], fill=0)

    # 转为numpy
    h, w = mask.size
    h_img, w_img = img.size

    if h > h_img:
        h = h_img

    if w > w_img:
        w = w_img

    img = img.resize((h, w))
    mask = mask.resize((h, w))
    # print(mask.size)
    # img_array = np.array((mask.size[0], mask.size[1]))
    img_array = np.array(img)
    mask_array = np.array(mask)

    # temp_mask = np.zeros((h, w))
    # for i in range(n, h):
    #     for j in range(w):
    #         temp_mask[i][j] = mask_array[i-n][j]
    #
    # mask_array = temp_mask
    mask = (mask_array > 0)

    contour = sort_edge(mask)

    result = cv2.drawContours(img_array, contour, -1, color, 1)  # 后肋为红色

    result = Image.fromarray(np.uint8(result))

    # plt.figure()
    # plt.imshow(result, cmap='gray')
    # # plt.axis('off')
    # plt.show()

    return result


# mask 向下平移 N 个像素，作为 MUNIT 生成图像的 mask
def mask_translate_vertical(mask, h, w, n):
    mask = mask.resize((h, w))
    mask_array = np.array(mask)

    temp_mask = np.zeros((h, w))
    for i in range(n, h):
        for j in range(w):
            temp_mask[i][j] = mask_array[i - n][j]

    result = Image.fromarray(np.uint8(temp_mask))

    return result

def save(img, file_name, save_img_path, type='none'):
    w, h = img.size
    plt.figure(figsize=(w, h), dpi=1)

    if type == 'none':
        plt.imshow(img)
    if type == 'gray':
        # print('111')
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    # 图片的路径
    plt.savefig(os.path.join(save_img_path, file_name), format=file_name.split('.')[-1], transparent=True, dpi=1,  pad_inches=0)


# mask 平移
def mask_translate():
    # 读入 mask
    root = r'/home/zdd2020/zdd_experiment/3_15_experiment_set/image_preprocess/images'
    label_list = ['bone', 'clavicle', 'post_rib', 'pre_rib']  #
    index = 2

    file_list = get_filePath(os.path.join(root, label_list[index]))
    save_img_path = os.path.join(r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/test_900000_mask_pixel3',
                                 label_list[index])
    mkdir(save_img_path)

    pbar = tqdm(total=len(file_list), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE))
    for file_name in file_list:
        mask = Image.open(os.path.join(root, label_list[index], file_name)).convert('1')
        # 获得平移后的结果
        result = mask_translate_vertical(mask, h=512, w=512, n=3)

        # 保存
        save(result, file_name, save_img_path, type='gray')

        pbar.update(1)
    pbar.close()

def fun():
    ''' 想将标注的所有骨画在 munit 增强的图像上，挑选合适的增强数据 '''
    # 首先从 AllData 中获取 image 文件
    all_data_root = r'/home/zdd2020/zdd_experiment/All_Data/bone'
    # label_root = r'/home/zdd2020/zdd_experiment/3_15_experiment_set/image_preprocess/images'
    label_root = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/test_900000_mask_pixel3'  # test_900000_mask
    label_list = ['bone', 'clavicle', 'post_rib', 'pre_rib']
    index = 1

    munit_root = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/test_900000_contour_cla_pingyi_pixel3' # test_900000

    file_list = get_filePath(os.path.join(all_data_root, 'image'))

    pbar = tqdm(total=len(file_list), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE))
    for file_name in file_list:
        file_name_prefix = file_name.split('.')[0]

        munit_file_list = get_filePath(os.path.join(munit_root, file_name_prefix))

        label_path = os.path.join(label_root, label_list[index], file_name)
        mask = Image.open(label_path).convert('L')

        for munit_file_name in munit_file_list:
            # if munit_file_name == 'input.jpg':
            #     continue

            img_path = os.path.join(munit_root, file_name_prefix, munit_file_name)

            # 打开文件
            img = Image.open(img_path).convert('RGB')

            # 绘制contour
            final_img = draw_contour_new(img, mask, (0, 0, 255))  # 红色

            # 保存
            save_img_path = os.path.join(r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/test_900000_contour_cla_pingyi_pixel3', file_name_prefix)
            mkdir(save_img_path)
            save(final_img, munit_file_name, save_img_path)

        # munit for 循环结束

        pbar.update(1)
    pbar.close()

    # mask_translate()

if __name__ == '__main__':
    ''' 在原始图像上画 前肋、后肋骨的 contour '''
    all_data_root = r'/home/zdd2020/zdd_experiment/All_Data/bone'  # 首先从 AllData 中获取 image 文件
    label_root = r'/home/zdd2020/zdd_experiment/3_15_experiment_set/image_preprocess/images'
    label_list = ['bone', 'clavicle', 'post_rib', 'pre_rib']
    index = 1

    file_list = get_filePath(os.path.join(all_data_root, 'image'))

    pbar = tqdm(total=len(file_list), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE))
    for file_name in file_list:
        img_path = os.path.join(all_data_root, 'image', file_name)
        img = Image.open(img_path).convert('RGB')

        mask_pre_path = os.path.join(label_root, label_list[-1], file_name)
        mask_post_path = os.path.join(label_root, label_list[-2], file_name)

        mask_pre = Image.open(mask_pre_path).convert('L')
        mask_post = Image.open(mask_post_path).convert('L')

        # 绘制contour
        post_img = draw_contour_new(img, mask_post, (255, 0, 0))  # 红色
        pre_img = draw_contour_new(post_img, mask_pre, (0, 0, 255))

        # 保存
        save_img_path = os.path.join(
            r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/post_pre_contour')
        mkdir(save_img_path)
        save(pre_img, file_name, save_img_path)

        # munit for 循环结束

        pbar.update(1)
    pbar.close()



