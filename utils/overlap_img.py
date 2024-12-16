# 2种骨骼重叠的区域用黄色表示，3种重叠的区域用红色表示


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
import cv2


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


def save_img(path_pre, path_gt, img_name, file_name):
    path = r'../../result/ourMethod'  # unet_unetPlusPlus
    mkdir(path)

    # 基础网络
    # path = os.path.join(path, 'baseNetwork_compare')

    # 基础网络 + 数据扩充(10倍+hanshengData)
    # path = os.path.join(path, 'base_10bei_hanshengData_Network_compare')

    # 基础网络+loss
    # path = os.path.join(path, 'base_diceloss_lovasz_Network_compare')

    # 我们的方法
    # path = os.path.join(path, 'ourMethod_compare')

    # unet single task
    # path = os.path.join(path, 'unet_single_task_compare')

    # # unet mutil task
    path = os.path.join(path, 'unet_mutil_task_compare')
    #
    # unetPlusPlus single task
    # path = os.path.join(path, 'unetPlusPlus_single_task_compare')

    # # unetPlusPlus mutil task
    # path = os.path.join(path, 'unetPlusPlus_mutil_task_compare')

    # ourMethod single task
    # path = os.path.join(path, 'ourMethod_single_task_compare')

    mkdir(path)

    result_path = os.path.join(path, file_name)
    mkdir(result_path)

    plt.figure()
    bone = getImg(path_pre, path_gt)
    plt.imshow(bone)  # save rgb image
    # plt.show()
    plt.axis('off')
    #
    # 图片的路径
    save_img_path = os.path.join(result_path, img_name)
    plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0)  # 保存图像


def getImg(path_origin, path_clavicle, path_postrib, path_prerib, path_allbone, img_name):
    img_origin = Image.open(path_origin).convert('RGB')
    img_clavicle = Image.open(path_clavicle).convert('L')
    img_postrib = Image.open(path_postrib).convert('L')
    img_prerib = Image.open(path_prerib).convert('L')
    img_allbone = Image.open(path_allbone).convert('L')

    # resize
    h = 512
    w = 512

    img_origin = img_origin.resize((h, w))
    img_clavicle = img_clavicle.resize((h, w))
    img_postrib = img_postrib.resize((h, w))
    img_prerib = img_prerib.resize((h, w))
    img_allbone = img_allbone.resize((h, w))

    # array
    img_origin_array = np.array(img_origin)
    img_clavicle_array = np.array(img_clavicle)
    img_postrib_array = np.array(img_postrib)
    img_prerib_array = np.array(img_prerib)
    img_allbone_array = np.array(img_allbone)

    # index
    img_allbone_index = np.where(img_allbone_array > 0)

    print(img_allbone_index)
    # print(img_allbone_index[0][0])
    # print(img_allbone_index[1][0])

    color_red = [0, 255, 0]  # red, 3个重叠
    color_yellow = [255, 255, 0]  # yellow, 2个重叠

    overlap_area = np.zeros((h, w, 3))
    overlap_area_allbone = np.zeros((h, w, 3))
    overlap_area_allbone[img_allbone_index[0], img_allbone_index[1], :] = [255, 255, 255]

    for i in range(len(img_allbone_index[0])):
        overlap_count = 0
        if img_clavicle_array[img_allbone_index[0][i], img_allbone_index[1][i]] > 0:
            overlap_count = overlap_count + 1
        if img_postrib_array[img_allbone_index[0][i], img_allbone_index[1][i]] > 0:
            overlap_count = overlap_count + 1
        if img_prerib_array[img_allbone_index[0][i], img_allbone_index[1][i]] > 0:
            overlap_count = overlap_count + 1

        # 如果2个重叠，用黄色填充，3个用红色填充
        if overlap_count == 2:
            img_origin_array[img_allbone_index[0][i], img_allbone_index[1][i], :] = color_yellow

        if overlap_count == 3:
            img_origin_array[img_allbone_index[0][i], img_allbone_index[1][i], :] = color_red

    # 展示overlap_area
    final_origin_overlap = Image.fromarray(np.uint8(img_origin_array))
    plt.imshow(final_origin_overlap)
    # plt.show()

    # plt.axis('off')

    # overlap_name = ['overlap_2_3', 'overlap_2_3_allbone', 'overlap_origin', 'overlap_origin_half']
    # # 图片的路径
    # path = os.path.join('/home/zdd2020/ZDD', 'overlap_img')
    # mkdir(path)
    # result_path = os.path.join(path, overlap_name[1]+'test')
    # mkdir(result_path)
    # save_img_path = os.path.join(result_path, img_name)
    # plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0, dpi=800)  # 保存图像

    return final_origin_overlap


def gray2rgb(pic):
    pic = np.expand_dims(pic, axis=2)
    img = np.concatenate((pic, pic, pic), axis=2)  # axes are 0-indexed, i.e. 0,1,2
    return img


def sort_edge(image):
    gray = (image * 255).astype(np.uint8)
    ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours[0] != []:
        return contours[0]
    else:
        return 0


def draw_contour(path_origin, path_postrib, path_prerib, path_clavicle, img_name):
    postrib_mask = Image.open(path_postrib).convert('L')
    prerib_mask = Image.open(path_prerib).convert('L')
    clavicle_mask = Image.open(path_clavicle).convert('L')
    origin_img = Image.open(path_origin).convert('RGB')
    # origin_img = path_origin

    h = 1024
    w = 1024

    # resize
    postrib_mask = postrib_mask.resize((h, w))
    prerib_mask = prerib_mask.resize((h, w))
    clavicle_mask = clavicle_mask.resize((h, w))
    origin_img = origin_img.resize((h, w))

    # 转为numpy
    postrib_mask_array = np.array(postrib_mask)
    prerib_mask_array = np.array(prerib_mask)
    clavicle_mask_array = np.array(clavicle_mask)
    origin_img_arrray = np.array(origin_img)

    postrib_mask = (postrib_mask_array > 0)
    prerib_mask = (prerib_mask_array > 0)
    clavicle_mask = (clavicle_mask_array > 0)

    contour_postrib = sort_edge(postrib_mask)
    contour_prerib = sort_edge(prerib_mask)
    contour_clavicle = sort_edge(clavicle_mask)

    result_postrib = cv2.drawContours(origin_img_arrray, contour_postrib, -1, (255, 0, 0), 1)  # 后肋为红色

    result = cv2.drawContours(result_postrib, contour_prerib, -1, (0, 0, 255), 1)  # 前肋为蓝色

    result_clavicle = cv2.drawContours(result, contour_clavicle, -1, (0, 255, 0), 1)  # 锁骨为绿色

    result = Image.fromarray(np.uint8(result_clavicle))
    plt.imshow(result)
    plt.axis('off')
    # plt.show()

    # 图片的路径
    path = os.path.join('/home/zdd2020/zdd_experiment/3_25_experiment_set/topc', 'overlap_img')
    mkdir(path)
    result_path = os.path.join(path, img_name + '_contour_1')
    mkdir(result_path)
    save_img_path = os.path.join(result_path, 'result_lineWidth_2')
    plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0, dpi=800)  # 保存图像


if __name__ == '__main__':
    # read an image
    path_root = r'/home/zdd2020/ZDD/Unet++/data/one_folder/train/'  # 00005.png
    # path_root = r'/home/zdd2020/ZDD/Unet++/data/one_folder/test/'
    path_label = r'/home/zdd2020/zdd_experiment/3_15_experiment_set/image_preprocess/images'

    img_label_name = ['image0', 'clavicle', 'post_rib', 'pre_rib', 'bone']

    # save_img(path_pre, path_gt, '00005.png', seg_result_name[0])

    # file_list = get_filePath(os.path.join(path_root, img_label_name[0]))
    # print(file_list)

    # img_name = '00092.png'

    file_list = ['00027.png']

    for img_name in file_list:
        img_label_path = []
        for i in range(len(img_label_name)):
            if i == 0:
                path_temp = os.path.join(path_root, img_label_name[i])
            else:
                path_temp = os.path.join(path_label, img_label_name[i])

            img_label_path.append(os.path.join(path_temp, img_name))

        # origin_img = getImg(img_label_path[0], img_label_path[1], img_label_path[2], img_label_path[3],
        #                     img_label_path[4], img_name)
        draw_contour(img_label_path[0], img_label_path[2], img_label_path[3], img_label_path[1],  img_name)
