from math import ceil
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os

'''
    作用：将长宽不一致的图像填充为正方形
'''
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

# 填充到最接近base整数倍的长和宽图像大小
def get_padding_pic_mask(image_path, result_img_path, mask_path, result_mask_path):
    # C, H, W
    src = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    # plt.imshow(src, cmap='gray')
    # plt.show()

    src = to_tensor(src)
    mask = to_tensor(mask)
    # print(src.shape)  # torch.Size([3, 800, 600])
    # channel: (R, G, B) / 255
    origin_h, origin_w = src.shape[1], src.shape[2]
    print('原图像大小, height: {}, width: {}'.format(origin_h, origin_w))

    trans_size = origin_w if origin_w >= origin_h else origin_h
    # trans_size = 1024

    # img = torch.ones(3, tran_size, tran_size)
    # 如果想要填充是黑色则注释掉上一句，换下面这一句
    img = torch.zeros(3, trans_size, trans_size)
    mask_res = torch.zeros(3, trans_size, trans_size)

    img[:, :origin_h, :origin_w] = src
    mask_res[:, :origin_h, :origin_w] = mask

    # img = to_pil_image(img)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    # 保存填充后的图片
    to_pil_image(img).save(result_img_path)
    to_pil_image(mask_res).save(result_mask_path)


# 去除图片中多余的黑色区域
def eliminate_area_0(image_path, result_img_path, mask_path, result_mask_path):
    # 打开图像
    # C, H, W
    src = Image.open(image_path)
    mask = Image.open(mask_path)

    # plt.imshow(src, cmap='gray')
    # plt.show()

    src = to_tensor(src)
    mask = to_tensor(mask)

    origin_h, origin_w = src.shape[1], src.shape[2]
    print('原图像大小, height: {}, width: {}'.format(origin_h, origin_w))

    start_row = 0
    for row in range(0, origin_h):
        if not torch.equal(src[:, row, :][0], torch.zeros(origin_w)):
            start_row = row
            break

    end_row = origin_h
    for row in range(start_row + 1, origin_h):
        if torch.equal(src[:, row, :][0], torch.zeros(origin_w)):
            end_row = row
            break

    start_col = 0
    for col in range(0, origin_w):
        if not torch.equal(src[:, :, col][0], torch.zeros(origin_h)):
            start_col = col
            break

    end_col = origin_w
    for col in range(start_col + 1, origin_w):
        if torch.equal(src[:, :, col][0], torch.zeros(origin_h)):
            print(src[:, :, col][0])
            print(torch.zeros(origin_h))
            end_col = col
            break

    print(start_row, end_row, start_col, end_col)
    # 如果想要填充是黑色则注释掉上一句，换下面这一句
    img = torch.zeros(3, end_row - start_row, end_col - start_col)
    mask_res = torch.zeros(3, end_row - start_row, end_col - start_col)

    img[:, :, :] = src[:, start_row : end_row, start_col : end_col]
    mask_res[:, :, :] = mask[:, start_row : end_row, start_col : end_col]
    # img = to_pil_image(img)

    to_pil_image(img).save(result_img_path)
    to_pil_image(mask_res).save(result_mask_path)

    # 保存填充后的图片
    # plt.imshow(img, cmap='gray')
    # plt.show()

# 图像输出后我们需要clip一下
def clip_unpadding(input_png, output_png, origin_h, origin_w):
    # C, H, W
    img = Image.open(input_png)
    img = to_tensor(img)
    img = img[:, :origin_h, :origin_w]
    # 保存裁剪后的图片
    to_pil_image(img).save(output_png)



if __name__ == '__main__':
    root = r'/home/zdd2020/ZDD_paper/VinDr_RibCXR/train'
    res_root = r'/home/zdd2020/Desktop/xyl/dataset/train_square'

    res_img_path = os.path.join(res_root, 'img')
    res_mask_path = os.path.join(res_root, 'mask')
    mkdir(res_img_path)
    mkdir(res_mask_path)

    img_path = os.path.join(root, 'img')
    file_list = get_filePath(img_path)

    count = 0
    for file_name in file_list:
        count += 1
        print(count, file_name)
        image_file = os.path.join(root, 'img', file_name)
        mask_file = os.path.join(root, 'mask', file_name)

        res_img_file = os.path.join(res_img_path, file_name)
        res_mask_file = os.path.join(res_mask_path, file_name)

        get_padding_pic_mask(image_file, res_img_file, mask_file, res_mask_file)


    # temp = r'/home/zdd2020/Desktop/xyl/dataset/mg/test/image/MCUCXR_0041_0.png'
    # temp_mask = r'/home/zdd2020/Desktop/xyl/dataset/mg/test/lung/MCUCXR_0041_0.png'
    # # get_padding_pic_mask(temp, '', '')
    # eliminate_area_0(temp, './result.png', temp_mask, './mask_result.png')
    # # img = default_loader(temp, IorM='L')
    # #
    # # plt.imshow(img, cmap='gray')
    # plt.show()

     # 先改变大小
    # w, h = img.size

    # s = w if w < h else h
    #
    # img = img.resize((s, s))
    #
    # plt.imshow(img, cmap='gray')
    # plt.show()