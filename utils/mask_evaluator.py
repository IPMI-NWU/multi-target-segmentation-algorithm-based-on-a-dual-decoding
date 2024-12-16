
import torch

import torch.utils.data as Data
from LossAndEval import dice_coef, precision, recall, iou_score, acc, f1score, get_sensitivity, get_specificity
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")


'''
    作用：可以根据保存的预测图像与金标准计算指标
'''

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


# 保存图像, path为保存图片的父父目录，reg_result_name为4个类，img_name为test中的文件名
def save_img(output, img_name, fold, dir_name):
    path = r'../../result/' + dir_name

    mkdir(path)

    path = os.path.join(path, 'fold_' + str(fold))
    mkdir(path)

    seg_result_name = ['all_bone', 'prerib', 'clavicel', 'postrib']

    for name_index in range(len(seg_result_name)):
        result_path = os.path.join(path, seg_result_name[name_index])
        mkdir(result_path)

        bone = getResult(output[seg_result_name[name_index]])

        w, h = 512, 512
        plt.figure(figsize=(w, h), dpi=1)
        # img = Image.open(img)
        plt.imshow(bone, cmap='gray')

        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        # 图片的路径
        save_img_path = os.path.join(result_path, img_name)
        plt.savefig(save_img_path, format='png', transparent=True, dpi=1, pad_inches=0)


def getResult(bone):
    bone = bone.squeeze(0).squeeze(0).cpu()
    bone = (bone > 0.5).float() * 255

    return bone

# 获取文件夹中的文件名
def get_file_list(path):
    pathDir = os.listdir(path)
    file_list = []

    for file in pathDir:
        file_list.append(file)
    return file_list

# 计算指标和存储图像
def infer(mask_pre_root, outTxtPath):
    # img_true_root = r'../../dataset/Montgomery/lung_512_square/3_fold/test/lung'
    # img_true_root = r'../../dataset/bone/mask/pre_rib'
    # img_true_root = r'../../dataset/jsrt/image_mask/test/clavicle'
    img_true_root = r'../../dataset/VinDr_RibCXR_square/val/mask'

    # 从文件夹中获取文件
    file_list = get_file_list(mask_pre_root)

    dice = 0
    iou = 0
    precision_ = 0
    recall_ = 0
    acc_ = 0
    sensitivity = 0
    specificity = 0


    for file_name in file_list:
        from PIL import Image
        pre_mask = os.path.join(mask_pre_root, file_name)
        gt_mask = os.path.join(img_true_root, file_name)

        # 读入图像
        pre_mask = Image.open(pre_mask).convert('L')
        gt_mask = Image.open(gt_mask).convert('L')

        # resize
        h = 512
        w = 512

        toTensor = transforms.ToTensor()

        pre_mask = pre_mask.resize((h, w))
        gt_mask = gt_mask.resize((h, w))

        pre_mask = toTensor(pre_mask)
        gt_mask = toTensor(gt_mask)

        pre_mask = (pre_mask > 0.5).float()
        gt_mask = (gt_mask > 0.5).float()

        dice += dice_coef(pre_mask, gt_mask, True)
        iou += iou_score(pre_mask, gt_mask)
        precision_ += precision(pre_mask, gt_mask)
        recall_ += recall(pre_mask, gt_mask)
        acc_ += acc(pre_mask, gt_mask)
        sensitivity += get_sensitivity(pre_mask, gt_mask)
        specificity += get_specificity(pre_mask, gt_mask)
        # temp_pre = precision(pre_mask, gt_mask)
        # temp_rec = recall(pre_mask, gt_mask)
        # f1_score_ += computeQualityMeasures(pre_mask, gt_mask)
        # print(dice)
        # print(f1_score_)

        shuchu = file_name +  ' dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4} acc:{:.4}'. \
                format(dice_coef(pre_mask, gt_mask, True), precision(pre_mask, gt_mask), recall(pre_mask, gt_mask), iou_score(pre_mask, gt_mask), acc(pre_mask, gt_mask))

        with open(outTxtPath, 'a+') as shuchuTxt:
            shuchuTxt.write(shuchu + '\n')
        print(shuchu)

    dice = dice / len(file_list)
    iou = iou / len(file_list)
    precision_ = precision_ / len(file_list)
    recall_ = recall_ / len(file_list)
    acc_ = acc_ / len(file_list)
    sensitivity = sensitivity / len(file_list)
    specificity = specificity / len(file_list)
    # f1_score_ = f1_score_ / len(file_list)

    print('dice: ', dice)
    print('iou: ', iou)
    print('precision_', precision_)
    print('recall_', recall_)
    print('acc_', acc_)
    print('sensitivity', sensitivity)
    print('specificity', specificity)
    # print(f1_score_)

    shuchu = 'avg dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4} acc:{:.4} sensitivity:{:.4} specificity:{:.4}'. \
        format(dice, precision_, recall_,
               iou, acc_, sensitivity, specificity)

    with open(outTxtPath, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')
    print(shuchu)



if __name__ == '__main__':

    # mask_pre_root = r'../../result/bone5_others3_split_mg_mg/bone_modeljsrt/jsrt/lung'
    mask_pre_root = r'../../result_bone5_JSRT3_vindr_ribcxr_r_v_j_m/bone5_others3_split_vindr_ribcxr_r_j_v_m/vindr_ribcxr_fold_0/bone/vindr_rib'

    outTxtPath = r'./vindr_rib_r_j_v_m.txt'

    mask_pre_path = mask_pre_root

    shuchu = "****************" + '\n'  + 'vindr_rib 第1折' + '\n' + "****************"
    with open(outTxtPath, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')
    print(shuchu)

    infer(mask_pre_path, outTxtPath)
