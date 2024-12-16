import torch

# 读数据
from read_data.MyDataset3_more_porcess_no_contour_bone_deep_supervision import MyDataset

import torch.utils.data as Data
from LossAndEval import dice_coef, precision, recall, iou_score

import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


# 保存图像, path为保存图片的父父目录，reg_result_name为4个类，img_name为test中的文件名
def save_img(output, img_name, fold, dir_name, task_type):
    path = r'../../result_bone5_JSRT3_vindr_ribcxr_r_v_j_m/' + dir_name + ''

    mkdir(path)

    path = os.path.join(path, 'vindr_ribcxr_fold_' + str(fold), task_type)
    mkdir(path)

    seg_result_name = ['all_bone', 'clavicel', 'post_rib', 'pre_rib', 'vindr_rib', 'lung', 'heart', 'jsrtClavicel']

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


# 计算指标和存储图像
def infer(testloader, outTxtPath, fold, model_index, dir_name, type='bone'):
    # 导入模型
    model_pth = [
        # 0
        [
            [
                'r_j_v_m/vindr_ribcxr/model_vindr_rib_132-0.893601.pth',
            ],
        ],
    ]

    task_type = 0

    model_path = os.path.join(r'../../model_pth', model_pth[model_index][0][task_type])


    net = torch.load(model_path)
    net.eval()

    # labels_name = ['所有骨', '锁骨', '后肋', '前肋']
    labels_name = ['all_bone', 'clavicel', 'post_rib', 'pre_rib', 'vindr_rib', 'lung', 'heart', 'jsrtClavicel']
    with torch.no_grad():
        for step, (imgs, mask_map1, mask_map2, mask_map3, mask, file_name) in enumerate(testloader):
            imgs = imgs.float()
            # 前肋， 后肋， 锁骨， 肺实质， 心脏
            for i in range(len(mask)):
                mask[i] = mask[i].float()

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                for i in range(len(mask)):
                    mask[i] = mask[i].cuda()

            [output_d1, output_d2] = net(imgs, mode='test')  #
            output_dict = {}
            for i in range(len(labels_name)):
                if i < 5:
                    output_dict[labels_name[i]] = output_d2[-1][:, i, :, :]  # bone
                else:
                    output_dict[labels_name[i]] = output_d1[-1][:, i - 5, :, :] # jsrt

            img_name = file_name[0].split('/')[-1]
            # print(img_name)
            # save_img(output_dict, img_name, type_task, dir_name, type)

            masks_probs = []
            masks_probs_binary = []
            for i in range(len(labels_name)):
                if i < 5:
                    masks_probs.append(F.sigmoid(output_d2[-1][:, i, :, :]))
                else:
                    masks_probs.append(F.sigmoid(output_d1[-1][:, i - 5, :, :]))

            for i in range(len(labels_name)):
                masks_probs_binary.append((masks_probs[i] > 0.5).float())

            for i in range(len(labels_name)):
                output_dict[labels_name[i]] = masks_probs_binary[i]
            save_img(output_dict, img_name, fold, dir_name, type)


if __name__ == '__main__':
    num = 1
    model_index = 0  # 8
    txt_path = r''

    shuchu_list = \
        [
            # 0
            'bone5_others3_split_vindr_ribcxr_r_j_v_m',
        ]

    shuchu = shuchu_list[model_index]
    print(shuchu)


    for i in range(0, num):
        shuchu = "=" * 50 + '\n' + "第" + str(i + 1) + "折" + '\n' + "=" * 50
        print(shuchu)

        type_task = 'bone'

        test_img_label_txt = ''
        if type_task == 'jsrt':
            test_img_label_txt = r''
        if type_task == 'bone':
            test_img_label_txt = r'../../dataset/VinDr_RibCXR_square/txt_VinDr_RibCXR' + '/val.txt'

        # 使用Mydataset读入数据
        test_datasets = MyDataset(test_img_label_txt, mode='test')
        testloader = Data.DataLoader(dataset=test_datasets, batch_size=1, shuffle=False, num_workers=0)

        infer(testloader, txt_path, i, model_index, shuchu_list[model_index], type=type_task)
