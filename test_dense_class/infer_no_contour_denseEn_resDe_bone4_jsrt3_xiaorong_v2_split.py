import torch

# 读数据
from read_data.MyDataset3_more_porcess_no_contour_bone import MyDataset

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
    path = r'../../result_xiaorong_gengzheng_bone_binary/' + dir_name + ''

    mkdir(path)

    path = os.path.join(path, 'bone_model' + str(fold), task_type)
    mkdir(path)

    seg_result_name = ['prerib', 'postrib', 'clavicle', 'all_bone', 'lung', 'heart', 'jsrtClavicle']

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
        # 0 baseline
        [
            [
                'baseline/bone/fold_0/model_bone_80-0.880636.pth',
                'baseline/jsrt/model_jsrt_53-0.950068.pth'
            ],
            [
                'baseline/bone/fold_1/model_bone_82-0.865489.pth',
                'baseline/jsrt/model_jsrt_53-0.950068.pth'
            ],
            [
                'baseline/bone/fold_2/model_bone_96-0.868411.pth',
                'baseline/jsrt/model_jsrt_53-0.950068.pth'
            ],
            [
                'baseline/bone/fold_3/model_bone_80-0.862396.pth',
                'baseline/jsrt/model_jsrt_53-0.950068.pth'
            ],
        ],

        # 1 baseline + 2CAS + 2AG
        [
            [
                'baseline_cas_ag/bone/fold_0/model_bone_58-0.902267.pth',
                'baseline_cas_ag/jsrt/model_jsrt_43-0.951961.pth'
            ],
            [
                'baseline_cas_ag/bone/fold_1/model_bone_74-0.894483.pth',
                'baseline_cas_ag/jsrt/model_jsrt_43-0.951961.pth'
            ],
            [
                'baseline_cas_ag/bone/fold_2/model_bone_66-0.897732.pth',
                'baseline_cas_ag/jsrt/model_jsrt_43-0.951961.pth'
            ],
            [
                'baseline_cas_ag/bone/fold_3/model_bone_58-0.894103.pth',
                'baseline_cas_ag/jsrt/model_jsrt_43-0.951961.pth'
            ],
        ],

        # 2 baseline + 2CAS + 2AG + bone_AMFS + jsrt+AMFS137
        [
            [
                'baseline_cas_ag_amfs/bone/fold_0/model_bone_42-0.904494.pth',
                'baseline_cas_ag_amfs/jsrt/model_epoch_59-0.953616.pth'
            ],
            [
                'baseline_cas_ag_amfs/bone/fold_1/model_bone_72-0.895742.pth',
                'baseline_cas_ag_amfs/jsrt/model_epoch_59-0.953616.pth'
            ],
            [
                'baseline_cas_ag_amfs/bone/fold_2/model_bone_74-0.898047.pth',
                'baseline_cas_ag_amfs/jsrt/model_epoch_59-0.953616.pth'
            ],
            [
                'baseline_cas_ag_amfs/bone/fold_3/model_bone_66-0.892978.pth',
                'baseline_cas_ag_amfs/jsrt/model_epoch_59-0.953616.pth'
            ],
        ],

        # 3 baseline + 2CAS + 2AG + bone_AMFS + jsrt+AMFS137 + aug
        [
            [
                'baseline_cas_ag_amfs_aug/bone/fold_0/model_bone_44-0.906042.pth',
                'baseline_cas_ag_amfs_aug/jsrt/model_jsrt_81-0.955277.pth'
            ],
            [
                'baseline_cas_ag_amfs_aug/bone/fold_1/model_bone_48-0.897802.pth',
                'baseline_cas_ag_amfs_aug/jsrt/model_jsrt_81-0.955277.pth'
            ],
            [
                'baseline_cas_ag_amfs_aug/bone/fold_2/model_bone_50-0.901866.pth',
                'baseline_cas_ag_amfs_aug/jsrt/model_jsrt_81-0.955277.pth'
            ],
            [
                'baseline_cas_ag_amfs_aug/bone/fold_3/model_bone_58-0.898649.pth',
                'baseline_cas_ag_amfs_aug/jsrt/model_jsrt_81-0.955277.pth'
            ],
        ],
    ]

    task_type = 0
    if type == 'bone':
        task_type = 0
    if type == 'jsrt':
        task_type = 1

    model_path = os.path.join(r'../../model_pth', model_pth[model_index][fold][task_type])


    net = torch.load(model_path)
    net.eval()

    # print(net)
    # with open(outTxtPath, 'a+') as shuchuTxt:
    #     shuchuTxt.write(str(net) + '\n')

    test_loss = 0
    # dice, iou(jaccard), precision, recall = 0, 0, 0, 0
    dice = [0, 0, 0, 0, 0, 0, 0]
    iou = [0, 0, 0, 0, 0, 0, 0]
    precision_ = [0, 0, 0, 0, 0, 0, 0]
    recall_ = [0, 0, 0, 0, 0, 0, 0]

    mask_num = len(dice)
    # labels_name = ['所有骨', '锁骨', '后肋', '前肋']
    labels_name = ['prerib', 'postrib', 'clavicle', 'all_bone', 'lung', 'heart', 'jsrtClavicle']
    with torch.no_grad():
        for step, (imgs, mask, file_name) in enumerate(testloader):
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
                if i < 4:
                    output_dict[labels_name[i]] = output_d2[:, i, :, :]  # bone
                else:
                    output_dict[labels_name[i]] = output_d1[:, i - 4, :, :] # jsrt

            img_name = file_name[0].split('/')[-1]
            # print(img_name)
            # save_img(output_dict, img_name, fold, dir_name, type)

            masks_probs = []
            masks_probs_binary = []
            for i in range(len(labels_name)):
                if i < 4:
                    masks_probs.append(F.sigmoid(output_d2[:, i, :, :]))
                else:
                    masks_probs.append(F.sigmoid(output_d1[:, i - 4, :, :]))

            for i in range(len(labels_name)):
                masks_probs_binary.append((masks_probs[i] > 0.5).float())

            for i in range(len(labels_name)):
                output_dict[labels_name[i]] = masks_probs_binary[i]
            save_img(output_dict, img_name, fold, dir_name, type)

            # 真实的label
            true_mask_binary = []
            for i in range(len(mask)):
                true_mask_binary.append((mask[i] > 0.5).float())

            img_dice = img_name + ':\t'
            # 计算指标
            if type == 'bone':
                for i in range(len(mask)):
                    dice[i] = dice[i] + dice_coef(masks_probs_binary[i], true_mask_binary[i])
                    iou[i] = iou[i] + iou_score(masks_probs_binary[i], true_mask_binary[i])
                    precision_[i] = precision_[i] + precision(masks_probs_binary[i], true_mask_binary[i])
                    recall_[i] = recall_[i] + recall(masks_probs_binary[i], true_mask_binary[i])

                    # 每副图像的 dice
                    # img_dice = img_dice + labels_name[i] + '_dice: {:.4}'.format(
                    #     dice_coef(masks_probs_binary[i], true_mask_binary[i]).data.cpu().item()) + '\t'
                    # str(dice_coef(masks_probs_binary[i], true_mask_binary[i]).data.cpu().item()) + '\t'
                    #
                    # with open(txt_path, 'a+') as shuchuDice:
                    #     shuchuDice.write(img_dice + '\n')
                    # print(img_dice)

            bone_num = 4
            if type == 'jsrt':
                for i in range(len(mask)):
                    dice[i + bone_num] = dice[i + bone_num] + dice_coef(masks_probs_binary[i + bone_num], true_mask_binary[i])
                    iou[i + bone_num] = iou[i + bone_num] + iou_score(masks_probs_binary[i + bone_num], true_mask_binary[i])
                    precision_[i + bone_num] = precision_[i + bone_num] + precision(masks_probs_binary[i + bone_num], true_mask_binary[i])
                    recall_[i + bone_num] = recall_[i + bone_num] + recall(masks_probs_binary[i + bone_num], true_mask_binary[i])

    # 平均的指标
    for i in range(len(labels_name)):
        dice[i] = dice[i] / (step + 1)
        iou[i] = iou[i] / (step + 1)
        precision_[i] = precision_[i] / (step + 1)
        recall_[i] = recall_[i] / (step + 1)


    for i in range(len(labels_name)):
        shuchu = labels_name[i] + ' dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4}'. \
            format(dice[i], precision_[i], recall_[i], iou[i])
        with open(outTxtPath, 'a+') as shuchuTxt:
            shuchuTxt.write(shuchu + '\n')
        print(shuchu)

    dice_total = 0
    iou_total = 0
    precision_total = 0
    recall_total = 0
    for i in range(mask_num):
        dice_total = dice[i] + dice_total
        iou_total = iou_total + iou[i]
        precision_total = precision_total + precision_[i]
        recall_total = recall_total + recall_[i]

    shuchu = ''
    if type == 'bone':
        shuchu = 'average: ' + 'dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4}'. \
            format(dice_total / bone_num, precision_total / bone_num, recall_total / bone_num, iou_total / bone_num)
    if type == 'jsrt':
        temp_num = len(labels_name) - bone_num
        shuchu = 'average: ' + 'dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4}'. \
            format(dice_total / temp_num, precision_total / temp_num, recall_total / temp_num, iou_total / temp_num)

    print(shuchu)
    with open(outTxtPath, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')


if __name__ == '__main__':
    num = 4
    model_index = 2  # 8
    txt_path = r'./3_10_aug_three_new_denseEn_resDe_xiaorong_v3.txt'

    shuchu_list = \
        [
            # 0
            'baseline',
            # 1
            'baseline_2CAS_jsrt_2AG',
            # 2
            'baseline_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137',
            # 3
            'baseline_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_aug',
        ]

    shuchu = shuchu_list[model_index] + '修正hou数据'
    print(shuchu)

    with open(txt_path, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')

    for i in range(0, num):
        shuchu = "=" * 50 + '\n' + "第" + str(i + 1) + "折" + '\n' + "=" * 50
        with open(txt_path, 'a+') as shuchuTxt:
            shuchuTxt.write(shuchu + '\n')

        print(shuchu)

        type_task = 'bone'

        if type_task == 'jsrt':
            test_img_label_txt = r'../../dataset/' + type_task + '/txt_3organ_train158/' + 'test.txt'
        if type_task == 'bone':
            test_img_label_txt = r'../../dataset/' + type_task + '/txt_' + str(i) + '/test.txt'

        # 使用Mydataset读入数据
        test_datasets = MyDataset(test_img_label_txt, mode='test')
        testloader = Data.DataLoader(dataset=test_datasets, batch_size=1, shuffle=False, num_workers=0)

        infer(testloader, txt_path, i, model_index, shuchu_list[model_index], type=type_task)
