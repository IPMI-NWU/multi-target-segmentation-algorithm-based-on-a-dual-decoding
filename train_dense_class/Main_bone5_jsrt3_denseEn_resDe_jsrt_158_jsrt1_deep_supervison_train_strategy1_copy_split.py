from train.cfg import Model_CFG

# 读数据
from read_data.MyDataset3_more_porcess_no_contour_bone_deep_supervision import MyDataset
# from read_data.MyDataset3_more_porcess_no_contour_rgb import MyDataset
# from read_data.MyDataset3_sobel.MyDataset3_more_porcess_no_contour import MyDataset  # sobel

import torch.utils.data as Data
import torch
import torch.nn.functional as F
from torch import optim
from utils.LossAndEval import dice_coef, precision, recall, iou_score
from torchvision import transforms
import os
from datetime import datetime
from tensorboardX import SummaryWriter

from utils.write_image2txt import mkdir
import warnings

warnings.filterwarnings("ignore")

transform = transforms.Compose([transforms.ToTensor(), ])

# 计算损失
def compute_multi_loss(output, mask):
    from utils.LossAndEval import DiceLoss
    criterion2 = DiceLoss()

    loss_mask = 0
    loss_mask = loss_mask + criterion2(output.squeeze(), mask.squeeze())

    return loss_mask

# 计算损失
def compute_multi_loss_dice_bce(output, mask):
    from utils.LossAndEval import DiceLoss, BCELoss
    criterion2 = DiceLoss()
    criterion3 = BCELoss()

    loss_mask = criterion2(output, mask) + criterion3(output, mask)

    return loss_mask

# 计算损失
def compute_multi_loss_dice_focal(output, mask):
    from utils.LossAndEval import DiceLoss, FocalLoss
    criterion2 = DiceLoss()
    criterion3 = FocalLoss()

    loss_mask = 0
    for i in range(len(mask)):
        # print(output.shape)
        if i == 1:
            loss_mask += criterion2(output[:, i, :, :].squeeze(), mask[i].squeeze()) + criterion3(output[:, i, :, :].squeeze(), mask[i].squeeze())
        else:
            loss_mask = loss_mask + criterion2(output[:, i, :, :].squeeze(), mask[i].squeeze())
            # loss_mask += compute_multi_loss_dice_bce(output[:, i, :, :].squeeze(), mask[i].squeeze())

    return loss_mask / len(mask)

# 计算损失
def compute_multi_loss_dice_focal_bone(output, mask):
    from utils.LossAndEval import DiceLoss, FocalLoss
    criterion2 = DiceLoss()
    criterion3 = FocalLoss()

    loss_mask = 0
    for i in range(len(mask)):
        # print(output.shape)
        if i == 0:
            loss_mask += criterion2(output[:, i, :, :].squeeze(), mask[i].squeeze()) + criterion3(output[:, i, :, :].squeeze(), mask[i].squeeze())
        else:
            loss_mask = loss_mask + criterion2(output[:, i, :, :].squeeze(), mask[i].squeeze())
            # loss_mask += compute_multi_loss_dice_bce(output[:, i, :, :].squeeze(), mask[i].squeeze())

    return loss_mask / len(mask)

# 调整学习率
def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# 获取网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    total_num_MB = total_num / ((1024 * 1024) * 4)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # return {'Total': total_num, 'Total_MB': total_num_MB, 'Trainable': trainable_num}
    return total_num, total_num_MB, trainable_num


def train(net, epoch, trainloader, optimizer, outTxtPath, type='bone', task=None):
    '''
    函数定义：
    :param net: 训练的网络模型
    :param epoch: 训练迭代的第几个epoch
    :param trainloader: 训练数据
    :param optimizer:  优化器，如Adam
    :param outTxtPath: 训练过程输出的txt文件的路径
    :param write_loss: 可视化损失
    :param write_dice: 可视化dice
    '''
    if task is None:
        task = ['all_bone', 'clavicel', 'post_rib', 'pre_rib']
    net.train()
    train_loss = 0

    task_total_list = ['all_bone', 'clavicel', 'post_rib', 'pre_rib', 'vindr_rib', 'lung', 'heart', 'jsrtClavicel']
    dice = [0, 0, 0, 0, 0, 0, 0, 0]

    # 获取 taks 在 task_total_list 中的位置，就知道 dice 中哪些位置需要更新了
    task_index = []
    for task_ in task:
        task_index.append(task_total_list.index(task_))
    # 对 task_index 从小到大排序
    task_index.sort()
    print(task_index)

    iters = 1
    if type == 'bone':
        if len(task_index) == 1:
            iters = 5  # vindr
        else:
            iters = 10  # rcs-cxr
    if type == 'others':
        if len(task_index) == 1:
            iters = 10   # mg
        else:
            iters = 5  #jsrt

    label_num = 0
    for iter in range(0, iters):
        for step, (imgs, mask_map1, mask_map2, mask_map3, mask, _) in enumerate(trainloader):
            label_num = len(mask)

            imgs = imgs.float()

            # print(imgs.shape)
            # print(mask.shape)
            for mask_index in range(len(mask)):
                mask[mask_index] = mask[mask_index].float()
                mask_map1[mask_index] = mask_map1[mask_index].float()
                mask_map2[mask_index] = mask_map2[mask_index].float()
                mask_map3[mask_index] = mask_map3[mask_index].float()

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                for mask_index in range(len(mask)):
                    mask[mask_index] = mask[mask_index].cuda()
                    mask_map1[mask_index] = mask_map1[mask_index].cuda()
                    mask_map2[mask_index] = mask_map2[mask_index].cuda()
                    mask_map3[mask_index] = mask_map3[mask_index].cuda()

            mask_total = [mask_map1, mask_map2, mask_map3, mask]
            optimizer.zero_grad()
            output = net(imgs, mode='train',  task_type=type) # 2为肋骨, output_d1 3 , output_d2 5

            loss = 0
            if type == 'bone':
                # loss = 0
                bone_weight = [0.1, 0.4, 0.6, 1]
                # bone_weight = [1, 1, 1, 1]
                for loss_index in range(1, len(output)):  # 从1开始就是没有map1
                    if len(task_index) > 1:
                        for index_t in task_index: # 不同的官
                            if index_t == 1:
                                loss += bone_weight[loss_index] * 1.5 * compute_multi_loss(
                                    output[loss_index][:, index_t, :, :], mask_total[loss_index][index_t])
                            else:
                                loss += bone_weight[loss_index] * compute_multi_loss(output[loss_index][:, index_t, :, :], mask_total[loss_index][index_t])
                                # loss = compute_multi_loss_dice_focal_bone(output_d2, mask)

                    else:
                        loss = bone_weight[loss_index] * compute_multi_loss(output[loss_index][:, task_index[0], :, :], mask_total[loss_index][0])
                        # loss = compute_multi_loss(output[loss_index][:, task_index[0], :, :],
                        #                                                     mask_total[loss_index][0])
                loss /= len(mask)

            if type == 'others':
                # loss = compute_multi_loss_dice_focal(output_d1, mask)
                # loss = 0
                for loss_index in range(1, len(output)):
                    if len(task_index) > 1:
                        for index_t in task_index: # 不同的器官
                            if index_t == 6:
                                # loss += 2 * compute_multi_loss(output[loss_index][:, index_t - 5, :, :], mask_total[loss_index][index_t - 5])
                                # loss = compute_multi_loss_dice_focal_bone(output_d2, mask)
                                loss += compute_multi_loss(output[loss_index][:, index_t - 5, :, :],
                                                               mask_total[loss_index][index_t - 5])
                            else:
                                loss += compute_multi_loss(output[loss_index][:, index_t - 5, :, :],
                                                           mask_total[loss_index][index_t - 5])
                    else:
                        loss = compute_multi_loss(output[loss_index][:, task_index[0] - 5, :, :], mask_total[loss_index][0])
                loss /= len(mask)

            train_loss += loss.data.cpu().item()

            loss.backward()
            optimizer.step()

            # 将预测的结果用sigmoid激活，变为0~1之间的数，与gt计算指标
            masks_probs = []
            masks_probs_binary = []

            if type == 'bone':
                if len(task_index) > 1:
                    for index_t in task_index:
                        masks_probs.append(F.sigmoid(output[-1][:, index_t, :, :]))
                else:
                    masks_probs.append(F.sigmoid(output[-1][:, task_index[0], :, :]))

            if type == 'others':
                if len(task_index) > 1:
                    for index_t in task_index:
                        masks_probs.append(F.sigmoid(output[-1][:, index_t - 5, :, :]))
                else:
                    masks_probs.append(F.sigmoid(output[-1][:, task_index[0] - 5, :, :]))


            for i in range(len(mask)):
                masks_probs_binary.append((masks_probs[i] > 0.5).float())

            # 真实的label
            true_mask_binary = []
            for mask_index in range(len(mask)):
                true_mask_binary.append((mask[mask_index] > 0.5).float())

            mydice_labels = []
            for i in range(len(mask)):
                mydice_labels.append(dice_coef(masks_probs_binary[i], true_mask_binary[i]))
                dice[task_index[i]] = dice[task_index[i]] + mydice_labels[i]

            # avg_dice = (dice[0] + dice[1] + dice[2] + dice[3]) /
            # 每8个step输出一次
            shuchu = 'Epoch: {:d} iter: {:d} step: {:d}  Train loss: {:.6f} '.format(epoch, (iter + 1), step, loss.data.cpu().item())

            if type == 'bone' and step % 8 == 0:
                if len(task_index) > 1:
                    for index_t in task_index:
                        shuchu += '  dice_' + task_total_list[index_t] + ': {:.6f}'.format(mydice_labels[index_t])
                else:
                    shuchu += '  dice_' + task_total_list[task_index[0]] + ': {:.6f}'.format(mydice_labels[0])

            if type == 'others' and step % 8 == 0:
                if len(task_index) > 1:
                    for index_t in task_index:
                        shuchu += '  dice_' + task_total_list[index_t] + ': {:.6f}'.format(mydice_labels[index_t - 5])
                else:
                    shuchu += '  dice_' + task_total_list[task_index[0]] + ': {:.6f}'.format(mydice_labels[0])

            shuchu += '  lr:{:.12f}'.format(optimizer.param_groups[0]["lr"])

            if step % 8 == 0:
                print(shuchu)
                with open(outTxtPath, 'a+') as shuchuTxt:
                    shuchuTxt.write(shuchu + '\n')

    # 一个epoch的dice
    for i in range(label_num):
        dice[task_index[i]] = dice[task_index[i]] / ((step + 1) * iters)


    avg_dice = 0
    for index_t in task_index:
        avg_dice += dice[index_t]
    avg_dice /= len(task_index)

    # print(train_loss / (step + 1))
    shuchu = 'Epoch: {:d} Train Loss: {:.6f} avg_dice: {:.6f}'.format(epoch, train_loss / ((step + 1) * iters), avg_dice)
    for index_t in task_index:
        shuchu += '  dice_' + task_total_list[index_t] + ': {:.6f}'.format(dice[index_t])
    shuchu += '  lr:{:.12f}'.format(optimizer.param_groups[0]["lr"])

    with open(outTxtPath, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')
        shuchuTxt.write('\n')
    print(shuchu)
    print()

    # 返回值，一个epoch的train_loss, dice
    res_dice = []
    for index_t in task_index:
        res_dice.append(dice[index_t])

    return train_loss / ((step + 1) * iters), res_dice


def val(net, epoch, testloader, outTxtPath, type='bone', task=None):
    if task is None:
        task = ['all_bone', 'clavicel', 'post_rib', 'pre_rib']
    net.eval()
    test_loss = 0

    # dice, iou(jaccard), precision, recall = 0, 0, 0, 0
    task_total_list = ['all_bone', 'clavicel', 'post_rib', 'pre_rib', 'vindr_rib', 'lung', 'heart', 'jsrtClavicel']
    dice = [0, 0, 0, 0, 0, 0, 0, 0]
    iou = [0, 0, 0, 0, 0, 0, 0, 0]
    precision_ = [0, 0, 0, 0, 0, 0, 0, 0]
    recall_ = [0, 0, 0, 0, 0, 0, 0, 0]

    # 获取 taks 在 task_total_list 中的位置，就知道 dice 中哪些位置需要更新了
    task_index = []
    for task_ in task:
        task_index.append(task_total_list.index(task_))
    # 对 task_index 从小到大排序
    task_index.sort()

    print(task_index)

    mask_num = 0
    with torch.no_grad():
        for step, (imgs, mask_map1, mask_map2, mask_map3, mask, _) in enumerate(testloader):
            mask_num = len(mask)
            imgs = imgs.float()

            for mask_index in range(len(mask)):
                mask[mask_index] = mask[mask_index].float()

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                for mask_index in range(len(mask)):
                    mask[mask_index] = mask[mask_index].cuda()


            [output_d1, output_d2] = net(imgs, mode='test')

            output_p = torch.zeros(output_d2[-1].shape[0], len(mask), output_d2[-1].shape[-2],
                                   output_d2[-1].shape[-1]).cuda()
            if type == 'bone':
                count_p = 0
                for index_t in task_index:
                    # print(task_index)
                    output_p[:, count_p, :, :] = output_d2[-1][:, index_t, :, :].clone()
                    count_p += 1

            if type == 'others':
                count_p = 0
                for index_t in task_index:
                    # print(task_index)
                    output_p[:, count_p, :, :] = output_d1[-1][:, index_t - 5, :, :].clone()
                    count_p += 1

            masks_probs = []
            masks_probs_binary = []

            for index_m in range(len(mask)):
                masks_probs.append(F.sigmoid(output_p[:, index_m, :, :]))

            for i in range(len(mask)):
                masks_probs_binary.append((masks_probs[i] > 0.5).float())

            # 真实的label
            true_mask_binary = []
            for mask_index in range(len(mask)):
                true_mask_binary.append((mask[mask_index] > 0.5).float())

            for i in range(len(mask)):
                dice[task_index[i]] = dice[task_index[i]] + dice_coef(masks_probs_binary[i], true_mask_binary[i])
                iou[task_index[i]] = iou[task_index[i]] + iou_score(masks_probs_binary[i], true_mask_binary[i])
                precision_[task_index[i]] = precision_[task_index[i]] + precision(masks_probs_binary[i], true_mask_binary[i])
                recall_[task_index[i]] = recall_[task_index[i]] + recall(masks_probs_binary[i], true_mask_binary[i])

            loss = 0
            for i in range(len(mask)):
                loss = compute_multi_loss(output_p[:, i, :, :], mask[i])

            loss /= len(mask)

            test_loss += loss.data.cpu().item()

        for i in range(mask_num):
            dice[task_index[i]] = dice[task_index[i]] / (step + 1)
            iou[task_index[i]] = iou[task_index[i]] / (step + 1)
            precision_[task_index[i]] = precision_[task_index[i]] / (step + 1)
            recall_[task_index[i]] = recall_[task_index[i]] / (step + 1)

        shuchu = 'Epoch: {:d} Test Loss: {:.6f}'.format(epoch, test_loss / (step + 1))
        with open(outTxtPath, 'a+') as shuchuTxt:
            shuchuTxt.write(shuchu + '\n')
        print(shuchu)

        for i in range(mask_num):
            shuchu = task_total_list[task_index[i]] + ' dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4} '. \
                format(dice[task_index[i]], precision_[task_index[i]], recall_[task_index[i]], iou[task_index[i]])
            with open(outTxtPath, 'a+') as shuchuTxt:
                shuchuTxt.write(shuchu + '\n')
            print(shuchu)

        res_dice = []
        res_iou = []
        res_precision = []
        res_recall = []
        for index_t in task_index:
            res_dice.append(dice[index_t])
            res_iou.append(iou[index_t])
            res_precision.append(precision_[index_t])
            res_recall.append(recall_[index_t])

        return res_dice, res_iou, res_precision, res_recall, test_loss / (step + 1)


def one_step(i, outTxtPath_bone,  outTxtPath_jsrt, num_fold_path_bone, num_fold_path_jsrt, file_name, model):
    # 基础设置
    global task
    cfg = Model_CFG
    lr = cfg['lr']
    epochs = cfg['epochs']
    batch_size = cfg['bn_size']
    batch_oftest = cfg['batch_oftest']

    fold_num = i

    # 创建一个可视化的文件夹
    visual = r'../../compare_experiment/visual'
    visual_bone = os.path.join(visual, file_name, 'fold_' + str(i), 'bone')
    visual_jsrt = os.path.join(visual, file_name, 'fold_' + str(i), 'others')
    mkdir(visual_bone)
    mkdir(visual_jsrt)

    # 可视化
    # 数据扩充与损失的
    write_loss_test_bone = SummaryWriter(log_dir=os.path.join(visual_bone, 'Loss' + str(fold_num + 1) + '_runs', 'test_loss'))
    write_loss_train_bone = SummaryWriter(log_dir=os.path.join(visual_bone, 'Loss' + str(fold_num + 1) + '_runs', 'train_loss'))
    write_dice_train_bone = SummaryWriter(log_dir=os.path.join(visual_bone, 'Dice' + str(fold_num + 1) + '_runs', 'train_Dice'))
    write_dice_test_bone = SummaryWriter(log_dir=os.path.join(visual_bone, 'Dice' + str(fold_num + 1) + '_runs', 'test_Dice'))

    write_dice_test1_bone = SummaryWriter(log_dir=os.path.join(visual_bone, 'Test' + str(fold_num + 1) + '_runs', 'test_Dice'))
    write_pre_test_bone = SummaryWriter(log_dir=os.path.join(visual_bone, 'Test' + str(fold_num + 1) + '_runs', 'test_pre'))
    write_recall_test_bone = SummaryWriter(log_dir=os.path.join(visual_bone, 'Test' + str(fold_num + 1) + '_runs', 'test_recall'))
    write_iou_test_bone = SummaryWriter(log_dir=os.path.join(visual_bone, 'Test' + str(fold_num + 1) + '_runs', 'test_iou'))

    # jsrt 的
    write_loss_test_jsrt = SummaryWriter(log_dir=os.path.join(visual_jsrt, 'Loss' + str(fold_num + 1) + '_runs', 'test_loss'))
    write_loss_train_jsrt = SummaryWriter(log_dir=os.path.join(visual_jsrt, 'Loss' + str(fold_num + 1) + '_runs', 'train_loss'))
    write_dice_train_jsrt = SummaryWriter(log_dir=os.path.join(visual_jsrt, 'Dice' + str(fold_num + 1) + '_runs', 'train_Dice'))
    write_dice_test_jsrt = SummaryWriter(log_dir=os.path.join(visual_jsrt, 'Dice' + str(fold_num + 1) + '_runs', 'test_Dice'))

    write_dice_test1_jsrt = SummaryWriter(log_dir=os.path.join(visual_jsrt, 'Test' + str(fold_num + 1) + '_runs', 'test_Dice'))
    write_pre_test_jsrt = SummaryWriter(log_dir=os.path.join(visual_jsrt, 'Test' + str(fold_num + 1) + '_runs', 'test_pre'))
    write_recall_test_jsrt = SummaryWriter(log_dir=os.path.join(visual_jsrt, 'Test' + str(fold_num + 1) + '_runs', 'test_recall'))
    write_iou_test_jsrt = SummaryWriter(log_dir=os.path.join(visual_jsrt, 'Test' + str(fold_num + 1) + '_runs', 'test_iou'))

    # 跑一折
    with open(outTxtPath_bone, 'a+') as resultTxt_bone, open(outTxtPath_jsrt, 'a+') as resultTxt_jsrt:
        shuchu = '------------------------------------第' + str(fold_num + 1) + '折------------------------------------'
        print(shuchu)
        resultTxt_bone.write('\n\n' + shuchu + '\n')
        resultTxt_jsrt.write('\n\n' + shuchu + '\n')

    # net
    net = model

    if torch.cuda.is_available():
        net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80, 100], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100, 140], gamma=0.1)

    with open(outTxtPath_bone, 'a+') as resultTxt_bone, open(outTxtPath_jsrt, 'a+') as resultTxt_jsrt:
        shuchu = 'lr:' + str(lr) + '\tepochs:' + str(epochs) + '\tbatch_size:' + str(batch_size)
        print(shuchu)
        resultTxt_bone.write('\n\n' + shuchu + '\n')
        resultTxt_jsrt.write('\n\n' + shuchu + '\n')

    # 统计参数量
    # total, total_MB, trainable_num = get_parameter_number(net)
    # print('total params: %.6f' % total)
    # print('total_MB params: %.6fMB' % total_MB)
    # print('trainable_num: %.6f' % trainable_num)

    best_dice_bone = 0
    best_dice_jsrt = 0
    best_dice_mg = 0
    best_dice_vindr_rib = 0
    type_task = 'bone'
    train_img_label_txt = r''
    test_img_label_txt = r''

    label_num = 0
    for epoch in range(epochs):
        #
        if epoch % 4 == 1:
            type_task = 'bone'
            label_num = 4
            task = ['all_bone', 'clavicel', 'post_rib', 'pre_rib'] # ['pre_rib', 'post_rib', 'clavicel', 'all_bone']
            train_img_label_txt = r'../../dataset/bone_fake_data/txt_lihangsheng_fake_data_20/fold' + str(fold_num) + '/train.txt'
            test_img_label_txt = r'../../dataset/bone_fake_data/txt_lihangsheng_fake_data_20/fold' + str(fold_num) + '/test.txt'
        if epoch % 4 == 2:
            type_task = 'others'
            label_num = 3
            task = ['lung', 'heart', 'jsrtClavicel']
            train_img_label_txt = r'../../dataset/jsrt/txt_3organ_train158_aug' + '/train.txt'
            test_img_label_txt = r'../../dataset/jsrt/txt_3organ_train158_aug' + '/test.txt'
        if epoch % 4 == 0:
            type_task = 'bone'
            label_num = 1
            task = ['vindr_rib']
            train_img_label_txt = r'../../dataset/VinDr_RibCXR_square/txt_VinDr_RibCXR' + '/train.txt'
            test_img_label_txt = r'../../dataset/VinDr_RibCXR_square/txt_VinDr_RibCXR' + '/val.txt'
        if epoch % 4 == 3:
            type_task = 'others'
            label_num = 1
            task = ['lung']
            train_img_label_txt = r'../../dataset/mg_square/txt_mg' + '/train_val.txt'
            test_img_label_txt = r'../../dataset/mg_square/txt_mg' + '/test.txt'
            # if fold_num < 3:
            #     train_img_label_txt = r'../../dataset/Montgomery/lung_512_square/' + str(fold_num+1) + '_fold/txt_mg' + '/train.txt'
            #     test_img_label_txt = r'../../dataset/Montgomery/lung_512_square/' + str(fold_num+1) + '_fold/txt_mg' + '/test.txt'
            # else:
            #     train_img_label_txt = r'../../dataset/Montgomery/lung_512_square/' + str(
            #         fold_num) + '_fold/txt_mg' + '/train.txt'
            #     test_img_label_txt = r'../../dataset/Montgomery/lung_512_square/' + str(
            #         fold_num) + '_fold/txt_mg' + '/test.txt'

        # 读取数据的方式，mode 为 test，就是不为其设置数据增强
        train_datasets = MyDataset(train_img_label_txt, mode='train')
        trainloader = Data.DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        test_datasets = MyDataset(test_img_label_txt, mode='test')
        testloader = Data.DataLoader(dataset=test_datasets, batch_size=batch_oftest, shuffle=False, num_workers=0, drop_last=True)


        if epoch % 2 == 0:
            train_loss, train_dice = train(net, epoch, trainloader, optimizer, outTxtPath_bone, type_task, task)
        else:
            train_loss, train_dice = train(net, epoch, trainloader, optimizer, outTxtPath_jsrt, type_task, task)
        # 调整学习率
        # lr = adjust_learning_rate_poly(optimizer, epoch, epochs, base_lr, power=0.9)
        scheduler.step()

        # print(optimizer.param_groups[0]["lr"])
        if epoch % 2 == 0:
            test_dice, iou, precision_, recall_, test_loss = val(net, epoch, testloader, outTxtPath_bone, type_task, task)  # 根据dice保存的模型
        else:
            print(type_task)
            test_dice, iou, precision_, recall_, test_loss = val(net, epoch, testloader, outTxtPath_jsrt, type_task, task)

        test_dice_ = 0  # 平均dice
        test_iou_ = 0
        test_percision_ = 0
        test_recall_ = 0

        train_dice_ = 0
        for i in range(label_num):
            train_dice_ += train_dice[i].cpu().numpy()
            test_dice_ += test_dice[i].cpu().numpy()
            test_iou_ += iou[i]
            test_percision_ += precision_[i]
            test_recall_ += recall_[i]

        train_dice_ = train_dice_ / label_num  # 四个部分平均的train_dice
        test_dice_ = test_dice_ / label_num

        if epoch % 2 == 0:
            # 可视化
            write_loss_train_bone.add_scalar('Loss', train_loss, epoch)
            write_dice_train_bone.add_scalar('Avg_Dice', train_dice_, epoch)

            write_loss_test_bone.add_scalar('Loss', test_loss, epoch)

            write_dice_test_bone.add_scalar('Avg_Dice', test_dice_, epoch)

            write_dice_test1_bone.add_scalar('Avg_Test', test_dice_, epoch)
            # 平均dice与其他评估指标
            write_iou_test_bone.add_scalar('Avg_Test', test_iou_ / label_num, epoch)
            write_pre_test_bone.add_scalar('Avg_Test', test_percision_ / label_num, epoch)
            write_recall_test_bone.add_scalar('Avg_Test', test_recall_ / label_num, epoch)

            with open(outTxtPath_bone, 'a+') as shuchuTxt_bone:
                shuchuTxt_bone.write('avg_test_dice:' + str(test_dice_) + '\n')
                shuchuTxt_bone.write('\n')
                print('avg_test_dice:', test_dice_)
                print()

        if epoch % 2 == 1:
            # 可视化
            write_loss_train_jsrt.add_scalar('Loss', train_loss, epoch)
            write_dice_train_jsrt.add_scalar('Avg_Dice', train_dice_, epoch)

            write_loss_test_jsrt.add_scalar('Loss', test_loss, epoch)

            write_dice_test_jsrt.add_scalar('Avg_Dice', test_dice_, epoch)

            write_dice_test1_jsrt.add_scalar('Avg_Test', test_dice_, epoch)
            # 平均dice与其他评估指标
            write_iou_test_jsrt.add_scalar('Avg_Test', test_iou_ / label_num, epoch)
            write_pre_test_jsrt.add_scalar('Avg_Test', test_percision_ / label_num, epoch)
            write_recall_test_jsrt.add_scalar('Avg_Test', test_recall_ / label_num, epoch)

            with open(outTxtPath_jsrt, 'a+') as shuchuTxt_jsrt:
                shuchuTxt_jsrt.write('avg_test_dice:' + str(test_dice_) + '\n')
                shuchuTxt_jsrt.write('\n')
                print('avg_test_dice:', test_dice_)
                print()


        savenumpy = ('{:3f}'.format(test_dice_))
        # dir = model_path + '/unet_resblock' + str(epoch) + '-' + str(savenumpy) + '.pth'
        if epoch % 4 == 1 and test_dice_ >= best_dice_bone and test_dice_ > 0.82:
            # torch.save(net.state_dict(), dir)
            temp_path = os.path.join(num_fold_path_bone, 'bone')
            mkdir(temp_path)
            dir = temp_path + '/model_bone_' + str(epoch) + '-' + str(savenumpy) + '.pth'
            torch.save(net, dir)
            best_dice_bone = test_dice_

        if epoch % 4 == 0 and test_dice_ >= best_dice_vindr_rib and test_dice_ > 0.82:
            # torch.save(net.state_dict(), dir)
            temp_path = os.path.join(num_fold_path_bone, 'vindr_rib')
            mkdir(temp_path)
            dir = temp_path + '/model_vindr_rib_' + str(epoch) + '-' + str(savenumpy) + '.pth'
            torch.save(net, dir)
            best_dice_vindr_rib = test_dice_


        if epoch % 4 == 2 and test_dice_ >= best_dice_jsrt and test_dice_ > 0.82:
            # torch.save(net.state_dict(), dir)
            temp_path = os.path.join(num_fold_path_jsrt, 'jsrt')
            mkdir(temp_path)
            dir = temp_path + '/model_jsrt_' + str(epoch) + '-' + str(savenumpy) + '.pth'
            torch.save(net, dir)
            best_dice_jsrt = test_dice_

        if epoch % 4 == 3 and test_dice_ >= best_dice_mg and test_dice_ > 0.82:
            # torch.save(net.state_dict(), dir)
            temp_path = os.path.join(num_fold_path_jsrt, 'mg')
            mkdir(temp_path)
            dir = temp_path + '/model_mg_' + str(epoch) + '-' + str(savenumpy) + '.pth'
            torch.save(net, dir)
            best_dice_mg = test_dice_

        # if epoch % 10 == 0:
        #     # torch.save(net.state_dict(), dir)
        #     dir = num_fold_path_bone + '/model_epoch_' + str(epoch) + '-' + str(savenumpy) + '.pth'
        #     torch.save(net, dir)
        # if (epoch + 1) % 10 == 0:
        #     dir = num_fold_path_jsrt + '/model_epoch_' + str(epoch) + '-' + str(savenumpy) + '.pth'
        #     torch.save(net, dir)

    write_dice_train_bone.close()
    write_dice_test_bone.close()
    write_loss_train_bone.close()
    write_loss_test_bone.close()
    write_iou_test_bone.close()
    write_recall_test_bone.close()
    write_pre_test_bone.close()
    write_dice_test1_bone.close()

    write_dice_train_jsrt.close()
    write_dice_test_jsrt.close()
    write_loss_train_jsrt.close()
    write_loss_test_jsrt.close()
    write_iou_test_jsrt.close()
    write_recall_test_jsrt.close()
    write_pre_test_jsrt.close()
    write_dice_test1_jsrt.close()


def main1():
    from dense_class.encoder_densenet_121_1center_bone_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_deep_supervision_map2_map3_weight_std_split import densenet121_resDecoder

    model = densenet121_resDecoder(pretrained=True, num_classes_d1=3, num_classes_d2=5)
    # file_name = '2023_2_17_clavicle1.5_bone5_jsrt3_split_denseEn_resDe_two_decoder_1center_denseblock_4fold_bone_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_deep_supervision_map2_map3_bone_0.6_0.4_1'
    # file_name = '2023_2_20_bone_clavicle1.5_jsrt_heart_1.5_bone5_jsrt3_split_denseEn_resDe_two_decoder_1center_denseblock_4fold_bone_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_deep_supervision_map2_map3_std'
    # file_name = '2023_2_21_bone_clavicle1.5_jsrt_heart_2_bone5_jsrt3_split_denseEn_resDe_two_decoder_1center_denseblock_4fold_bone_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_deep_supervision_map2_map3_std'
    # file_name = '2023_3_10_mg_bone_clavicle1.5_bone5_jsrt3_split_denseEn_resDe_two_decoder_1center_denseblock_4fold_bone_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_deep_supervision_map2_map3_std'
    file_name = '2023_3_14_bone_0.1_0.4_0.6_1_clavicle1.5_bone5_jsrt3_split_denseEn_resDe_two_decoder_1center_denseblock_4fold_bone_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_deep_supervision_map2_map3_bone_0.6_0.4_1'
    return file_name, model


if __name__ == "__main__":
    # 文档的输出位置
    outModelPath_dir = r'../../compare_experiment/OUT/no_contour/bone5_others3_class'
    mkdir(outModelPath_dir)

    zhe = 1

    model_num = [1]
    file_name = ''
    model = ''

    for model_index in model_num:
        # the num of fold
        for i in range(0, zhe):  # 每一折应该是一个新的model,之前是个傻逼吧,竟然用同一个model在跑
            if model_index == 1:
                file_name, model = main1()
            if model_index == 2:
                file_name, model = main2()
            if model_index == 3:
                file_name, model = main3()
            if model_index == 4:
                file_name, model = main4()
            if model_index == 5:
                file_name, model = main5()
            if model_index == 6:
                file_name, model = main6()
            if model_index == 7:
                file_name, model = main7()

            outModelPath = os.path.join(outModelPath_dir, file_name, 'fold_' + str(i))
            mkdir(outModelPath)

            num_fold_path_bone = os.path.join(outModelPath, 'bone')
            mkdir(num_fold_path_bone)
            num_fold_path_jsrt = os.path.join(outModelPath, 'others')
            mkdir(num_fold_path_jsrt)

            # 训练网络
            outTxtPath_bone = num_fold_path_bone + '/bone.txt'
            outTxtPath_jsrt = num_fold_path_jsrt + '/others.txt'

            with open(outTxtPath_bone, 'a+') as resultTxt_bone, open(outTxtPath_jsrt, 'a+') as resultTxt_jsrt:
                dt = datetime.now()  # 创建一个datetime类对象
                resultTxt_bone.write('\n\n' + 'start time:' + dt.strftime('%y-%m-%d %I:%M:%S %p') + '\n')
                resultTxt_jsrt.write('\n\n' + 'start time:' + dt.strftime('%y-%m-%d %I:%M:%S %p') + '\n')
                print(dt.strftime('%y-%m-%d %I:%M:%S %p'))

                one_step(i, outTxtPath_bone,  outTxtPath_jsrt, num_fold_path_bone, num_fold_path_jsrt, file_name, model)

                dt = datetime.now()
                resultTxt_bone.write("end time:" + dt.strftime('%y-%m-%d %I:%M:%S %p') + "\n")
                resultTxt_jsrt.write("end time:" + dt.strftime('%y-%m-%d %I:%M:%S %p') + "\n")
