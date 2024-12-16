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
    for i in range(len(mask)):
        # print(output.shape)
        loss_mask = loss_mask + criterion2(output[:, i, :, :].squeeze(), mask[i].squeeze())

    return loss_mask / len(mask)

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


def train(net, epoch, trainloader, optimizer, outTxtPath, type='bone'):
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
    net.train()
    train_loss = 0

    dice = []
    if type == 'bone':
        dice = [0, 0, 0, 0]  # 4个部分分别的dice系数
    if type == 'jsrt':
        dice = [0, 0, 0]

    iters = 1
    if type == 'bone':
        iters = 10 # 10
    if type == 'jsrt':
        iters = 3

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
            output = net(imgs, mode='train', task_type=type) # 2为肋骨

            loss = 0
            if type == 'bone':
                loss = 0
                # bone_weight = [0.1, 0.4, 0.6, 1]
                bone_weight = [1, 1, 1, 1]
                for loss_index in range(1, len(output)):
                    loss += bone_weight[loss_index] * compute_multi_loss(output[loss_index], mask_total[loss_index])
                    # loss = compute_multi_loss_dice_focal_bone(output_d2, mask)
            if type == 'jsrt':
                # loss = compute_multi_loss_dice_focal(output_d1, mask)
                loss = 0
                for loss_index in range(1, len(output)):
                    loss += compute_multi_loss(output[loss_index], mask_total[loss_index])

            train_loss += loss.data.cpu().item()

            loss.backward()
            optimizer.step()

            # 将预测的结果用sigmoid激活，变为0~1之间的数，与gt计算指标
            masks_probs = []
            masks_probs_binary = []
            for index_m in range(len(mask)):
                masks_probs.append(F.sigmoid(output[-1][:, index_m, :, :]))


            for i in range(len(mask)):
                masks_probs_binary.append((masks_probs[i] > 0.5).float())

            # 真实的label
            true_mask_binary = []
            for mask_index in range(len(mask)):
                true_mask_binary.append((mask[mask_index] > 0.5).float())

            mydice_labels = []
            for i in range(len(mask)):
                mydice_labels.append(dice_coef(masks_probs_binary[i], true_mask_binary[i]))
                dice[i] = dice[i] + mydice_labels[i]

            # avg_dice = (dice[0] + dice[1] + dice[2] + dice[3]) /
            # 每8个step输出一次
            shuchu = ''
            if type == 'bone' and step % 8 == 0:
                shuchu = 'Epoch: {:d} iter: {:d} step: {:d}  Train loss: {:.6f} dice_labelBone: {:.6f} dice_labelClavicel: {:.6f}  dice_labelPostrib: {:.6f} dice_Prerib: {:.6f} lr:{:.12f}'.format(
                    epoch, (iter + 1), step, loss.data.cpu().item(), mydice_labels[0], mydice_labels[1],
                    mydice_labels[2], mydice_labels[3], optimizer.param_groups[0]["lr"])

            if type == 'jsrt' and step % 8 == 0:
                shuchu = 'Epoch: {:d} iter: {:d} step: {:d}  Train loss: {:.6f} dice_labellung: {:.6f} dice_labelheart: {:.6f} dice_clavicel: {:.6f} lr:{:.12f}'.format(
                    epoch, (iter + 1), step, loss.data.cpu().item(), mydice_labels[0], mydice_labels[1], mydice_labels[2], optimizer.param_groups[0]["lr"])


            if step % 8 == 0:
                print(shuchu)
                with open(outTxtPath, 'a+') as shuchuTxt:
                    shuchuTxt.write(shuchu + '\n')

    # 一个epoch的dice
    for i in range(label_num):
        dice[i] = dice[i] / ((step + 1) * iters)

    # print(train_loss / (step + 1))
    if type == 'bone':
        shuchu = 'Epoch: {:d} Train Loss: {:.6f} avg_dice: {:.6f}  dice_labelPrerib: {:.6f} dice_labelPosteriorrib: {:.6f} dice_labelClavicel: {:.6f} dice_all_bone: {:.6f}  lr:{:.12f}'.format(
            epoch,
            train_loss / ((step + 1) * iters), (dice[0] + dice[1] + dice[2] + dice[3]) / label_num, dice[0],
            dice[1], dice[2], dice[3], optimizer.param_groups[0]["lr"])

    if type == 'jsrt':
        shuchu = 'Epoch: {:d} Train Loss: {:.6f} avg_dice: {:.6f}  dice_labellung: {:.6f} dice_labelheart: {:.6f} dice_clavicel: {:.6f} lr:{:.12f}'.format(
            epoch,
            train_loss / ((step + 1) * iters), (dice[0] + dice[1] + dice[2]) / label_num, dice[0],
            dice[1], dice[2], optimizer.param_groups[0]["lr"])

    with open(outTxtPath, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')
        shuchuTxt.write('\n')
    print(shuchu)
    print()

    # 返回值，一个epoch的train_loss, dice
    return train_loss / ((step + 1) * iters), dice


def val(net, epoch, testloader, outTxtPath, type='bone'):
    net.eval()
    test_loss = 0
    # dice, iou(jaccard), precision, recall = 0, 0, 0, 0
    dice = []
    iou = []
    precision_ = []
    recall_ = []
    labels_name = []

    if type == 'bone':
        dice = [0, 0, 0, 0]
        iou = [0, 0, 0, 0]
        precision_ = [0, 0, 0, 0]
        recall_ = [0, 0, 0, 0]
        # labels_name = ['prerib', 'postrib', 'calvicel', 'all_bone']
        labels_name = ['bone', 'clavicle', 'post_rib', 'pre_rib']
    if type == 'jsrt':
        dice = [0, 0, 0]
        iou = [0, 0, 0]
        precision_ = [0, 0, 0]
        recall_ = [0, 0, 0]
        labels_name = ['lung', 'heart', 'clavicel']

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

            output_p = output_d1[-1].clone()
            if type == 'bone':
                output_p = output_d2[-1].clone()

            if type == 'jsrt':
                output_p = output_d1[-1].clone()

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
                dice[i] = dice[i] + dice_coef(masks_probs_binary[i], true_mask_binary[i])
                iou[i] = iou[i] + iou_score(masks_probs_binary[i], true_mask_binary[i])
                precision_[i] = precision_[i] + precision(masks_probs_binary[i], true_mask_binary[i])
                recall_[i] = recall_[i] + recall(masks_probs_binary[i], true_mask_binary[i])

            loss = compute_multi_loss(output_p, mask)
            test_loss += loss.data.cpu().item()

        for i in range(mask_num):
            dice[i] = dice[i] / (step + 1)
            iou[i] = iou[i] / (step + 1)
            precision_[i] = precision_[i] / (step + 1)
            recall_[i] = recall_[i] / (step + 1)

        shuchu = 'Epoch: {:d} Test Loss: {:.6f}'.format(epoch, test_loss / (step + 1))
        with open(outTxtPath, 'a+') as shuchuTxt:
            shuchuTxt.write(shuchu + '\n')
        print(shuchu)

        for i in range(mask_num):
            shuchu = labels_name[i] + ' dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4} '. \
                format(dice[i], precision_[i], recall_[i], iou[i])
            with open(outTxtPath, 'a+') as shuchuTxt:
                shuchuTxt.write(shuchu + '\n')
            print(shuchu)

        return dice, iou, precision_, recall_, test_loss / (step + 1)


def one_step(i, outTxtPath_bone,  outTxtPath_jsrt, num_fold_path_bone, num_fold_path_jsrt, file_name, model):
    # 基础设置
    cfg = Model_CFG
    lr = cfg['lr']
    epochs = cfg['epochs']
    batch_size = cfg['bn_size']
    batch_oftest = cfg['batch_oftest']

    fold_num = i

    # 创建一个可视化的文件夹
    visual = r'../../compare_experiment/visual'
    visual_bone = os.path.join(visual, file_name, 'fold_' + str(i), 'bone')
    visual_jsrt = os.path.join(visual, file_name, 'fold_' + str(i), 'jsrt')
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80, 100], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=0.1)

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
    type_task = 'bone'
    train_img_label_txt = r''
    test_img_label_txt = r''

    label_num = 0
    for epoch in range(epochs):
        if epoch % 2 == 0:
            type_task = 'bone'
            label_num = 4
            train_img_label_txt = r'../../dataset/bone/txt_' + str(fold_num) + '/train.txt'
            test_img_label_txt = r'../../dataset/bone/txt_' + str(fold_num) + '/test.txt'
        if epoch % 2 == 1:
            type_task = 'jsrt'
            label_num = 3
            train_img_label_txt = r'../../dataset/jsrt/txt_3organ_train158' + '/train.txt'
            test_img_label_txt = r'../../dataset/jsrt/txt_3organ_train158' + '/test.txt'


        # 读取数据的方式，mode 为 test，就是不为其设置数据增强
        train_datasets = MyDataset(train_img_label_txt, mode='train')
        trainloader = Data.DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        test_datasets = MyDataset(test_img_label_txt, mode='test')
        testloader = Data.DataLoader(dataset=test_datasets, batch_size=batch_oftest, shuffle=False, num_workers=0, drop_last=True)

        if epoch % 2 == 0:
            train_loss, train_dice = train(net, epoch, trainloader, optimizer, outTxtPath_bone, type_task)
        else:
            train_loss, train_dice = train(net, epoch, trainloader, optimizer, outTxtPath_jsrt, type_task)
        # 调整学习率
        # lr = adjust_learning_rate_poly(optimizer, epoch, epochs, base_lr, power=0.9)
        scheduler.step()

        # print(optimizer.param_groups[0]["lr"])
        if epoch % 2 == 0:
            test_dice, iou, precision_, recall_, test_loss = val(net, epoch, testloader, outTxtPath_bone, type_task)  # 根据dice保存的模型
        else:
            print(type_task)
            test_dice, iou, precision_, recall_, test_loss = val(net, epoch, testloader, outTxtPath_jsrt, type_task)

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
        if epoch % 2 == 0 and test_dice_ >= best_dice_bone and test_dice_ > 0.82:
            # torch.save(net.state_dict(), dir)
            dir = num_fold_path_bone + '/model_bone_' + str(epoch) + '-' + str(savenumpy) + '.pth'
            torch.save(net, dir)
            best_dice_bone = test_dice_

        if epoch % 2 == 1 and test_dice_ >= best_dice_jsrt and test_dice_ > 0.82:
            # torch.save(net.state_dict(), dir)
            dir = num_fold_path_jsrt + '/model_jsrt_' + str(epoch) + '-' + str(savenumpy) + '.pth'
            torch.save(net, dir)
            best_dice_jsrt = test_dice_

        if epoch % 10 == 0:
            # torch.save(net.state_dict(), dir)
            dir = num_fold_path_bone + '/model_epoch_' + str(epoch) + '-' + str(savenumpy) + '.pth'
            torch.save(net, dir)
        if (epoch + 1) % 10 == 0:
            dir = num_fold_path_jsrt + '/model_epoch_' + str(epoch) + '-' + str(savenumpy) + '.pth'
            torch.save(net, dir)

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

    model = densenet121_resDecoder(pretrained=True, num_classes_d1=3, num_classes_d2=4)
    file_name = '2023_3_9_have_trad_10_3_gengzheng_bone4_jsrt3_denseEn_resDe_two_decoder_1center_denseblock_4fold_bone_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_deep_supervision_map2_map3_weight_std_split'
    return file_name, model


if __name__ == "__main__":
    # 文档的输出位置
    outModelPath_dir = r'../../compare_experiment/OUT/no_contour/bone_jsrt_dense_class_baseline_xiaorong_gengzheng'
    mkdir(outModelPath_dir)

    zhe = 4

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
            num_fold_path_jsrt = os.path.join(outModelPath, 'jsrt')
            mkdir(num_fold_path_jsrt)

            # 训练网络
            outTxtPath_bone = num_fold_path_bone + '/bone.txt'
            outTxtPath_jsrt = num_fold_path_jsrt + '/jsrt.txt'

            with open(outTxtPath_bone, 'a+') as resultTxt_bone, open(outTxtPath_jsrt, 'a+') as resultTxt_jsrt:
                dt = datetime.now()  # 创建一个datetime类对象
                resultTxt_bone.write('\n\n' + 'start time:' + dt.strftime('%y-%m-%d %I:%M:%S %p') + '\n')
                resultTxt_jsrt.write('\n\n' + 'start time:' + dt.strftime('%y-%m-%d %I:%M:%S %p') + '\n')
                print(dt.strftime('%y-%m-%d %I:%M:%S %p'))

                one_step(i, outTxtPath_bone,  outTxtPath_jsrt, num_fold_path_bone, num_fold_path_jsrt, file_name, model)

                dt = datetime.now()
                resultTxt_bone.write("end time:" + dt.strftime('%y-%m-%d %I:%M:%S %p') + "\n")
                resultTxt_jsrt.write("end time:" + dt.strftime('%y-%m-%d %I:%M:%S %p') + "\n")
