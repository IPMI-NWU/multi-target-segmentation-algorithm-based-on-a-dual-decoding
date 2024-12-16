from train.cfg import Model_CFG

# 读数据
from read_data.MyDataset3_more_porcess_no_contour_bone import MyDataset
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


def train(net, epoch, trainloader, optimizer, outTxtPath):
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
    dice = [0, 0, 0, 0]  # 4个部分分别的dice系数
    label_num = 4  # mask的数量

    iters = 10
    for iter in range(0, iters):
        for step, (imgs, mask, _) in enumerate(
                trainloader):
            imgs = imgs.float()

            # print(imgs.shape)
            # print(mask.shape)

            for mask_index in range(len(mask)):
                mask[mask_index] = mask[mask_index].float()

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                for mask_index in range(len(mask)):
                    mask[mask_index] = mask[mask_index].cuda()

            optimizer.zero_grad()
            output1, output2, output3, output4 = net(imgs, mask)

            loss = net.loss
            train_loss += loss.data.cpu().item()

            loss.backward()
            optimizer.step()

            # 将预测的结果用sigmoid激活，变为0~1之间的数，与gt计算指标
            masks_probs = []
            masks_probs_binary = []
            masks_probs.append(F.sigmoid(output1))
            masks_probs.append(F.sigmoid(output2))
            masks_probs.append(F.sigmoid(output3))
            masks_probs.append(F.sigmoid(output4))

            for i in range(label_num):
                masks_probs_binary.append((masks_probs[i] > 0.5).float())

            # 真实的label
            true_mask_binary = []
            for mask_index in range(len(mask)):
                true_mask_binary.append((mask[mask_index] > 0.5).float())

            mydice_labels = []
            for i in range(label_num):
                mydice_labels.append(dice_coef(masks_probs_binary[i], true_mask_binary[i]))
                dice[i] = dice[i] + mydice_labels[i]

            # avg_dice = (dice[0] + dice[1] + dice[2] + dice[3]) / 4
            # 每8个step输出一次
            if step % 8 == 0:
                shuchu = 'Epoch: {:d} iter: {:d} step: {:d}  Train loss: {:.6f} dice_label: {:.6f} dice_labelClavicel: {:.6f} dice_labelPosteriorrib: {:.6f} dice_labelPrerib: {:.6f} lr:{:.12f}'.format(
                    epoch, (iter + 1), step, loss.data.cpu().item(), mydice_labels[0], mydice_labels[1],
                    mydice_labels[2], mydice_labels[3], optimizer.param_groups[0]["lr"])
                print(shuchu)
                with open(outTxtPath, 'a+') as shuchuTxt:
                    shuchuTxt.write(shuchu + '\n')

    # 一个epoch的dice
    for i in range(label_num):
        dice[i] = dice[i] / ((step + 1) * iters)

    # print(train_loss / (step + 1))
    shuchu = 'Epoch: {:d} Train Loss: {:.6f} avg_dice: {:.6f} dice_label: {:.6f} dice_labelClavicel: {:.6f} dice_labelPosteriorrib: {:.6f} dice_labelPrerib: {:.6f}'.format(
        epoch,
        train_loss / ((step + 1) * iters), (dice[0] + dice[1] + dice[2] + dice[3]) / label_num, dice[0],
        dice[1], dice[2], dice[3])

    with open(outTxtPath, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')
        shuchuTxt.write('\n')
    print(shuchu)
    print()

    # 返回值，一个epoch的train_loss, dice
    return train_loss / ((step + 1) * iters), dice


def val(net, epoch, testloader, outTxtPath):
    net.eval()
    test_loss = 0
    # dice, iou(jaccard), precision, recall = 0, 0, 0, 0
    dice = [0, 0, 0, 0]
    iou = [0, 0, 0, 0]
    precision_ = [0, 0, 0, 0]
    recall_ = [0, 0, 0, 0]

    mask_num = 4
    labels_name = ['all_bone', 'calvicel', 'postrib', 'prerib']
    with torch.no_grad():
        for step, (imgs, mask, _) in enumerate(testloader):
            imgs = imgs.float()

            for mask_index in range(len(mask)):
                mask[mask_index] = mask[mask_index].float()

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                for mask_index in range(len(mask)):
                    mask[mask_index] = mask[mask_index].cuda()

            output1, output2, output3, output4 = net(imgs, mask)

            masks_probs = []
            masks_probs_binary = []
            masks_probs.append(F.sigmoid(output1))
            masks_probs.append(F.sigmoid(output2))
            masks_probs.append(F.sigmoid(output3))
            masks_probs.append(F.sigmoid(output4))

            for i in range(4):
                masks_probs_binary.append((masks_probs[i] > 0.5).float())

            # 真实的label
            true_mask_binary = []
            for mask_index in range(len(mask)):
                true_mask_binary.append((mask[mask_index] > 0.5).float())

            for i in range(mask_num):
                dice[i] = dice[i] + dice_coef(masks_probs_binary[i], true_mask_binary[i])
                iou[i] = iou[i] + iou_score(masks_probs_binary[i], true_mask_binary[i])
                precision_[i] = precision_[i] + precision(masks_probs_binary[i], true_mask_binary[i])
                recall_[i] = recall_[i] + recall(masks_probs_binary[i], true_mask_binary[i])

            loss = net.loss
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


def one_step(train_img_label_txt, test_img_label_txt, zhe, outTxtPath, model_path, file_name, model):
    # 基础设置
    cfg = Model_CFG
    lr = cfg['lr']
    epochs = cfg['epochs']
    batch_size = cfg['bn_size']
    batch_oftest = cfg['batch_oftest']

    label_num = 4

    # 创建一个可视化的文件夹
    visual = r'../../compare_experiment/visual/4_3_have_data_aug'
    visual = os.path.join(visual, file_name)

    mkdir(visual)

    # 可视化
    # 数据扩充与损失的
    write_loss_test = SummaryWriter(log_dir=os.path.join(visual, 'Loss' + str(zhe + 1) + '_runs', 'test_loss'))
    write_loss_train = SummaryWriter(log_dir=os.path.join(visual, 'Loss' + str(zhe + 1) + '_runs', 'train_loss'))
    write_dice_train = SummaryWriter(log_dir=os.path.join(visual, 'Dice' + str(zhe + 1) + '_runs', 'train_Dice'))
    write_dice_test = SummaryWriter(log_dir=os.path.join(visual, 'Dice' + str(zhe + 1) + '_runs', 'test_Dice'))

    write_dice_test1 = SummaryWriter(log_dir=os.path.join(visual, 'Test' + str(zhe + 1) + '_runs', 'test_Dice'))
    write_pre_test = SummaryWriter(log_dir=os.path.join(visual, 'Test' + str(zhe + 1) + '_runs', 'test_pre'))
    write_recall_test = SummaryWriter(log_dir=os.path.join(visual, 'Test' + str(zhe + 1) + '_runs', 'test_recall'))
    write_iou_test = SummaryWriter(log_dir=os.path.join(visual, 'Test' + str(zhe + 1) + '_runs', 'test_iou'))

    # 跑一折
    with open(outTxtPath, 'a+') as resultTxt:
        shuchu = '------------------------------------第' + str(zhe + 1) + '折------------------------------------'
        print(shuchu)
        resultTxt.write('\n\n' + shuchu + '\n')

    # 读取数据的方式，mode 为 test，就是不为其设置数据增强
    train_datasets = MyDataset(train_img_label_txt, mode='train')
    trainloader = Data.DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, num_workers=0)

    test_datasets = MyDataset(test_img_label_txt, mode='test')
    testloader = Data.DataLoader(dataset=test_datasets, batch_size=batch_oftest, shuffle=False, num_workers=0)

    # net
    net = model

    if torch.cuda.is_available():
        net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)

    with open(outTxtPath, 'a+') as resultTxt:
        shuchu = 'lr:' + str(lr) + '\tepochs:' + str(epochs) + '\tbatch_size:' + str(batch_size)
        print(shuchu)
        resultTxt.write('\n\n' + shuchu + '\n')

    # 统计参数量
    # total, total_MB, trainable_num = get_parameter_number(net)
    # print('total params: %.6f' % total)
    # print('total_MB params: %.6fMB' % total_MB)
    # print('trainable_num: %.6f' % trainable_num)

    best_dice = 0
    base_lr = lr
    for epoch in range(epochs):
        train_loss, train_dice = train(net, epoch, trainloader, optimizer, outTxtPath)
        # 调整学习率
        # lr = adjust_learning_rate_poly(optimizer, epoch, epochs, base_lr, power=0.9)
        scheduler.step()

        # print(optimizer.param_groups[0]["lr"])
        test_dice, iou, precision_, recall_, test_loss = val(net, epoch, testloader, outTxtPath)  # 根据dice保存的模型
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

        with open(outTxtPath, 'a+') as shuchuTxt:
            shuchuTxt.write('avg_test_dice:' + str(test_dice_) + '\n')
            shuchuTxt.write('\n')
            print('avg_test_dice:', test_dice_)
            print()

        # 可视化
        write_loss_train.add_scalar('Loss', train_loss, epoch)
        write_dice_train.add_scalar('Avg_Dice', train_dice_, epoch)

        write_loss_test.add_scalar('Loss', test_loss, epoch)

        write_dice_test.add_scalar('Avg_Dice', test_dice_, epoch)

        write_dice_test1.add_scalar('Avg_Test', test_dice_, epoch)
        # 平均dice与其他评估指标
        write_iou_test.add_scalar('Avg_Test', test_iou_ / label_num, epoch)
        write_pre_test.add_scalar('Avg_Test', test_percision_ / label_num, epoch)
        write_recall_test.add_scalar('Avg_Test', test_recall_ / label_num, epoch)

        # 可视化每一个分割结果的dice
        r_name = ['all_bone', 'Clavicel', 'Postrib', 'Prerib']
        for i in range(label_num):
            write_dice_train.add_scalar('TrainAndTestDice_' + r_name[i], train_dice[i], epoch)
            write_dice_test.add_scalar('TrainAndTestDice_' + r_name[i], test_dice[i], epoch)

            # 各自的指标
            write_dice_test1.add_scalar('Test_' + r_name[i], test_dice[i], epoch)
            write_iou_test.add_scalar('Test_' + r_name[i], iou[i], epoch)
            write_pre_test.add_scalar('Test_' + r_name[i], precision_[i], epoch)
            write_recall_test.add_scalar('Test_' + r_name[i], recall_[i], epoch)

        savenumpy = ('{:3f}'.format(test_dice_))
        dir = model_path + '/dense_resblock' + str(epoch) + '-' + str(savenumpy) + '.pth'
        if test_dice_ >= best_dice and test_dice_ > 0.82:
            # torch.save(net.state_dict(), dir)
            torch.save(net, dir)
            best_dice = test_dice_

        elif epoch % 10 == 0 or epoch == (epochs - 1):
            # torch.save(net.state_dict(), dir)
            torch.save(net, dir)

    write_dice_train.close()
    write_dice_test.close()
    write_loss_train.close()
    write_loss_test.close()
    write_iou_test.close()
    write_recall_test.close()
    write_pre_test.close()
    write_dice_test1.close()


# 无数据增强
def main1():
    from dense_class.encoder_densenet_121_bone_AMFS126 import densenet121_resDecoder

    model = densenet121_resDecoder(pretrained=True, num_classes_d1=4)
    file_name = '4_3_dense_bone_have_aug'

    return file_name, model

if __name__ == "__main__":
    # 文档的输出位置
    outModelPath_dir = r'../../compare_experiment/OUT/no_contour/dense'
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

            outModelPath = os.path.join(outModelPath_dir, file_name)
            mkdir(outModelPath)

            num_fold_path = os.path.join(outModelPath, 'fold_' + str(i))
            mkdir(num_fold_path)

            train_img_label_txt = r'../../dataset/bone_fake_data/txt_lihangsheng_fake_data_20/fold' + str(
                i) + '/train.txt'
            test_img_label_txt = r'../../dataset/bone_fake_data/txt_lihangsheng_fake_data_20/fold' + str(
                i) + '/test.txt'

            # 训练网络
            outTxtPath = num_fold_path + '/no_contour.txt'

            with open(outTxtPath, 'a+') as resultTxt:
                dt = datetime.now()  # 创建一个datetime类对象
                resultTxt.write('\n\n' + 'start time:' + dt.strftime('%y-%m-%d %I:%M:%S %p') + '\n')
                print(dt.strftime('%y-%m-%d %I:%M:%S %p'))

                one_step(train_img_label_txt, test_img_label_txt, i, outTxtPath, num_fold_path, file_name, model)

                dt = datetime.now()
                resultTxt.write("end time:" + dt.strftime('%y-%m-%d %I:%M:%S %p') + "\n")
