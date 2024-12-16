import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score, accuracy_score
from torch import nn, Tensor


class RobustCrossEntropyLoss(nn.BCEWithLogitsLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input = F.sigmoid(input)
        # input = input > 0.5
        # target = target > 0.5

        # if len(target.shape) == len(input.shape):
        #     assert target.shape[1] == 1
        #     target = target[:, 0]
        return super().forward(input, target)

class Topkloss(RobustCrossEntropyLoss):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self,  weight=None, ignore_index=-100, k=10, reduce=False):
        self.k = k
        super(Topkloss, self).__init__()

    def forward(self, inp, target):
        target = (target > 0.5).float()
        res = super(Topkloss, self).forward(inp, target)

        print(res)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

# PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU

# 1.基于二分类交叉熵实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        target = (target > 0.5).float()

        return self.bce(input, target)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, weight=1):

        input = F.sigmoid(input)

        return 1-dice_coef(input, target, weight=weight)
        # 注意此处dice系数的计算分为两种形式：总体计算和分开样本计算再平均


'''使用的dice系数分母上有平方'''
class DiceLoss2(nn.Module):
    def __init__(self):
        super(DiceLoss2, self).__init__()

    def forward(self, input, target, weight=1):

        input = F.sigmoid(input)

        return 1-dice_coef(input, target, weight=weight, isPow=True)
        # 注意此处dice系数的计算分为两种形式：总体计算和分开样本计算再平均

class Dice_BCELoss(nn.Module):
    def __init__(self):
        super(Dice_BCELoss, self).__init__()

    def forward(self, input, target, weight=1):
        input_ = F.sigmoid(input)

        dice_coef_loss = dice_coef(input_, target, weight=weight)

        bce = nn.BCEWithLogitsLoss()
        bce_loss = bce(input, target)

        loss = bce_loss - dice_coef_loss
        return loss

def dice_coef(input, target, weight=1, isPow=True):
    smooth = 1e-5
    target = (target > 0.5).float()
    input_flat = input.view(-1)  # 拉平成一列，即第一维度是计算得来的
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat)

    if isPow:
        return (2 * weight * intersection.sum() + smooth) / (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth)
    else:
        return (2 * weight * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)


'''
    jaccard相关系数与iou是一样的
'''
def iou_score(output, target):
    smooth = 1e-5

    output = output.view(-1)
    target = target.view(-1)

    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

'''
    使用混淆矩阵计算precision、recall、jaccard
'''
def precision(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output = output > 0.5
    target = target > 0.5

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    # output_flat = output.view(-1)  # 拉平成一列，即第一维度是计算得来的
    # target_flat = target.view(-1)

    # print(111)
    precision_ = precision_score(y_true=target, y_pred=output)
    # precision_ = precision_score(y_true=target_flat, y_pred=output_flat)
    #
    # print(precision_)
    return precision_

def acc(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output = output > 0.5
    target = target > 0.5

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    acc_ = accuracy_score(y_true=target, y_pred=output)
    return acc_

def recall(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output = output > 0.5
    target = target > 0.5

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    recall_ = recall_score(y_true=target, y_pred=output)

    return recall_

def jaccard(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output = output > 0.5
    target = target > 0.5

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    jaccard_ = jaccard_score(y_true=target, y_pred=output)

    return jaccard_


def sensitivity_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()  # 0,1之间
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (target.sum() + smooth)

'''
    input:预测的结果（>0.5判断后的）
    target:真实的标签
'''
def get_sens(input,target):
    smooth = 1e-5
    input_flat = input.view(-1)  # 拉平成一列，即第一维度是计算得来的
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat)

    result = (intersection.sum() + smooth)/(target_flat.sum() + smooth)

    return result

def get_ppv(input,target):
    smooth = 1e-5
    input_flat = input.view(-1)  # 拉平成一列，即第一维度是计算得来的
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat)

    result = (intersection.sum() + smooth) / (input_flat.sum() + smooth)

    return result

def ppv_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (output.sum() + smooth)

def f1score(output, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()
    f1 = f1_score(y_true=target, y_pred=output)
    return  f1

# 计算Hausdorff距离
# def computeQualityMeasures(output, target):
#     import SimpleITK as sitk
#     # output = output.unsqueeze(dim=1)
#     # target = target.unsqueeze(dim=1)
#
#     # output = (output > 0.5).float()
#     # target = (target > 0.5).float()
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#
#     # channel = output.shape[0]
#
#     # for i in range(channel):
#     # print(output[i].shape)
#     quality = dict()
#     output_ = sitk.GetImageFromArray(output, isVector=False)
#     target_ = sitk.GetImageFromArray(target, isVector=False)
#
#     hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
#     hausdorffcomputer.Execute(target_ > 0.5, output_ > 0.5)
#     quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
#     quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
#     print(quality["Hausdorff"])

    # return quality["Hausdorff"]


def get_sensitivity(output, target): # 求敏感度 se=TP/(TP+FN)
    SE = 0.

    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    # output = output.view(-1)  # 拉平成一列，即第一维度是计算得来的
    # target = target.view(-1)


    TP = 0
    FN = 0
    for i in range(0, len(output)):
        if output[i] == 1 and target[i] == 1:
            TP += 1
        if output[i] == 0 and target[i] == 1:
            FN += 1

    SE = float(TP) / (float(TP + FN) + 1e-6)  # 原本只用这一句
    return SE  #返回batch中所有样本的SE和


def get_specificity(output, target):#求特异性 sp=TN/(FP+TN)
    output.tolist(), target.tolist()
    output, target = np.array(output).flatten(), np.array(target).flatten()

    SP=0.# wfy
    # TN : True Negative
    # FP : False Positive
    TN = 0
    FP = 0
    for i in range(0, len(output)):
        if output[i] == 0 and target[i] == 0:
            TN += 1
        if output[i] == 1 and target[i] == 0:
            FP += 1


    SP = float(TN) / (float(TN + FP) + 1e-6)
    return SP




