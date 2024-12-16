from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

import random
import cv2
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image
from torchvision.transforms import RandomAffine
# from torchvision.transforms.functional import InterpolationMode

'''
    自己写的工具类，可以对image与mask做相同的操作
'''

def affine(image, shear):
    random_affine = RandomAffine(degrees=0, translate=None, scale=None, shear=shear, resample=Image.BILINEAR)
    return random_affine(image)

# 旋转
def transform_rotate(image, mask):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([0, 36])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle, expand=True)

    mask_result = []

    for i in range(len(mask)):
        mask_result.append(mask[i].rotate(angle, expand=True))

    # image = tf.to_tensor(image)
    # mask = tf.to_tensor(mask)
    return image, mask_result

# 错切
def transform_shear(image, mask, scale=0.05):
    mask_result = []

    w, h = image.size
    # 选一个最小值
    temp = w if w < h else h
    # angle = transforms.RandomRotation.get_params([-int(temp*scale), int(temp*scale)])
    angle = transforms.RandomRotation.get_params([-10, 10])
    # 水平错切
    if random.random() > 0.5:
        image = affine(image, shear=(angle, angle, 0, 0))
        for i in range(len(mask)):
            mask_result.append(affine(mask[i], shear=(angle, angle, 0, 0)))
    # 垂直错切
    else:
        image = affine(image, shear=(0, 0, angle, angle))
        for i in range(len(mask)):
            mask_result.append(affine(mask[i], shear=(0, 0, angle, angle)))

    return image, mask_result

# 水平/垂直翻转
# 该实验中，只进行水平翻转
def transform_flip(image, mask):
    mask_result = []
    if random.random() > 0.5:
        image = tf.hflip(image)
        for i in range(len(mask)):
            mask_result.append(tf.hflip(mask[i]))

        return image, mask_result
    else:
        return image, mask
    # else:
    #     image = tf.vflip(image)
    #     for i in range(len(mask)):
    #         mask[i] = tf.vflip(mask[i])



# 水平移动
# padding 为int/tuple，为一个时，上下左右都填充，为两个时分别用于填充left/right和top/bottom,
# 为4时，分别用来填充left,top,right,和bottom
def transform_translate_horizontal(image, mask, scale=0.05):
    # 获取image的长宽
    w, h = image.size

    # 如果随机数大于0.5，则裁剪左边，否则，裁剪右边
    mask_pad = []
    if random.random() > 0.5:
        # 留下的是左边，说明右部分被裁剪掉了，向右平移，左边填充 0
        image = tf.crop(image, top=0, left=0, height=h, width=w - int(w*scale))  # 25.6，
        image_pad = tf.pad(image, padding=[int(w * scale), 0, 0, 0], fill=0) # padding: left top right bottom


        for i in range(len(mask)):
            mask_temp = tf.crop(mask[i], top=0, left=0, height=h, width=w - int(w*scale))
            mask_pad.append(tf.pad(mask_temp, padding=[int(w * scale), 0, 0, 0], fill=0))

    else:
        # 左部分被裁剪掉了，向左平移，右边填充 0
        image = tf.crop(image, top=0, left=int(w*scale), height=h, width=w - int(w*scale))
        image_pad = tf.pad(image, padding=[0, 0, int(w*scale), 0], fill=0)  # padding: left top right bottom

        for i in range(len(mask)):
            mask_temp = tf.crop(mask[i], top=0, left=w*scale, height=h, width=w - int(w*scale))
            mask_pad.append(tf.pad(mask_temp, padding=[0, 0, int(w*scale), 0], fill=0))

    return image_pad, mask_pad

# 上下平移
def transform_translate_vertical(image, mask, scale=0.05):
    # 获取image的长宽
    w, h = image.size
    # 如果随机数大于0.5，则裁剪下边，否则，裁剪上边
    mask_pad = []
    if random.random() > 0.5:
        # 留下的是下边，说明上部分被裁剪掉了，向上平移，下边填充 0
        image = tf.crop(image, top=int(h*scale), left=0, height=h - int(h*scale), width=w)
        image_pad = tf.pad(image, padding=[0, 0, 0, int(h * scale)], fill=0)

        for i in range(len(mask)):
            mask_temp = tf.crop(mask[i], top=int(h*scale), left=0, height=h - int(h*scale), width=w)
            mask_pad.append(tf.pad(mask_temp, padding=[0, 0, 0, int(h * scale)], fill=0))

    else:
        # 留下的是上边，说明下部分被裁剪掉了，向下平移，上边填充 0
        image = tf.crop(image, top=0, left=0, height=h - int(h*scale), width=w)
        image_pad = tf.pad(image, padding=[0, int(h*scale), 0,  0], fill=0)

        for i in range(len(mask)):
            mask_temp = tf.crop(mask[i], top=0, left=0, height=h - int(h*scale), width=w)
            mask_pad.append(tf.pad(mask_temp, padding=[0, int(h*scale), 0,  0], fill=0))

    return image_pad, mask_pad


if __name__ == '__main__':
    img_path = r'/home/zdd2020/Desktop/xyl/dataset/jsrt_172_24_50/train/image/JPCLN003.png'
    label_path = r'/home/zdd2020/Desktop/xyl/dataset/jsrt_172_24_50/train/lung/JPCLN003.png'
    img = Image.open(img_path).convert('L')
    label = Image.open(label_path).convert('L')

    img = img.resize((512, 512))
    label = label.resize((512, 512))

    mask = [label]

    print(img.size)
    print(label.size)

    plt.imshow(img, cmap='gray')
    plt.show()

    plt.imshow(label, cmap='gray')
    plt.show()


    image_pad, mask_pad = transform_rotate(img, mask)


    plt.imshow(image_pad, cmap='gray')
    plt.show()
    plt.imshow(mask_pad[0], cmap='gray')
    plt.show()

    plt.imshow(img, cmap='gray')
    plt.show()
    plt.imshow(mask[0], cmap='gray')
    plt.show()



    # img_tensor, label_tensor = transform_rotate(img, label)
    #
    # img_rotate = transforms.ToPILImage()(img_tensor).convert('L')
    # label_rotate = transforms.ToPILImage()(label_tensor).convert('L')
    #
    # plt.imshow(img_rotate)
    # plt.show()
    #
    # plt.imshow(label_rotate)
    # plt.show()
