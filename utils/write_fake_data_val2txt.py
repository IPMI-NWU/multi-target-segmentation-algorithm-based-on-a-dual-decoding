import os
import random

'''将4折交叉验证的图像数据的路径写入txt'''

# 获取文件夹中的文件名
def get_file_list(path):
    pathDir = os.listdir(path)
    file_list = []

    for file in pathDir:
        file_list.append(file)
    return file_list


# 写入txt中
def write_line2txt(path, line):
    with open(path, 'a+', encoding='utf-8') as f:
        f.write(line + '\n')

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

def fake_data_lihansheng():
    root = r'../../dataset/bone/image'  # 从这里获得第几折的image
    set = ['train', 'val', 'test']

    label_root = r'../../dataset/bone/mask'  # 原始图像的 label 路径，修正后的
    task = ['bone', 'clavicle', 'post_rib', 'pre_rib']

    fold = '3'

    txt_path = r'../../dataset/bone_fake_data/txt_lihangsheng_fake_data_20/' + 'fold' + fold  # txt 保存的路径
    mkdir(txt_path)

    # fake_data
    fake_data_root = r'../../dataset/bone_fake_data'
    fake_data_list = ['20']  # '10', '15', 'fake_val', '25'
    # fake_data 中的图像文件名
    fake_data_file_list_num = get_file_list(os.path.join(fake_data_root, fake_data_list[0]))
    fake_data_file_list_val = get_file_list(os.path.join(fake_data_root, fake_data_list[-1]))

    # 从第 n 折中获取文件名
    for sub_set in set:
        # 获取 train, val, test 集合的图像
        path_img = os.path.join(root, sub_set, 'image' + fold)

        file_list = get_file_list(path_img)  # 从imageX中获取图像
        print(file_list)

        for file_name in file_list:
            # real data
            line_img = os.path.join(path_img, file_name)

            path_label = ''
            for i in range(0, len(task)):
                if i == len(task) - 1:
                    path_label += os.path.join(label_root, task[i], file_name)
                else:
                    path_label += os.path.join(label_root, task[i], file_name) + '\t'

            line = line_img + '\t' + path_label
            print(line)

            write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)

            if sub_set == 'train' or sub_set == 'val':
                if file_name in fake_data_file_list_num:
                    for fake_dir in fake_data_list:
                        if fake_dir != fake_data_list[-1]:
                            line_img = os.path.join(fake_data_root, fake_dir, file_name)
                            line = line_img + '\t' + path_label
                            print(line)

                            write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)

                if file_name in fake_data_file_list_val:
                    line_img = os.path.join(fake_data_root, fake_data_list[-1], file_name)
                    line = line_img + '\t' + path_label
                    print(line)

                    write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)

if __name__ == '__main__':
    # 合并两个 txt 中的内容
    fake_data_lihansheng()





