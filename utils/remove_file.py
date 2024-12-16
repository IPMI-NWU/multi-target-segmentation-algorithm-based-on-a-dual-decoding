import shutil
import os

'''
    作用：可以根据文件名对文件进行移动
'''

# 获取文件夹中的文件名
def get_file_list(path):
    pathDir = os.listdir(path)
    file_list = []

    for file in pathDir:
        file_list.append(file)
    return file_list

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

def remove_file(file_list, target_file, src_path, target_path, split_str='.'):
    for file in file_list:
        temp = file.split(split_str)[0]
        if temp in target_file:
            shutil.move(os.path.join(src_path, file), target_path)

if __name__ == "__main__":
    # test
    # file_list = ['00036', '00078', '00027', '00043', '00076', '00081', '00086',
    #              '00005', '00007', '00008', '00025', '00052', '00077', '00067',
    #              '00087', '00039', '00032', '00082', '00064', '00001', '00016',
    #              '00055']


    # val
    file_list = ['00024', '00029', '00033', '00037', '00038', '00042', '00045',
                 '00049', '00050', '00053']

    dir_set = ['image', 'all_bone', 'clavicle', 'post_rib', 'pre_rib']
    # index = 2

    for index in range(0, len(dir_set)):
        label_path = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/train/' + dir_set[index]

        file_label = get_file_list(label_path)

        target_label_path = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/val/' + dir_set[index]
        mkdir(target_label_path)


        remove_file(file_label, file_list, label_path, target_label_path, split_str='.')
