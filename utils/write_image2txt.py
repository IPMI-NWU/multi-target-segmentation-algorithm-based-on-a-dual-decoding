import os

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

def signal_organ():
    for sub_set in set:
        for i in range(0, len(task)):
            path_img = os.path.join(root, sub_set, task[i], 'image')
            path_label = os.path.join(root, sub_set, task[i], label[i])

            file_list = get_file_list(path_img)  # 从imageX中获取图像
            print(file_list)

            for file_name in file_list:
                # if task[i] == '1Lung' and file_name[:5] in ['JPCLN', 'JPCNN']:
                #     continue
                line_img = os.path.join(path_img, file_name)
                line_label = os.path.join(path_label, file_name)

                line = line_img + '\t' + line_label
                print(line)

                write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)

def fun_mulitple_bone():
    set = ['train', 'val', 'test']
    task = ['pre_rib', 'post_rib', 'clavicle', ]  # 'all_bone',

    root = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset'  # 从这里获得image

    txt_path = r'/home/zdd2020/ZDD_paper/ZDD_part_data/dataset/bone/txt'
    mkdir(txt_path)

    for sub_set in set:
        path_img = os.path.join(root, sub_set, 'image')
        # path_label = os.path.join(root, sub_set, task[i], label[i])

        file_list = get_file_list(path_img)  # 从imageX中获取图像
        print(file_list)

        for file_name in file_list:
            line_img = os.path.join(path_img, file_name)
            path_label = ''
            for i in range(0, len(task)):
                if i == len(task) - 1:
                    path_label += os.path.join(root, sub_set, task[i], file_name)
                else:
                    path_label += os.path.join(root, sub_set, task[i], file_name) + '\t'

            line = line_img + '\t' + path_label
            print(line)

            write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)

def no_pingyi():
    set = ['train', 'val', 'test']
    task = ['all_bone', 'clavicle', 'post_rib', 'pre_rib']

    root = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset'  # 从这里获得image

    txt_path = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/txt_munit_new_pingyi'
    mkdir(txt_path)

    munit_root = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset/test_900000_delete'

    for sub_set in set:
        path_img = os.path.join(root, sub_set, 'image')
        # path_label = os.path.join(root, sub_set, task[i], label[i])

        file_list = get_file_list(path_img)  # 从imageX中获取图像
        print(file_list)

        for file_name in file_list:
            line_img = os.path.join(path_img, file_name)
            path_label = ''
            for i in range(0, len(task)):
                if i == len(task) - 1:
                    path_label += os.path.join(root, sub_set, task[i], file_name)
                else:
                    path_label += os.path.join(root, sub_set, task[i], file_name) + '\t'

            line = line_img + '\t' + path_label
            print(line)

            write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)

            # 只在训练时扩充，验证与测试时依然是真实标注的
            if sub_set == 'train':
                # munit 扩充的数据
                file_name_prefix = file_name.split('.')[0]
                temp_munit = os.path.join(munit_root, file_name_prefix)

                # 获取里面的文件
                munit_file_list = get_file_list(temp_munit)
                for munit_file_name in munit_file_list:
                    if munit_file_name == 'input.jpg':
                        continue
                    line_img = os.path.join(temp_munit, munit_file_name)  # 扩充的image图像
                    line = line_img + '\t' + path_label

                    write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)


def bone():
    # root = r'/home/zdd2020/ZDD_paper/ZDD_data_aug/dataset'  # 从这里获得image，这里有一折的 train\test\val
    root = r'../../dataset/bone/image'
    set = ['train', 'val', 'test']
    fold = '1'

    label_root = r'../../dataset/bone/mask'  # 原始图像的 label 路径，修正后的
    task = ['pre_rib', 'post_rib', 'clavicle', 'bone']

    txt_path = r'../../dataset/bone/txt_' + fold  # txt 保存的路径
    mkdir(txt_path)

    for sub_set in set:
        # 获取 train, val, test 集合的图像
        path_img = os.path.join(root, sub_set, 'image' + fold)

        file_list = get_file_list(path_img)  # 从imageX中获取图像
        print(file_list)

        for file_name in file_list:
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

def jsrt():
    root = r'../../dataset/jsrt/image_mask'  # 从这里获得image，这里有一折的 train\test\val
    set = ['train', 'validation', 'test']

    label_root = r'../../dataset/jsrt/image_mask'  # 原始图像的 label 路径，修正后的
    task = ['lung', 'heart', 'clavicle']

    txt_path = r'../../dataset/jsrt/txt_3organ_train158/'  # txt 保存的路径
    mkdir(txt_path)

    for sub_set in set:
        # 获取 train, val, test 集合的图像
        path_img = os.path.join(root, sub_set, 'image')

        file_list = get_file_list(path_img)  # 从imageX中获取图像
        print(file_list)

        for file_name in file_list:
            line_img = os.path.join(path_img, file_name)

            path_label = ''
            for i in range(0, len(task)):
                if i == len(task) - 1:
                    path_label += os.path.join(label_root, sub_set, task[i], file_name)
                else:
                    path_label += os.path.join(label_root, sub_set, task[i], file_name) + '\t'

            line = line_img + '\t' + path_label
            print(line)

            write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)

def mg():
    root = r'../../dataset/mg_square'  # 从这里获得image，这里有一折的 train\test\val
    set = ['train', 'val', 'test']

    txt_path = r'../../dataset/mg_square/txt_mg/'  # txt 保存的路径
    mkdir(txt_path)

    for sub_set in set:
        # 获取 train, val, test 集合的图像
        path_img = os.path.join(root, sub_set, 'image')

        file_list = get_file_list(path_img)  # 从imageX中获取图像
        print(file_list)

        for file_name in file_list:
            line_img = os.path.join(path_img, file_name)

            path_label = os.path.join(root, sub_set, 'lung', file_name)
            line = line_img + '\t' + path_label
            print(line)

            write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)

def vindr_ribcxr():
    root = r'../../dataset/VinDr_RibCXR_square'  # 从这里获得image，这里有一折的 train\test\val
    set = ['train', 'val']

    txt_path = r'../../dataset/VinDr_RibCXR_square/txt_VinDr_RibCXR/'  # txt 保存的路径
    mkdir(txt_path)

    for sub_set in set:
        # 获取 train, val, test 集合的图像
        path_img = os.path.join(root, sub_set, 'img')

        file_list = get_file_list(path_img)  # 从imageX中获取图像
        print(file_list)

        for file_name in file_list:
            line_img = os.path.join(path_img, file_name)

            path_label = os.path.join(root, sub_set, 'mask', file_name)
            line = line_img + '\t' + path_label
            print(line)

            write_line2txt(os.path.join(txt_path, sub_set + '.txt'), line)

if __name__ == '__main__':
    vindr_ribcxr()
