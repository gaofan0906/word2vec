import os
import re
import hashlib
import numpy as np


# 字符串清理
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 从样本中获得词库
def get_examples_from_dir(data_dir, max_length, is_line_as_word=False):
    examples = []
    if not os.path.isdir(data_dir):
        return examples
    # fname文件夹下名称列表
    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        f = open(full_path, "r")
        data = f.read()
        line_num = len(data.split("\n"))
        if line_num < 5:
            continue
        if not is_line_as_word:
            examples.append(data.strip())
        else:
            # is_line_as_word为true 分词
            lines = data.split("\n")
            # replace each line as md5 每一行用md5算法加密 十六进制返回摘要
            words = [hashlib.md5(line.encode("utf-8")).hexdigest() for line in lines]
            examples.append(" ".join(words[:max_length]))
        f.close()

    return examples


def get_example_filenames_from_dir(data_dir, max_length, is_line_as_word=False):
    examples = []
    filenames = []
    if not os.path.isdir(data_dir):
        return examples, filenames
    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        f = open(full_path, "r")
        data = f.read()
        line_num = len(data.split("\n"))
        new_lines = []
        for line in data.split("\n"):
            if not line.startswith("#"):
                new_lines.append(line)
        data = "\n".join(new_lines)
        if line_num < 5:
            continue
        filenames.append(full_path)
        if not is_line_as_word:
            examples.append(data.strip())
        else:
            lines = data.split("\n")
            # replace each line as md5
            words = [hashlib.md5(line).hexdigest() for line in lines]
            examples.append(" ".join(words[:max_length]))
        f.close()

    return examples, filenames


# 加载数据和标签
def load_data_and_labels(data_dirs, max_document_length, is_line_as_word):
    '''

    :param data_dirs:
    :param max_document_length:
    :param is_line_as_word:
    :return:
    '''
    x_text = []
    y = []
    # 将二分类标签[[0,1],[1,0]]
    labels = np.eye(len(data_dirs), dtype=np.int32).tolist()
    for i, data_dir in enumerate(data_dirs):
        # 每次读取pos、neg文件夹
        # examples是正样本与负样本的词库，md5加密过得。
        examples = get_examples_from_dir(data_dir, max_document_length, is_line_as_word)
        x_text += [clean_str(sent) for sent in examples]
        y += [labels[i]] * len(examples)
    y = np.array(y)
    # ==================debug
    print("=============debug load_data_label_and_filenames x len")
    print(len(x_text))
    print(x_text)
    print(y)
    np.save("data/save.txt", zip(x_text, y))
    return [x_text, y]


def load_data_label_and_filenames(data_dirs, max_document_length, is_line_as_word):
    x_text = []
    y = []
    fnames = []
    labels = np.eye(len(data_dirs), dtype=np.int32).tolist()
    for i, data_dir in enumerate(data_dirs):
        examples, fname = get_example_filenames_from_dir(data_dir, max_document_length, is_line_as_word)
        x_text += [clean_str(sent) for sent in examples]
        y += [labels[i]] * len(examples)
        fnames += fname
    y = np.array(y)
    return x_text, y, fnames


def data_iter(data, batch_size, ecoph_num, shuffle=True):
    # data = np.array(data)
    data = np.array(list(data))
    # print("======================datta==========================")
    # print(data)
    # <zip object at 0x000001CCB0B3BDC8>
    print("==============enter data_iter:")
    print(data)
    print("==============list data:")
    print(len(data))
    # data_size = sum(1 for _ in list(data))
    # print(data_size)
    # print(len(list(data)))
    # print(len(data))
    # data_size = len(data)
    # data_size = sum(1 for _ in data)
    data_size = len(data)
    # data_size = len(list(data))
    batch_count = int((data_size - 1) / batch_size) + 1
    print("data_size: {}".format(data_size))
    print("batch_count: {}".format(batch_count))
    for e in range(ecoph_num):
        shuffle_data = data
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]

        for i in range(batch_count):
            yield shuffle_data[i * batch_size: (i + 1) * batch_size]
