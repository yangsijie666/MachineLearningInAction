# coding=utf-8
from numpy import *


# 4-5-1 词表转为词向量
def load_data_set():
    # 句子转为词
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him']
    ]
    # 类标签，0为正常，1为侮辱性文字
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        # document 为一行文字拆成的以单词为元素的list
        vocab_set = vocab_set | set(document)  # 两个集合合并
    # vocab_set 为 data_set 中的所有不重复的词条（整个文章所有词条）
    return list(vocab_set)


def set_of_words_2_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)  # 创建一个元素为0的向量（一行*len(vocab_list)列）
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word: %s is not in my vocabulary!" % word
    # 返回的列表中的元素含义：文章的所有单词中，有哪些存在于该行中（这里的所有单词即想要检查的单词）
    # [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    return return_vec


# 测试词表转为词向量
def test_set_of_words_2_vec():
    list_of_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_of_posts)
    print my_vocab_list
    print set_of_words_2_vec(my_vocab_list, list_of_posts[0])
    print set_of_words_2_vec(my_vocab_list, list_of_posts[1])


# 4-5-2 朴素贝叶斯分类器训练函数
def train_n_b_0(train_matrix, train_category):  # train_matrix为二维矩阵，一行表示关注的词汇表中各单词在这句话中出现的次数；train_category为类标签，该句话属于哪一类
    num_train_docs = len(train_matrix)  # 行数（样本数，一行为一句话，即一个测试样本）
    num_words = len(train_matrix[0])  # 词汇表长度，即关注的词汇数
    p_abusive = sum(train_category) / float(num_train_docs)  # 侮辱性类比例（侮辱类为1，所以可以这么用）；因为这里是二分类，所以 p(c0)=1-p(c1)
    # 初始化概率 p(w|c0) 和 p(w|c1)
    p_0_num = zeros(num_words)
    p_1_num = zeros(num_words)
    p_0_denom = 0.0
    p_1_denom = 0.0
    # 计算 p(w|c0) 和 p(w|c1)
    for i in range(num_train_docs):  # 逐行计算（逐个样本）
        if train_category[i] == 1:  # 侮辱性
            p_1_num += train_matrix[i]  # 用于计算侮辱性类别中，各单词出现的总数
            p_1_denom += sum(train_matrix[i])  # sum(train_matrix[i]) 为单行中所有出现的单词的总数(因为单行样本中，1表示该词在该句话中出现了)
        else:
            p_0_num += train_matrix[i]
            p_0_denom += sum(train_matrix[i])
    # p_1_num 为一个1*len的向量，每个元素表示在类别1中各单词出现的次数；p_1_denom为float类型，表示在类别1中，所有单词出现的次数总和。因此前者中各元素与后者相除，可以求得对应单词在类别1
    # 中出现的频率，即 p(wi|c1)
    p_1_vect = p_1_num / p_1_denom
    p_0_vect = p_0_num / p_0_denom
    return p_0_vect, p_1_vect, p_abusive    # p_0_vect即[p(w0|c0),p(w1|c0),...]；p_abusive为一个float类型，即p(c1)


# 测试朴素贝叶斯分类器训练函数
def test_train_n_b_0():
    list_of_posts, list_classes = load_data_set()
    # 需要将list_of_posts转换为我们关注的词汇表对应的词条向量
    my_vocab_list = create_vocab_list(list_of_posts)    # 构建我们关注的词汇表
    # 将构建好的词条向量依次加入训练矩阵中
    train_matrix = []
    for post_in_doc in list_of_posts:
        train_matrix.append(set_of_words_2_vec(my_vocab_list, post_in_doc))
    # 通过训练函数计算 p(w|c0) 和 p(w|c1) 向量，以及 p(c0) 和 p(c1)
    p_0_v, p_1_v, p_abusive = train_n_b_0(train_matrix, list_classes)
    p_imabusive = 1 - p_abusive
    print my_vocab_list
    print p_0_v
    print p_1_v
    print p_abusive
    print p_imabusive


if __name__ == "__main__":
    test_train_n_b_0()
