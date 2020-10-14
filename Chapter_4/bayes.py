# coding=utf-8
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
        vocab_set = vocab_set | set(document)   # 两个集合合并
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


if __name__ == "__main__":
    test_set_of_words_2_vec()
