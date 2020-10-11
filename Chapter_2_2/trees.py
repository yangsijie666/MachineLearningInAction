# coding=utf-8
from math import log
import operator


# 3-1-1 计算给定数据集的香农熵（信息熵，衡量数据类别无序程度，越大，越无序）
def calc_shannon_ent(data_set):
    num_entries = len(data_set)     # 样本数
    label_counts = {}   # 类别-数量 map
    # 填充类别-数量 map
    for feat_vec in data_set:
        current_label = feat_vec[-1]    # feat_vec，即数据集中的一行，即一个样本，最后一列为标签（类别）
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # 计算香农熵
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries   # prob 为对应类别占总数比
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


# 测试香农熵计算用的简单数据集
def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


# 测试香农熵计算
def test_cal_shannon_ent():
    my_dat, labels = create_data_set()
    print calc_shannon_ent(my_dat)


# 3-1-2 按照给定特征划分数据集（将根据axis特征=value对数据集进行划分，抽取出符合这个要求的样本集）
def split_data_set(data_set, axis, value):  # axis即特征类别，value即特征值
    ret_data_set = []   # 用于保存数据集划分后的结果
    for feat_vec in data_set:   # feat_vec即一个样本
        # 抽取符合axis特征值=value的样本
        if feat_vec[axis] == value:
            # 样本将保留除axis以外的其他特征
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])  # a = [1, 2], b = [2, 3], a.extend(b) => a = [1, 2, 2, 3]
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


# 测试数据集划分
def test_split_data_set():
    my_dat, labels = create_data_set()
    print split_data_set(my_dat, 0, 1)
    print split_data_set(my_dat, 0, 0)


# 3-1-3 选择最好的数据集划分方式
def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1     # data_set[0] 为一行，取len则为列数，最后一列为标签，所以需要-1获取特征数
    base_ent = calc_shannon_ent(data_set)   # 基础熵（肯定最大，因为此时最数据集最无序，类别最多）
    best_info_gain = 0.0    # 用于保存最优的信息增益
    best_feature = -1   # 用于保存选择的最优特征
    # 依次尝试根据每一种特征分类的结果，选最优
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]    # 取所有样本的第i个特征值作为一个新列表
        unique_vals = set(feat_list)    # 对feat_list去重
        new_ent = 0.0
        for value in unique_vals:
            # 根据第i个特征值=value对数据集进行抽取，并计算抽取出的数据集的信息熵
            sub_data_set = split_data_set(data_set, i, value)
            # 下面这两步根据公式推导可以发现，计算得到的是信息熵，p(x)分母变成了len(data_set)，分子不变
            prob = len(sub_data_set) / float(len(data_set))
            new_ent += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_ent - new_ent      # info_gain为以当前特征抽取出来的子集的信息增益；信息增益为原数据集的信息熵与划分后的数据集信息熵的差值
        if info_gain > best_info_gain:  # 信息增益越大，表示数据变得越有序，对应特征越优
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# 测试数据集最优划分方式下的特征
def test_choose_best_feature_to_split():
    my_dat, labels = create_data_set()
    print choose_best_feature_to_split(my_dat)
    print my_dat


# 没有特征可以使用，但剩下的样本仍具有不同的类标签，需采用多数表决方法决定叶子节点分类
def majority_cnt(class_list):       # 输入为类标签列表
    class_count = {}    # 类-数量map
    # 构建 类-数量map
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)      # 根据数量进行排序，最前面的为数量最多的类别
    return sorted_class_count[0][0]     # 返回数量最多的类别


# 3-1-4 创建决策树
def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]      # class_list为类标签
    if class_list.count(class_list[0]) == len(class_list):  # class_list中仅剩唯一的类标签，可直接返回该标签
        return class_list[0]
    if len(data_set[0]) == 1:   # 若仅剩最后一个标签了，采用多数表决法，选择数量最多的类标签返回
        return majority_cnt(class_list)
    # 准备递归
    best_feat = choose_best_feature_to_split(data_set)  # 选出最优特征
    best_feat_label = labels[best_feat]
    # 开始构建当前节点的决策树
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])  # 因为递归子节点中会抽取掉最优特征，所以相应的最优特征对应的类标签也应该抽取掉
    feat_values = [example[best_feat] for example in data_set]  # 各个样本在最优特征对应下的特征值
    unique_feat_values = set(feat_values)   # 下面开始根据最优特征对样本进行分类，并递归构建子节点（最优特征对应当前子节点）
    for value in unique_feat_values:
        sub_labels = labels[:]      # 复制一个labels列表，防止递归子节点中del类标签时产生误伤
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


# 测试构建决策树
def test_create_tree():
    my_dat, labels = create_data_set()
    my_tree = create_tree(my_dat, labels)
    print my_tree


if __name__ == '__main__':
    test_create_tree()
