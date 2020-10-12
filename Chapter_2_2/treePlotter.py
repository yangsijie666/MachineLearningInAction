# coding=utf-8
# 3-2-1 使用文本注解绘制树节点
import matplotlib.pyplot as plt

# 定义决策节点、叶子节点、箭头的格式
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    # 绘制节点及指向该点的线
    # node_txt为节点内容；center_pt为节点坐标；parent_pt为线的起始坐标
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


def create_plot(in_tree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    # plot_node('DecisionNode', (0.5, 0.1), (0.1, 0.5), decision_node)
    # plot_node('LeafNode', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


# 3-2-2 获取决策树的叶子节点数目（树宽度）和树的层数（树高度）
def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = my_tree.keys()[0]  # 获取第一个特征值（或类标签）
    second_dict = my_tree[first_str]  # 它的value对应了一个集合，为一个子树
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            # 如果子节点仍然是一棵树，则递归计算
            num_leafs += get_num_leafs(second_dict[key])  # 递归计算
        else:
            # 如果子节点是类标签，则+1即可
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = my_tree.keys()[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            # 如果子节点仍然是一棵树，则递归计算
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            # 如果子节点是类标签，则当前深度为1
            this_depth = 1
        # 取以当前节点为根的树的最大深度
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


# 为了便于获取测试数据写的获取树的函数
def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                     ]
    return list_of_trees[i]


# 测试计算树宽度和高度函数
def test_get_tree_leafs_depth():
    my_tree = retrieve_tree(0)
    print get_num_leafs(my_tree)
    print get_tree_depth(my_tree)


# 3-2-3 绘制树
def plot_mid_text(cntr_pt, parent_pt, txt_string):  # cntr_pt子节点坐标，parent_pt父节点坐标，txt_string文本信息
    # 在父子节点中间添加文本
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):  # my_tree树结构，parent_pt根节点坐标，node_txt节点文本
    # 绘制树
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = my_tree.keys()[0]
    # 计算线的箭头坐标(约等于子节点坐标)，并绘制
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)

    second_dict = my_tree[first_str]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d


# 测试绘制树
def test_plot_tree():
    my_tree = retrieve_tree(0)
    create_plot(my_tree)
    my_tree['no surfacing'][3] = 'maybe'
    create_plot(my_tree)


if __name__ == '__main__':
    test_plot_tree()
