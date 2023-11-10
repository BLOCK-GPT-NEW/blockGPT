def traverse_tree(node, node_list):
    # 将当前节点加入列表
    node_list.append(node)
    # 递归遍历子节点
    #for child, _ in node.children:
    #   traverse_tree(child, node_list)