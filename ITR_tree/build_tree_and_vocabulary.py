from build_ITR_tree import build_ITR_tree
from build_vocabulary import build_vocabulary
from tokenize_text import tokenize_tree

def build_tree_and_vocabulary(Seqsstate_1, Seqslog_1, Seqscall_1,vocabulary):
    root_node = build_ITR_tree(Seqsstate_1, Seqslog_1, Seqscall_1)
    #print(f"孩子节点：{len(root_node.children)}")
    tokenize_tree(root_node)
    build_vocabulary(root_node, vocabulary)
    return root_node, vocabulary
