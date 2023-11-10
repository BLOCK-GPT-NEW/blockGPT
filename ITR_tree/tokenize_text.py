import re

####Tokenization部分####
def tokenize_text(text,node_tag):
    if isinstance(text, str):
        # 使用逗号和下划线作为分隔符
        tokens = re.split(r'[,]', text)
        # 仅保留长度大于1的单词
        tokens = [token.strip() for token in tokens if len(token) > 1]
        # 添加[START]和[END]标记
        tokens = ['[START]'] + [f'[{node_tag.upper()}]'] + tokens + ['[END]']
        
        # 对call_trace中in和out部分添加[OUTs]和[INs]标签
        if  node_tag == 'call':
            in_indices = [i for i, token in enumerate(tokens) if 'input_type' in token]
            out_indices = [i for i, token in enumerate(tokens) if 'output_type' in token]
            
            # 在第一个 input_type 之前插入 [INs]
            if in_indices:
                tokens = tokens[:in_indices[0]] + ['[INs]'] + tokens[in_indices[0]:]
            
            # 在第一个 output_type 之前插入 [OUTs]
            if out_indices:
                tokens = tokens[:out_indices[0] + 1] + ['[OUTs]'] + tokens[out_indices[0] + 1:]
        
        return tokens
    else:
        return text

def tokenize_tree(root):
    # 递归地对树的每个节点进行标记化
    root.data = tokenize_text(root.data,root.tag)
    for child, edge_data in root.children:
        child.data = tokenize_text(child.data,child.tag)
        tokenize_tree(child)

    