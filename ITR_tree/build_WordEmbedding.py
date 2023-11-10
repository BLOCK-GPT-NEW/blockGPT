import torch
import math
import torch.nn as nn

#递归调用转化结点-生成初始的work embedding
def build_WordEmbedding(root, vocabulary, vocab_size, d_model):
    embedding_layer = WordEmbedding(vocab_size, d_model)
    tensor_from_add = torch.rand(1, d_model).unsqueeze(0)  # 对不同from要加上的形状为 (1, 512) 的张量
    tensor_to_add = -tensor_from_add                       # 对不同to要加上的张量#在此处对其加上tree position信息
    tree_position_embedding = build_tree_position_embedding(root,d_model)   #生成对应的tree_position
    tokens = root.data
    token_indices = torch.tensor([[vocabulary.get_index(token) for token in tokens]], dtype=torch.long)
    embeddings = embedding_layer(token_indices)
    root.embedding = embeddings     ##此处完成基础的token embedding
    #获取动态生成的position embedding
    max_len = len(tokens)
    position_embedding = generate_position_embedding(max_len, d_model)
    #将词向量和 position embedding 相加
    embeddings = embeddings + position_embedding
    root.embedding = embeddings     ##此处完成加上root信息后的的token embeddin
    ##在此处对其加上src信息
    if(root.tag == 'call'):
        embeddings[0, 2, :] +=  tensor_from_add.squeeze().expand_as(embeddings[0, 2, :])
        embeddings[0, 3, :] +=  tensor_to_add.squeeze().expand_as(embeddings[0, 3, :])
    root.embedding = embeddings     #此处完成了对call trace中的form和to加上信息的操作
    root.embedding = root.embedding.reshape(-1, root.embedding.size(-1))
    def trave(root, vocabulary, vocab_size, d_model):
        for child, edge_data in root.children:
            tokens = child.data
            token_indices = torch.tensor([[vocabulary.get_index(token) for token in tokens]], dtype=torch.long)
            embeddings = embedding_layer(token_indices)
            child.embedding = embeddings    #此处完成基础的token embedding
            #获取动态生成的position embedding
            max_len = len(tokens)
            position_embedding = generate_position_embedding(max_len, d_model)
            #将词向量和 position embedding 相加
            embeddings = embeddings + position_embedding
            child.embedding = embeddings     ##此处完成加上root信息后的的token embeddin
            ##在此处对其加上src信息
            if(child.tag == 'call'):
                embeddings[0, 2, :] += tensor_from_add.squeeze().expand_as(embeddings[0, 2, :])
                embeddings[0, 3, :] += tensor_to_add.squeeze().expand_as(embeddings[0, 3, :])
            child.embedding = embeddings
            #加上tree position
            child.embedding += tree_position_embedding[child.deep,:].unsqueeze(0)
            child.embedding = child.embedding.reshape(-1, child.embedding.size(-1))
            trave(child, vocabulary, vocab_size, d_model)
    trave(root, vocabulary, vocab_size, d_model)

#生成动态的position embedding
def generate_position_embedding(seq_len, d_model):
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

#转化为词向量
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(WordEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed = self.embedding

    def forward(self, x):
        return self.embed(x) * (self.d_model ** 0.5)
    
def build_tree_position_embedding(root, d_model):
    max_depth = depth(root)
    position_embedding = torch.zeros(max_depth+1, d_model)
    for i in range(max_depth):
        for j in range(d_model):
            position_embedding[i][j] = math.sin(i / (10000 ** (2 * j / d_model)))
    return position_embedding

#计算节点的相对深度
def depth(node):
    if not node:
        return 0
    d = 0
    for child, edge_data in node.children:
        d = max(d, depth(child) + 1)
    return d