import torch.nn as nn
import time
import torch
# Transformer 编码器模型
class TransformerEncoderModel(nn.Module):
    def __init__(self,vocab_size,d_model, hidden_size, num_heads, num_layers):
        super(TransformerEncoderModel, self).__init__()
        
        self.encoder = nn.TransformerEncoder( nn.TransformerEncoderLayer(d_model, num_heads, hidden_size) , num_layers)  

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x,src_mask):
        
        x = self.encoder(x,src_mask)
        x = self.output_layer(x)
        return x

def generate_square_subsequent_mask(sz):
    #"""生成一个负无穷大的上三角矩阵，对角线上元素为 0。"""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)